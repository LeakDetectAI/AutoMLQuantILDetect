import logging

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state
from tabpfn import TabPFNClassifier

from autoqild.automl.automl_core import AutomlClassifier
from ..utilities import create_dimensionality_reduction_model


class AutoTabPFNClassifier(AutomlClassifier):
    """
    AutoTabPFNClassifier is an AutoML model wrapper designed to work with the TabPFN (Tabular Prior-based
    Fully Bayesian Network) for classification tasks.

    This class provides a high-level interface to automatically build, train, and evaluate a
    TabPFN model on tabular data. It supports various configurations and allows for dimensionality
    reduction if the number of features exceeds a specified threshold. The class is equipped to
    handle different feature reduction techniques and can operate on both CPU and GPU, depending on
    the available resources.

    Parameters
    ----------
    n_features : int
        The number of features in the input data.

    n_classes : int
        The number of classes in the classification task.

    n_ensembles : int, default=100
        The number of ensemble configurations used by the TabPFN model.

    n_reduced : int, default=20
        The number of features to reduce to if `n_features` exceeds 50.

    reduction_technique : str, optional, default='select_from_model_rf'
        Technique to use for feature reduction, provided by scikit-learn.
        Must be one of:

        - 'recursive_feature_elimination_et': Uses ExtraTreesClassifier to recursively remove features and build a model.
        - 'recursive_feature_elimination_rf': Uses RandomForestClassifier to recursively remove features and build a model.
        - 'select_from_model_et': Meta-transformer for selecting features based on importance weights using ExtraTreesClassifier.
        - 'select_from_model_rf': Meta-transformer for selecting features based on importance weights using RandomForestClassifier.
        - 'pca': Principal Component Analysis for dimensionality reduction.
        - 'lda': Linear Discriminant Analysis for separating classes.
        - 'tsne': t-Distributed Stochastic Neighbor Embedding for visualization purposes.
        - 'nmf': Non-Negative Matrix Factorization for dimensionality reduction.

    base_path : str or None, default=None
        The path where the trained model and other outputs are saved. If None, no model is saved.

    random_state : int or None, default=None
        Seed for random number generation to ensure reproducibility.

    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    n_features : int
        The number of features in the input data.

    n_classes : int
        The number of classes in the classification task.

    n_ensembles : int
        The number of ensemble configurations used by the TabPFN model.

    n_reduced : int
        The number of features to reduce to if `n_features` exceeds 50.

    reduction_technique : str
        The technique used for feature reduction.

    base_path : str or None
        The path where the trained model and other outputs are saved.

    random_state : int or None
        Seed for random number generation to ensure reproducibility.

    device : str
        The device used for computation, either 'cpu' or 'cuda' depending on the availability of a GPU.

    selection_model : object or None
        The model used for dimensionality reduction. Initialized during the first call to `transform`.

    logger : logging.Logger
        Logger object used for logging messages and errors.

    model : TabPFNClassifier or None
        The TabPFN model object, initialized after fitting.

    __is_fitted__ : bool
        Flag indicating whether the dimensionality reduction model is fitted.

    Private Methods
    ---------------
    __clear_memory__()
        Clear memory to release resources by torch.

    __transform__(X, y=None):
        Transform and reduce the feature matrix with 'n_features' features, using the specified reduction
        technique to the feature matrix with 'n_reduced' features.
    """
    def __init__(self, n_features, n_classes, n_ensembles=100, n_reduced=20, reduction_technique='select_from_model_rf',
                 base_path=None, random_state=None, **kwargs):
        self.n_features = n_features
        self.n_classes = n_classes
        self.logger = logging.getLogger(name=AutoTabPFNClassifier.__name__)
        self.random_state = check_random_state(random_state)
        self.n_reduced = n_reduced
        self.reduction_technique = reduction_technique
        self.selection_model = None
        self.__is_fitted__ = False

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.logger.info(f"Device {self.device}")
        self.n_ensembles = n_ensembles
        self.model = None
        self.base_path = base_path

    def __transform__(self, X, y=None):
        """
        Transform and reduce the feature matrix with 'n_features' features, using the specified reduction
        technique to the feature matrix with 'n_reduced' features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,), optional
            Target vector.

        Returns
        -------
        X : array-like of shape (n_samples, n_reduced)
            Transformed feature matrix.
        """
        self.logger.info(f"Before transform n_instances {X.shape[0]} n_features {X.shape[-1]}")
        if y is not None:
            classes, n_classes = np.unique(y, return_counts=True)
            self.logger.info(f"Classes {classes} No of Classes {n_classes}")
        if not self.__is_fitted__:
            if self.n_features != X.shape[-1]:
                raise ValueError(f"Dataset passed does not contain {self.n_features}")
            if y is not None:
                if self.n_classes != len(np.unique(y)):
                    raise ValueError(f"Dataset passed does not contain {self.n_classes}")
            self.selection_model = create_dimensionality_reduction_model(reduction_technique=self.reduction_technique,
                                                                         n_reduced=self.n_reduced)
            self.logger.info(f"Creating the model")
            if self.n_features > 50 and self.n_reduced < self.n_features:
                self.logger.info(f"Transforming and reducing the {self.n_features} features to {self.n_reduced}")
                self.selection_model.fit(X, y)
                X = self.selection_model.transform(X)
                self.__is_fitted__ = True
        else:
            if self.n_features > 50 and self.n_reduced < self.n_features:
                X = self.selection_model.transform(X)
        self.logger.info(f"After transform n_instances {X.shape[0]} n_features {X.shape[-1]}")
        return X

    def fit(self, X, y, **kwd):
        """
        Fit the TabPFN model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        **kwd : dict, optional
            Additional keyword arguments."""
        X = self.__transform__(X, y)
        params = dict(device=self.device, base_path=self.base_path, N_ensemble_configurations=self.n_ensembles)
        if self.base_path is not None:
            params['base_path'] = self.base_path

        self.model = TabPFNClassifier(**params)
        self.model.fit(X, y, overwrite_warning=True)
        self.__clear_memory__()
        self.logger.info("Fitting Done")

    def predict(self, X, verbose=0):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels."""
        p = self.predict_proba(X, verbose=0)
        y_pred = np.argmax(p, axis=-1)
        self.logger.info("Predict Done")
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        """
        Compute the balanced accuracy score for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            True labels.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        acc : float
            Balanced accuracy score."""
        y_pred = self.predict(X)
        acc = balanced_accuracy_score(y, y_pred)
        return acc

    def predict_proba(self, X, batch_size=32, verbose=0):
        """
        Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        batch_size : int, optional, default=32
            Number of samples for which predictions are obtained at one time using the learned model.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities."""
        self.logger.info("Predicting Probabilities")
        n_samples = X.shape[0]
        X = self.__transform__(X)
        if batch_size is None:
            y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        else:
            n_batches = np.ceil(n_samples / batch_size).astype(int)
            predictions = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                self.logger.info(f"Processing batch {i + 1}/{n_batches} Start id {start_idx} end id {end_idx}")
                batch_pred = self.model.predict_proba(X_batch, normalize_with_test=True, return_logits=False)
                predictions.append(batch_pred)

            y_pred = np.concatenate(predictions, axis=0)
        self.logger.info("Predicting Probabilities Done")
        self.__clear_memory__()
        return y_pred

    def decision_function(self, X, verbose=0):
        """
        Compute the decision function in form of class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values."""
        return self.predict_proba(X, verbose)

    @staticmethod
    def __clear_memory__():
        """Clear memory to release resources by torch."""
        import gc
        gc.collect()
        # Explicitly clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()