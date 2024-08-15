"""MI estimator integrating the TabPFN model, optimized for small tabular datasets with efficient MI estimation."""
from autoqild.automl import AutoTabPFNClassifier
from autoqild.mi_estimators.mi_estimator_classification import ClassficationMIEstimator
from autoqild.utilities._constants import LOG_LOSS_MI_ESTIMATION


class TabPFNMIEstimator(ClassficationMIEstimator):
    """
    TabPFNMIEstimator integrates the TabPFN framework into the Mutual Information (MI) estimation process for
    classification tasks.

    This class extends the `ClassficationMIEstimator` by using TabPFN as the base estimator. TabPFN is a powerful and
    efficient AutoML tool for small tabular datasets, capable of providing rapid predictions with pre-trained
    transformer models. The integration supports advanced feature reduction techniques, making it a robust choice for
    MI estimation in scenarios where both accuracy and efficiency are critical.

    Parameters
    ----------
    n_features : int
        The number of features in the input data.

    n_classes : int
        The number of classes in the classification task.

    n_ensembles : int, optional, default=100
        Number of ensemble models used in TabPFN to enhance prediction stability.

    n_reduced : int, optional, default=20
        Number of features to reduce to if `reduction_technique` is applied.

    reduction_technique : str, optional, default=`select_from_model_rf`
        Technique to use for feature reduction, provided by scikit-learn.
        Must be one of:

        - `recursive_feature_elimination_et`: Uses ExtraTreesClassifier to recursively remove features and build a model.
        - `recursive_feature_elimination_rf`: Uses RandomForestClassifier to recursively remove features and build a model.
        - `select_from_model_et`: Meta-transformer for selecting features based on importance weights using ExtraTreesClassifier.
        - `select_from_model_rf`: Meta-transformer for selecting features based on importance weights using RandomForestClassifier.
        - `pca`: Principal Component Analysis for dimensionality reduction.
        - `lda`: Linear Discriminant Analysis for separating classes.
        - `tsne`: t-Distributed Stochastic Neighbor Embedding for visualization purposes.
        - `nmf`: Non-Negative Matrix Factorization for dimensionality reduction.

    base_path : str, optional
        Directory to save model files. Default is None.

    random_state : int or None, optional, default=None
        Seed for random number generation to ensure reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `AutoTabPFNClassifier` constructor.

    Attributes
    ----------
    base_estimator : AutoTabPFNClassifier
        The base AutoML estimator used for classification.

    learner_params : dict
        Parameters used to configure the base learner.

    base_learner : AutoTabPFNClassifier
        Instance of the TabPFN classifier used for learning.

    """

    def __init__(self, n_features, n_classes, n_ensembles=100, n_reduced=20, reduction_technique="select_from_model_rf",
                 base_path=None, random_state=None, **kwargs):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state, **kwargs)
        self.base_estimator = AutoTabPFNClassifier
        self.learner_params = dict(n_features=n_features, n_classes=n_classes, n_ensembles=n_ensembles,
                                   n_reduced=n_reduced, reduction_technique=reduction_technique,
                                   base_path=base_path, random_state=random_state)
        self.base_learner = self.base_estimator(**self.learner_params)

    def fit(self, X, y, **kwd):
        """
        Fit the TabPFN classification model to the data.

        This method trains the TabPFN model using the provided dataset. It leverages the hyperparameters
        and reduction techniques specified during initialization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        **kwd : dict, optional
            Additional keyword arguments passed to the `fit` method of the base learner.

        Returns
        -------
        self : TabPFNMIEstimator
            Fitted estimator.
        """
        super().fit(X, y, **kwd)

    def predict(self, X, verbose=0):
        """
        Predict class labels for samples in X using the TabPFN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        return super().predict(X, verbose=verbose)

    def score(self, X, y, sample_weight=None, verbose=0):
        """
        Return the accuracy score of the TabPFN model on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return super().score(X, y, sample_weight=sample_weight, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        """
        Predict class probabilities for samples in X using the TabPFN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        p_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return super().predict_proba(X, verbose=verbose)

    def decision_function(self, X, verbose=0):
        """
        Predict confidence scores for samples using the TabPFN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        scores : array-like of shape (n_samples, n_classes)
            Predicted confidence scores.
        """
        scores = super().decision_function(X, verbose=verbose)
        return scores

    def estimate_mi(self, X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        """
        Estimate Mutual Information using the specified method with the TabPFN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        method : str, optional, default=`LogLossMI`
            The method to use for mutual information estimation.
            Options include:

            - `LogLossMI`: Estimate MI using Log-Loss method.
            - `LogLossMIIsotonicRegression`: Estimate MI using Log-Loss method with Isotonic Regression.
            - `LogLossMIPlattScaling`: Estimate MI using Log-Loss method with Platt Scaling.
            - `LogLossMIBetaCalibration`: Estimate MI using Log-Loss method with Beta Calibration.
            - `LogLossMITemperatureScaling`: Estimate MI using Log-Loss method with Temperature Scaling.
            - `LogLossMIHistogramBinning`: Estimate MI using Log-Loss method with Histogram Binning.
            - `PCSoftmaxMI`: Estimate MI using Softmax probabilities.

        **kwargs : dict, optional
            Additional keyword arguments passed to the estimation methods.

        Returns
        -------
        mutual_information : float
            A mean of estimated MI values from cross-validation splits.
        """
        return super().estimate_mi(X=X, y=y, method=method, **kwargs)
