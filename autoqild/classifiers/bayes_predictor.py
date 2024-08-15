import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from ..bayes_search import get_scores
from ..utilities import normalize


class BayesPredictor(BaseEstimator, ClassifierMixin):
    """
    A Bayes-optimal classifier that predicts on the given dataset using the defined joint and conditional
    distributions. This classifier leverages a dataset object, used to generate underlying data,
    which represents the best-performing classifier. This class stores the PDFs and predicts class probabilities
    and labels given the input features.

    Parameters
    ----------
    dataset_obj : object
        An object representing the dataset. This object should provide methods like
        `generate_dataset`, `get_prob_y_given_x`, and `get_prob_flip_y_given_x`.

    random_state : int or None, optional, default=None
        Random state for reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    ----------
    dataset_obj : object
        The dataset object provided during initialization. Used for generating datasets
        and computing class probabilities.

    random_state : RandomState
        Random state instance for reproducibility.

    logger : logging.Logger
        Logger instance for logging information.

    n_classes : int or None
        Number of classes in the classification data samples. Set during the `fit` method.
    """

    def __init__(self, dataset_obj, random_state=None, **kwargs):
        self.dataset_obj = dataset_obj
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(BayesPredictor.__name__)
        self.n_classes = None

    def fit(self, X, y, **kwd):
        """
        Fit the BayesPredictor model.

        This method sets the number of classes in the training data but does not perform any
        actual fitting. It is intended to be overridden or expanded in a subclass.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        **kwd : dict, optional
            Additional keyword arguments.
        """
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X, verbose=0):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        pred_probabilities = self.predict_proba(X=X, verbose=verbose)
        y_pred = pred_probabilities.argmax(axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        """
        Compute the accuracy of the predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The true labels.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        accuracy_score : float
            The accuracy score.
        """
        y_pred = self.predict(X)
        accuracy_score = np.mean(y_pred == y)
        return accuracy_score

    def decision_function(self, X, verbose=0):
        """
        Compute the decision function for the input samples.

        The decision function returns the probability estimates of the positive class
        for binary classification problems.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        scores : array-like of shape (n_samples,)
            The decision function values.
        """
        scores = self.predict_proba(X)
        if self.n_classes == 2:
            scores = scores[:, 1]
        return scores

    def predict_proba(self, X, verbose=0):
        """
        Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        p_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        p_pred = np.zeros((X.shape[0], self.dataset_obj.n_classes))
        for k_class in self.dataset_obj.class_labels:
            if self.dataset_obj.flip_y == 0.0:
                p_pred[:, k_class] = self.dataset_obj.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                p_pred[:, k_class] = self.dataset_obj.get_prob_flip_y_given_x(X=X, class_label=k_class)
        p_pred = normalize(p_pred, axis=1)
        return p_pred

    def get_bayes_predictor_scores(self):
        """
        Generate datasets and evaluate the accuracy of the Bayes predictor.

        This method generates multiple datasets and evaluates the accuracy of the Bayes predictor
        on each one. It returns the true and predicted labels along with the prediction probabilities
        for the dataset that achieved the highest accuracy.

        Returns
        -------
        y_true : array-like of shape (n_samples,)
            The true labels for the dataset with the highest accuracy.

        y_pred : array-like of shape (n_samples,)
            The predicted labels for the dataset with the highest accuracy.

        p_pred : array-like of shape (n_samples, n_classes)
            The predicted probabilities for the dataset with the highest accuracy.
        """
        max_acc = -np.inf
        y_true = None
        y_pred = None
        p_pred = None
        for i in range(100):
            X, y = self.dataset_obj.generate_dataset()
            pred = self.predict(X)
            acc_bp = np.mean(pred == y)
            if acc_bp > max_acc:
                self.logger.info(f"Accuracy of Bayes Predictor is {acc_bp}")
                max_acc = acc_bp
                y_true = np.copy(y)
                p_pred, y_pred = get_scores(X, self)
        return y_true, y_pred, p_pred