"""Base class for classification-based MI estimators, providing a framework for
estimating MI in supervised learning."""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state

from autoqild.bayes_search.bayes_search_utils import probability_calibration, get_scores
from autoqild.detectors._utils import (
    calibrators,
    calibrator_params,
    mi_estimation_metrics,
)
from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from autoqild.utilities import *


class ClassficationMIEstimator(MIEstimatorBase):
    """Class to estimate Mutual Information (MI) using a classification model.

    This class leverages a classification model, such as `RandomForestClassifier`, to estimate the Mutual Information
    between input features and class labels using various metrics, including log-loss and softmax probabilities.
    It extends the `MIEstimatorBase` class, inheriting its basic structure and functionalities.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification data samples.
    n_features : int
        Number of features or dimensionality of the inputs of the classification data samples.
    random_state : int or object, optional, default=42
        Random state for reproducibility.
    **kwargs : dict, optional
        Additional keyword arguments passed to the base learner `RandomForestClassifier`.

    Attributes
    ----------
    random_state : RandomState instance
        Random state instance for reproducibility.
    logger : logging.Logger
        Logger instance for logging information.
    base_estimator : sklearn.ensemble.RandomForestClassifier
        Base estimator used for classification.
    learner_params : dict
        Parameters passed to the base estimator.
    base_learner : object
        The instantiated base learner.

    Methods
    -------
    fit(X, y, **kwd):
        Fit the classification model to the data.

    predict(X, verbose=0):
        Predict class labels for samples in X.

    score(X, y, sample_weight=None, verbose=0):
        Return the accuracy score of the model on the given test data and labels.

    predict_proba(X, verbose=0):
        Predict class probabilities for samples in X.

    decision_function(X, verbose=0):
        Predict confidence scores for samples, which may coincide with the probability scores in X.

    estimate_mi(X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        Estimate Mutual Information using the specified method.
    """

    def __init__(self, n_classes, n_features, random_state=None, **kwargs):
        super().__init__(n_classes, n_features, random_state)
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(ClassficationMIEstimator.__name__)
        self.base_estimator = RandomForestClassifier
        self.learner_params = {}
        self.base_learner = self.base_estimator(**self.learner_params)

    def fit(self, X, y, **kwd):
        """Fit the classification model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        **kwd : dict, optional
            Additional keyword arguments passed to the `fit` method of the classifier.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.base_learner.fit(X, y)

    def predict(self, X, verbose=0):
        """Predict class labels for samples in X.

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
        y_pred = self.base_learner.predict(X)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        """Return the accuracy score of the model on the given test data and
        labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return self.base_learner.score(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, verbose=0):
        """Predict class probabilities for samples in X.

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
        p_pred = self.base_learner.predict_proba(X)
        return p_pred

    def decision_function(self, X, verbose=0):
        """Predict confidence scores for samples, which may coincide with the
        probability scores in X.

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
        scores = self.base_learner.decision_function(X)
        return scores

    def estimate_mi(self, X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        """Estimate Mutual Information using the specified method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        method : str, optional, default=`LogLossMI`
            The method to use for mutual information estimation. Options include:

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
        evaluation_metric = mi_estimation_metrics[method]
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.70, random_state=0)
        estimated_mis = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p_pred, y_pred = get_scores(X_train, self.base_learner)
            if LOG_LOSS_MI_ESTIMATION in method or PC_SOFTMAX_MI_ESTIMATION in method:
                calibrator_technique = None
                for key in calibrators.keys():
                    if key in method:
                        calibrator_technique = key
                if calibrator_technique is not None:
                    calibrator = calibrators[calibrator_technique]
                    c_params = calibrator_params[calibrator_technique]
                    calibrator = calibrator(**c_params)
                    try:
                        p_pred_cal = probability_calibration(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            classifier=self.base_learner,
                            calibrator=calibrator,
                        )
                        estimated_mi = evaluation_metric(y, p_pred_cal)
                    except Exception as error:
                        log_exception_error(self.logger, error)
                        self.logger.error(
                            "Error while calibrating the probabilities estimating MI without calibration"
                        )
                        estimated_mi = evaluation_metric(y_train, p_pred)
                else:
                    estimated_mi = evaluation_metric(y_train, p_pred)
            else:
                estimated_mi = evaluation_metric(y_train, y_pred)
            estimated_mis.append(estimated_mi)
        mutual_information = np.mean(estimated_mis)
        return mutual_information
