"""Provides utility functions to support the hyperparameter tuning process,
including callback mechanisms, parameter extraction, and scoring functions."""

import logging

import numpy as np
import sklearn
from autogluon.core.models import AbstractModel
from packaging import version
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC

from .. import AutoGluonClassifier
from ..automl.tabpfn_classifier import AutoTabPFNClassifier
from ..utilities import print_dictionary, sigmoid
from ..utilities.metrics import remove_nan_values

__all__ = [
    "convert_value",
    "get_parameters_at_k",
    "update_params_at_k",
    "log_callback",
    "get_scores",
    "probability_calibration",
]

logger = logging.getLogger("BayesSearchUtils")


def convert_value(value):
    """Convert a value to its appropriate type.

    Parameters
    ----------
    value : str
        The value to be converted.

    Returns
    -------
    int, float, or str
        The converted value.

    Notes
    -----
    This function tries to convert the value to an integer first. If it fails, it tries to convert it to a float.
    If it still fails, it returns the value as a string.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_parameters_at_k(optimizers, search_keys, k):
    """Get the parameters and loss at the k-th position.

    Parameters
    ----------
    optimizers : list of skopt.optimizer.Optimizer
        The list of optimizers.

    search_keys : list of str
        The search keys for the parameters.

    k : int
        The position to retrieve the parameters from.

    Returns
    -------
    best_loss : float
        The best loss at the k-th position.

    best_params : dict
        The best parameters at the k-th position.
    """
    yis = []
    xis = []
    for opt in optimizers:
        yis.extend(opt.yi)
        xis.extend(opt.Xi)
    yis = np.array(yis)
    index_k = np.argsort(yis)[k]
    best_params = xis[index_k]
    best_loss = yis[index_k]
    best_params = dict(zip(search_keys, best_params))
    return best_loss, best_params


def update_params_at_k(bayes_search, search_keys, learner_params, k=0):
    """Update the learner parameters with the best parameters at the k-th
    position.

    Parameters
    ----------
    bayes_search : BayesSearchCV
        The BayesSearchCV instance.

    search_keys : list of str
        The search keys for the parameters.

    learner_params : dict
        The learner parameters to be updated.

    k : int, default=0
        The position to retrieve the parameters from.

    Returns
    -------
    loss : float
        The best loss at the k-th position.

    learner_params : dict
        The updated learner parameters.
    """
    loss, best_params = get_parameters_at_k(
        optimizers=bayes_search.optimizers_, search_keys=search_keys, k=k
    )
    if version.parse(sklearn.__version__) < version.parse("0.25.0"):
        if "criterion" in best_params.keys():
            if best_params["criterion"] == "squared_error":
                best_params["criterion"] = "mse"
    learner_params.update(best_params)
    params_str = print_dictionary(learner_params, sep="\t")
    logger.info(
        f"Parameters at position k:{k} are {params_str} with objective of: {-loss}\n"
    )
    return loss, learner_params


def log_callback(parameters):
    """Callback function for logging parameters and scores during Bayesian
    optimization.

    Parameters
    ----------
    parameters : list of str
        The parameters to log.

    Returns
    -------
    on_step : callable
        The callback function.
    """

    def on_step(opt_result):
        """Callback to view scores after each iteration while performing
        Bayesian Optimization in Skopt."""
        points = opt_result.x_iters
        scores = -opt_result.func_vals
        params = dict(zip(parameters, points[-1]))
        params_str = print_dictionary(params, sep=" : ")
        logger.info(f"For Parameters: {params_str}, Objective: {scores[-1]}")

    return on_step


def get_scores(X, estimator):
    """Get the predicted probabilities and labels for the input samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    estimator : estimator object
        The estimator to use for predictions.

    Returns
    -------
    p_pred : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    """
    try:
        pred_prob = estimator.predict_proba(X)
    except:
        pred_prob = estimator.decision_function(X)

    if len(pred_prob.shape) == 2 and pred_prob.shape[-1] > 1:
        p_pred = pred_prob
    else:
        p_pred = pred_prob.flatten()

    if isinstance(estimator, AbstractModel):
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    if isinstance(estimator, (SGDClassifier, LinearSVC, RidgeClassifier)):
        p_pred = sigmoid(p_pred)
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    if isinstance(estimator, AutoTabPFNClassifier):
        y_pred = np.argmax(p_pred, axis=-1)
    else:
        y_pred = estimator.predict(X)

    y_pred = np.array(y_pred)
    p_pred = np.array(p_pred)
    return p_pred, y_pred


def probability_calibration(X_train, y_train, X_test, classifier, calibrator):
    """Calibrate the predicted probabilities.

    Parameters
    ----------
    X_train : array-like of shape (n_samples_train, n_features)
        Training feature matrix.

    y_train : array-like of shape (n_samples_train,)
        Training target vector.

    X_test : array-like of shape (n_samples_test, n_features)
        Test feature matrix.

    classifier : estimator object
        The classifier to use for predictions.

    calibrator : calibrator object
        The calibrator to use for calibration.

    Returns
    -------
    y_pred_cal : array-like of shape (n_samples_test, n_classes)
        Calibrated predicted probabilities.
    """
    if isinstance(classifier, AbstractModel):
        n_features = X_train.shape[-1]
        n_classes = len(np.unique(y_train))
        X_train = AutoGluonClassifier(
            n_features=n_features, n_classes=n_classes
        ).convert_to_dataframe(X_train, None)
        X_test = AutoGluonClassifier(
            n_features=n_features, n_classes=n_classes
        ).convert_to_dataframe(X_test, None)

    y_pred_train, _ = get_scores(X_train, classifier)
    y_pred_test, _ = get_scores(X_test, classifier)

    if len(y_pred_train.shape) == 1:
        y_pred_train = np.hstack(((1 - y_pred_train)[:, None], y_pred_train[:, None]))
    if len(y_pred_test.shape) == 1:
        y_pred_test = np.hstack(((1 - y_pred_test)[:, None], y_pred_test[:, None]))

    y_pred_train, y_train = remove_nan_values(y_pred_train, y_true=y_train)
    y_pred_test, _ = remove_nan_values(y_pred_test, y_true=None)

    if y_train.size != 0:
        calibrator.fit(y_pred_train, y_train)
        y_pred_cal = calibrator.__transform__(y_pred_test)
        if len(y_pred_cal.shape) == 1:
            y_pred_cal = np.hstack(((1 - y_pred_cal)[:, None], y_pred_cal[:, None]))
    else:
        raise ValueError("All rows were nan, so cannot calibrate the probabilities")

    return y_pred_cal
