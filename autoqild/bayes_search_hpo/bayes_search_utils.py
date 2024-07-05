import logging

import numpy as np
import sklearn
from autogluon.core.models import AbstractModel
from packaging import version
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

from pycilt import AutoTabPFNClassifier
from pycilt.utils import print_dictionary, sigmoid

__all__ = ["get_scores", "convert_value", "get_parameters_at_k", "update_params_at_k", "log_callback", "get_scores"]

logger = logging.getLogger("BayesSearchUtils")
def convert_value(value):
    try:
        # Try converting to integer
        return int(value)
    except ValueError:
        try:
            # Try converting to float
            return float(value)
        except ValueError:
            # Return as string if conversion fails
            return value


def get_parameters_at_k(optimizers, search_keys, k):
    yis = []
    xis = []
    for opt in optimizers:
        yis.extend(opt.yi)
        xis.extend(opt.Xi)
    yis = np.array(yis)
    # xis = np.array(xis)
    index_k = np.argsort(yis)[k]
    best_params = xis[index_k]
    best_loss = yis[index_k]
    # best_params = [convert_value(p) for p in best_params]
    best_params = dict(zip(search_keys, best_params))
    return best_loss, best_params


def update_params_at_k(bayes_search, search_keys, learner_params, k=0):
    loss, best_params = get_parameters_at_k(optimizers=bayes_search.optimizers_, search_keys=search_keys, k=k)
    if version.parse(sklearn.__version__) < version.parse("0.25.0"):
        if 'criterion' in best_params.keys():
            if best_params['criterion'] == 'squared_error':
                best_params['criterion'] = 'mse'
    learner_params.update(best_params)
    params_str = print_dictionary(learner_params, sep='\t')
    logger.info(f"Parameters at position k:{k} are {params_str} with objective of: {-loss}\n")
    # logger.info(f"Model {k} with loss {loss} and parameters {learner_params}")
    return loss, learner_params


def log_callback(parameters):
    def on_step(opt_result):
        """
        Callback meant to view scores after
        each iteration while performing Bayesian
        Optimization in Skopt"""
        points = opt_result.x_iters
        scores = -opt_result.func_vals
        params = dict(zip(parameters, points[-1]))
        params_str = print_dictionary(params, sep=' : ')
        logger.info(f'For Parameters: {params_str}, Objective: {scores[-1]}')

    return on_step


def get_scores(X, estimator):
    try:
        pred_prob = estimator.predict_proba(X)
    except:
        pred_prob = estimator.decision_function(X)
    # logger.info("Predict Probability shape {}, {}".format(pred_prob.shape, y_test.shape))

    if len(pred_prob.shape) == 2 and pred_prob.shape[-1] > 1:
        p_pred = pred_prob
    else:
        p_pred = pred_prob.flatten()
    if isinstance(estimator, AbstractModel):
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    if isinstance(estimator, SGDClassifier) or isinstance(estimator, LinearSVC) or isinstance(estimator,
                                                                                              RidgeClassifier):
        p_pred = sigmoid(p_pred)
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    if isinstance(estimator, AutoTabPFNClassifier):
        y_pred = np.argmax(p_pred, axis=-1)
    else:
        y_pred = estimator.predict(X)
    y_pred = np.array(y_pred)
    p_pred = np.array(p_pred)
    # logger = logging.getLogger("Score")
    # logger.info(f"Scores Shape {p_pred.shape}, Classes {np.unique(y_pred)}")
    return p_pred, y_pred
