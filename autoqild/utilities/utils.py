import os
import sys
import traceback
import warnings

import h5py
import numpy as np
from autogluon.core.models import AbstractModel
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

from ..automl import AutoTabPFNClassifier

warnings.filterwarnings('ignore')

__all__ = ['logsumexp', 'softmax', 'sigmoid', 'normalize', 'progress_bar', 'print_dictionary', 'standardize_features',
           'standardize_features', 'create_directory_safely', 'log_exception_error', 'check_and_delete_corrupt_h5_file',
           'get_scores']


def logsumexp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def softmax(x, axis=1):
    """
    Take softmax for the given numpy array.
    :param axis: The axis around which the softmax is applied
    :param x: array-like, shape (n_samples, ...)
    :return: softmax taken around the given axis
    """
    lse = logsumexp(x, axis=axis)
    return np.exp(x - lse)


def sigmoid(x):
    x = 1.0 / (1.0 + np.exp(-x))
    return x


def normalize(x, axis=1):
    """
    Normalize the given two dimensional numpy array around the row.
    :param axis: The axis around which the norm is applied
    :param x: theano or numpy array-like, shape (n_samples, n_objects)
    :return: normalize the array around the axis=1
    """
    return x / np.sum(x, axis=axis, keepdims=True)


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep='\n', n_keys=None):
    output = "  "
    if n_keys is None:
        n_keys = len(dictionary)
    for i, (key, value) in enumerate(dictionary.items()):
        if i < n_keys - 1:
            output = output + str(key) + " => " + str(value) + sep
        else:
            output = output + str(key) + " => " + str(value)
        if i == n_keys - 1:
            break
    return output


def standardize_features(x_train, x_test):
    standardize = Standardize()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


class Standardize(object):
    def __init__(self, scalar=RobustScaler):
        self.scalar = scalar
        self.n_features = None
        self.scalars = dict()

    def fit(self, X):
        if isinstance(X, dict):
            self.n_features = list(X.keys())
            for k, x in X.items():
                scalar = self.scalar()
                self.scalars[k] = scalar.fit(x)
        if isinstance(X, (np.ndarray, np.generic)):
            self.scalar = self.scalar()
            self.scalar.fit(X)
            self.n_features = X.shape[-1]

    def transform(self, X):
        if isinstance(X, dict):
            for n in self.n_features:
                X[n] = self.scalars[n].transform(X[n])
        if isinstance(X, (np.ndarray, np.generic)):
            X = self.scalar.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X


def log_exception_error(logger, e):
    if hasattr(e, 'message'):
        message = e.message
    else:
        message = e
    logger.error(traceback.format_exc())
    logger.error(message)


def create_directory_safely(path, is_file_path=False):
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(str(e))


def check_and_delete_corrupt_h5_file(file_path, logger):
    basename = os.path.basename(file_path)
    if os.path.exists(file_path):
        try:
            if os.path.getsize(file_path) == 0:
                logger.info(f"The file '{basename}' is empty.")
                os.remove(file_path)
                logger.info(f"The file '{basename}' has been deleted.")
                return
            with h5py.File(file_path, 'r') as h5_file:
                group_names = list(h5_file.keys())
                if group_names:
                    group_name = group_names[0]
                    group = h5_file[group_name]
                    logger.info(f"The first group '{group_name}' in the file '{basename}' has been "
                                f"accessed successfully.")
                else:
                    logger.info(f"No groups found in the file '{basename}'.")
            logger.info(f"The file '{basename}' is not corrupt.")
        except (OSError, KeyError, ValueError, Exception) as error:
            log_exception_error(logger, error)
            logger.error(f"The file '{basename}' is corrupt.")
            os.remove(file_path)
            logger.error(f"The file '{basename}' has been deleted.")
    else:
        logger.info(f"File does not exist {basename}")


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
