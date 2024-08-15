import os
import sys
import traceback
import warnings

import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

__all__ = ['logsumexp', 'softmax', 'sigmoid', 'normalize', 'progress_bar', 'print_dictionary', 'standardize_features',
           'standardize_features', 'create_directory_safely', 'log_exception_error', 'check_and_delete_corrupt_h5_file']


def logsumexp(x, axis=1):
    """
    Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis along which the sum is computed. Default is 1.

    Returns
    -------
    lsum_exp : array-like
        An array with the log of the sum of exponentials of elements along the specified axis.
    """
    max_x = x.max(axis=axis, keepdims=True)
    lsum_exp = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))
    return lsum_exp


def softmax(x, axis=1):
    """
    Compute the softmax of input elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis along which the softmax is computed. Default is 1.

    Returns
    -------
    s_values : array-like
        An array with the softmax applied along the specified axis.
    """
    lse = logsumexp(x, axis=axis)
    s_values = np.exp(x - lse)
    return s_values


def sigmoid(x):
    """
    Compute the sigmoid of the input array.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    x : array-like
        The sigmoid of the input array.
    """
    x = 1.0 / (1.0 + np.exp(-x))
    return x


def normalize(x, axis=1):
    """
    Normalize the input array along the specified axis.

    Parameters
    ----------
    x : array-like
        Input array to normalize.
    axis : int, optional
        Axis along which the normalization is applied. Default is 1.

    Returns
    -------
    normed : array-like
        The normalized array.
    """
    normed = x / np.sum(x, axis=axis, keepdims=True)
    return normed


def progress_bar(count, total, status=''):
    """
    Display a progress bar in the console.

    Parameters
    ----------
    count : int
        Current progress count.
    total : int
        Total count for completion.
    status : str, optional
        A status message to display along with the progress bar.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep='\n', n_keys=None):
    """
    Print a dictionary with keys and values formatted with a separator.

    Parameters
    ----------
    dictionary : dict
        The dictionary to print.
    sep : str, optional
        Separator between key-value pairs (default is '\n').
    n_keys : int, optional
        Number of key-value pairs to print. If None, prints all.

    Returns
    -------
    output : str
        Formatted string representation of the dictionary.
    """
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
    """
    Standardize the features in the training and test sets using RobustScaler as a default.

    Parameters
    ----------
    x_train : array-like
        Training set features.
    x_test : array-like
        Test set features.

    Returns
    -------
    x_train : array-like
        Standardized training set features.
    x_test : array-like
        Standardized test set features.
    """
    standardize = Standardize()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


class Standardize(object):
    def __init__(self, scalar=RobustScaler):
        """
        A class for standardizing features using a specified scaler.

        Parameters
        ----------
        scalar : object, optional
            The scaling class to use (default is `RobustScaler`).

        """
        self.scalar = scalar
        self.n_features = None
        self.scalars = dict()

    def fit(self, X):
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : array-like or dict
            The data to fit the scaler on.

        Returns
        -------
        self : object
            Fitted scaler.
        """
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
        """
        Apply the scaling transformation to the data.

        Parameters
        ----------
        X : array-like or dict
            The data to transform.

        Returns
        -------
        X : array-like or dict
            The transformed data.
        """
        if isinstance(X, dict):
            for n in self.n_features:
                X[n] = self.scalars[n].__transform__(X[n])
        if isinstance(X, (np.ndarray, np.generic)):
            X = self.scalar.transform(X)
        return X

    def fit_transform(self, X):
        """
        Fit the scaler and transform the data.

        Parameters
        ----------
        X : array-like or dict
            The data to fit and transform.

        Returns
        -------
        X : array-like or dict
            The transformed data.
        """
        self.fit(X)
        X = self.transform(X)
        return X


def log_exception_error(logger, e):
    """
    Log an exception with traceback details.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to log the error.
    e : Exception
        Exception instance to log.
    """
    if hasattr(e, 'message'):
        message = e.message
    else:
        message = e
    logger.error(traceback.format_exc())
    logger.error(message)


def create_directory_safely(path, is_file_path=False):
    """
    Create a directory if it does not exist, handling potential errors safely.

    Parameters
    ----------
    path : str
        Path to the directory or file.
    is_file_path : bool, optional
        If True, considers `path` as a file path and creates the directory containing the file.
    """
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(str(e))


def check_and_delete_corrupt_h5_file(file_path, logger):
    """
    Check if an HDF5 file is corrupt and delete it if necessary.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    logger : logging.Logger
        Logger instance to log actions.
    """
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


