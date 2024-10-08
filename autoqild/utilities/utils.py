"""Implements several utility functions for array normalization, logging
exceptions, managing HDF5 files, and creating directories safely."""

import os
import sys
import traceback
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

__all__ = [
    "logsumexp",
    "softmax",
    "sigmoid",
    "normalize",
    "progress_bar",
    "print_dictionary",
    "standardize_features",
    "create_directory_safely",
    "log_exception_error",
    "check_and_delete_corrupt_h5_file",
]


def logsumexp(x, axis=1):
    """Compute the log of the sum of exponentials of input elements.

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
    """Compute the softmax of input elements.

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
    """Compute the sigmoid of the input array.

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
    """Normalize the input array along the specified axis.

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


def progress_bar(count, total, status=""):
    """Display a progress bar in the console.

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

    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s/%s ...%s\r" % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep="\n", n_keys=None):
    """Format the dictionary to print it in logs.

    Parameters
    ----------
    dictionary : dict
        The dictionary to print.
    sep : str, optional
        The separator between key-value pairs. The default is '\n'.
    n_keys : int, optional
        The number of key-value pairs to print. If None, all pairs are printed.

    Returns
    -------
    output : str
        Formatted string representation of the dictionary.
    """
    output = "  "
    if n_keys is None:
        n_keys = len(dictionary)
    for i, (key, value) in enumerate(dictionary.items()):
        output += f"{str(key)} => {str(value)}"
        if i < n_keys - 1:
            output += sep
        else:
            break
    return output


def log_exception_error(logger, e):
    """Log an exception with traceback details.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to log the error.
    e : Exception
        Exception instance to log.
    """
    if hasattr(e, "message"):
        message = e.message
    else:
        message = e
    logger.error(traceback.format_exc())
    logger.error(message)


def create_directory_safely(path, is_file_path=False):
    """Create a directory if it does not exist, handling potential errors
    safely.

    Parameters
    ----------
    path : str
        Path to the directory or file.
    is_file_path : bool, optional
        If True, considers "path" as a file path and creates the directory containing the file.
    """
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(str(e))


def check_and_delete_corrupt_h5_file(file_path, logger):
    """Check if an HDF5 file is corrupt and delete it if necessary.

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
            with h5py.File(file_path, "r") as h5_file:
                group_names = list(h5_file.keys())
                if group_names:
                    group_name = group_names[0]
                    group = h5_file[group_name]
                    logger.info(
                        f"The first group '{group_name}' in the file '{basename}' has been "
                        f"accessed successfully."
                    )
                else:
                    logger.info(f"No groups found in the file '{basename}'.")
            logger.info(f"The file '{basename}' is not corrupt.")
        except (OSError, KeyError, ValueError, Exception) as error:
            log_exception_error(logger, error)
            logger.error(f"The file '{basename}' is corrupt.")
            os.remove(file_path)
            logger.error(f"The file '{basename}' has been deleted.")
    else:
        logger.info(f"File does not exist '{basename}'")


def standardize_features(x_train, x_test, scaler=RobustScaler, scaler_params={}):
    """
    Standardize the features in the training and test sets using the specified scaler.

    The function offers flexibility to choose between `StandardScaler`, `RobustScaler`, and `MinMaxScaler`.
    It allows customization of the chosen scaler’s parameters using a dictionary and raises a ValueError
    if an unsupported scaler is passed.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training set features.
    x_test : array-like of shape (n_samples, n_features)
        Test set features.
    scaler : {StandardScaler, RobustScaler, MinMaxScaler}, optional, default=RobustScaler
        The scaling class to be used for standardization. Choose from:
        - StandardScaler: Standardize features by removing the mean and scaling to unit variance.
        - RobustScaler: Scale features using statistics that are robust to outliers.
        - MinMaxScaler: Scale features to a given range (usually between 0 and 1).
    scaler_params : dict, optional, default={}
        Parameters to be passed to the selected scaler. Example: {'with_mean': False} for `StandardScaler`.

    Returns
    -------
    x_train : array-like of shape (n_samples, n_features)
        Standardized training set features.
    x_test : array-like of shape (n_samples, n_features)
        Standardized test set features.

    Raises
    ------
    ValueError
        If the specified scaler is not one of `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    >>> import numpy as np
    >>> x_train = np.array([[1, 2], [2, 3], [3, 4]])
    >>> x_test = np.array([[4, 5], [5, 6]])

    # Example with StandardScaler and a custom parameter
    >>> scaler_params = {'with_mean': False}
    >>> x_train_scaled, x_test_scaled = standardize_features(
    ...     x_train, x_test, scaler=StandardScaler, scaler_params=scaler_params
    ... )

    # Example with RobustScaler (default)
    >>> x_train_scaled, x_test_scaled = standardize_features(x_train, x_test, scaler=RobustScaler)

    # Example with MinMaxScaler
    >>> x_train_scaled, x_test_scaled = standardize_features(x_train, x_test, scaler=MinMaxScaler)

    # Example with an invalid scaler (this will raise a ValueError)
    >>> try:
    ...     x_train_scaled, x_test_scaled = standardize_features(x_train, x_test, scaler="InvalidScaler")
    ... except ValueError as e:
    ...     print(e)
    'Invalid scaler specified. Choose from StandardScaler, RobustScaler, or MinMaxScaler.'
    """
    if scaler not in [StandardScaler, RobustScaler, MinMaxScaler]:
        raise ValueError(
            "Invalid scaler specified. Choose from StandardScaler, RobustScaler, or MinMaxScaler."
        )

    # Initialize the chosen scaler with the specified parameters
    scaler_instance = scaler(**scaler_params)

    # Fit the scaler on the training data and transform both training and test data
    x_train = scaler_instance.fit_transform(x_train)
    x_test = scaler_instance.transform(x_test)

    return x_train, x_test
