import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from .utils import normalize

__all__ = ['bin_ce', 'helmann_raviv_function', 'helmann_raviv_upper_bound', 'santhi_vardi_upper_bound',
           'fanos_lower_bound', 'fanos_adjusted_lower_bound', 'auc_score', 'pc_softmax_estimation',
           'log_loss_estimation', 'mid_point_mi', 'false_positive_rate', 'false_negative_rate']

logger = logging.getLogger("Metrics")


def bin_ce(p_e):
    """
    Computes the binary cross-entropy for a given probability `p_e`.

    Parameters
    ----------
    p_e : float
        Probability value for which binary cross-entropy is computed.

    Returns
    -------
    binary_cross_entropy : float
        The binary cross-entropy value.

    Notes
    -----
    - This function handles edge cases where `p_e` is 0 or 1 by adding or subtracting a small epsilon value
      to prevent division by zero errors.
    """
    if p_e == 0:
        p_e = p_e + np.finfo(np.float32).eps
    if p_e == 1.0:
        p_e = p_e - np.finfo(np.float32).eps
    binary_cross_entropy = -p_e * np.log2(p_e) - (1 - p_e) * np.log2(1 - p_e)
    return binary_cross_entropy


bce_f = np.vectorize(bin_ce)


def helmann_raviv_function(n_classes, pe):
    """
    Computes the Hellman-Raviv function for a given error probability `pe`.

    The Hellman-Raviv function is used to estimate the upper bound of mutual information based on classification error rates.

    Parameters
    ----------
    n_classes : int
        The number of classes in the classification task.
    pe : ndarray
        The error probability values for each sample.

    Returns
    -------
    hrf_values : ndarray
        The computed Hellman-Raviv function values.

    Notes
    -----
    - The function partitions the error probabilities into ranges based on the number of classes
      and computes the upper bound using a series of logarithmic transformations.
    """
    hrf_values = []
    indices = []
    num = pe.shape[0]

    for k in range(1, int(n_classes)):
        def cal_l(k, n_pe):
            T = (k + 1) / k
            T2 = (k - 1) / k
            l = np.log2(k) + k * (k + 1) * np.log2(T) * (n_pe - T2)
            return l

        l_mpe = (1 - 1 / k)
        u_mpe = (1 - 1 / (k + 1))
        idx = np.where((pe >= l_mpe) & (pe < u_mpe))[0]
        indices.extend(idx)
        if len(idx) != 0:
            n_pe = pe[idx]
            l = cal_l(k, n_pe)
            hrf_values.extend(l)

    idx = np.array(list(set(np.arange(num)) ^ set(indices)))
    if len(idx) != 0:
        n_pe = pe[idx]
        l = cal_l(k, n_pe)
        hrf_values.extend(l)

    hrf_values = np.array(hrf_values)
    return hrf_values


def helmann_raviv_upper_bound(y_true, y_pred):
    """
    Computes the Hellman-Raviv upper bound for mutual information based on classification performance.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted class labels.

    Returns
    -------
    hr_u : float
        The Hellman-Raviv upper bound for mutual information.

    Notes
    -----
    - The Hellman-Raviv bound is calculated as the difference between the logarithm of the number of classes
      and the computed Hellman-Raviv function for the error rate.
    """
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    error_rate = 1 - acc
    hmr = helmann_raviv_function(n_classes, np.array([error_rate]))[0]
    hr_u = np.log2(n_classes) - hmr
    return hr_u


def santhi_vardi_upper_bound(y_true, y_pred):
    """
    Computes the Santhi-Vardi upper bound for mutual information.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted class labels.

    Returns
    -------
    sv_u: float
        The Santhi-Vardi upper bound.

    Notes
    -----
    - The Santhi-Vardi bound is based on the classification error rate and gives an upper estimate of the
      mutual information, adjusted logarithmically based on the number of classes.
    """
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    error_rate = 1 - acc
    if error_rate == 1.0:
        error_rate = error_rate - np.finfo(np.float64).eps
    sv_u = np.log2(n_classes) + np.log2(1 - error_rate)
    return sv_u


def fanos_lower_bound(y_true, y_pred):
    """
    Computes Fano's lower bound for mutual information.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted class labels.

    Returns
    -------
    fanos_lb : float
        Fano's lower bound.

    Notes
    -----
    - Fano's bound gives a lower estimate of mutual information by considering the classification error
      and the complexity of the classification task (in terms of the number of classes).
    """
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    T = np.log(n_classes - 1) / np.log(n_classes)
    fanos_lb = np.log2(n_classes) * (1 - pe * T) - bin_ce(pe)
    return fanos_lb


def fanos_adjusted_lower_bound(y_true, y_pred):
    """
    Computes the adjusted Fano's lower bound for mutual information.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted class labels.

    Returns
    -------
    fanos_adjusted_lb : float
        Adjusted Fano's lower bound.

    Notes
    -----
    - This adjusted bound accounts for binary cross-entropy and provides a refined lower bound estimate
      compared to the standard Fano's bound.
    """
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    fanos_adjusted_lb = np.log2(n_classes) * (1 - pe) - bce_f(pe)
    return fanos_adjusted_lb


def mid_point_mi(y_true, y_pred):
    """
    Computes the midpoint mutual information estimate by averaging the upper and lower bounds.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted class labels.

    Returns
    -------
    mid_point : float
        Midpoint mutual information estimate.

    Notes
    -----
    - This estimate is computed as the average of the Hellman-Raviv upper bound and Fano's lower bound.
    - The estimate is constrained to be non-negative by taking the maximum with zero.
    """
    mid_point = helmann_raviv_upper_bound(y_true, y_pred) + fanos_lower_bound(y_true, y_pred)
    mid_point = mid_point / 2.0
    mid_point = np.max([mid_point, 0.0])
    return mid_point


def auc_score(y_true, p_pred):
    """
    Computes the AUC score for the given true labels and predicted probabilities.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    p_pred : ndarray
        Predicted probabilities.

    Returns
    -------
    auc_roc : float
        AUC score.

    Notes
    -----
    - For multi-class scenarios, the AUC is computed using a one-vs-rest approach.
    - The method includes normalization as a fallback if issues arise during computation.
    """
    logger = logging.getLogger("AUC")
    n_classes = len(np.unique(y_true))
    if n_classes > 2:
        try:
            auc_roc = roc_auc_score(y_true, p_pred, multi_class='ovr')
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            try:
                logger.error(f"Applying normalization to avoid exception")
                p_pred = normalize(p_pred, axis=1)
                auc_roc = roc_auc_score(y_true, p_pred, multi_class='ovr')
            except Exception as e:
                logger.error(f"After normalization Exception: {str(e)}")
                logger.error(f"Setting AUC to NaN")
                auc_roc = np.nan
    else:
        auc_roc = roc_auc_score(y_true, p_pred)
    return auc_roc


def false_positive_rate(y_true, y_pred):
    """
    Computes the false positive rate (FPR).

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_pred : ndarray
        Predicted binary labels.

    Returns
    -------
    fpr : float
        False positive rate.

    Notes
    -----
    - FPR is calculated as the ratio of false positives to the sum of false positives and true negatives.
    """
    tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    fpr = fp / (fp + tn)
    return fpr


def false_negative_rate(y_true, y_pred):
    """
    Computes the false negative rate (FNR).

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_pred : ndarray
        Predicted binary labels.

    Returns
    -------
    fnr : float
        False negative rate.

    Notes
    -----
    - FNR is calculated as the ratio of false negatives to the sum of false negatives and true positives.
    """
    tp = np.logical_and(y_true, y_pred).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
    fnr = fn / (tp + fn)
    return fnr


def remove_nan_values(y_pred, y_true=None):
    """
    Removes rows containing NaN values from the predicted probabilities and true labels.

    Parameters
    ----------
    y_pred : ndarray
        Predicted probabilities.
    y_true : ndarray, optional
        True labels corresponding to the predicted probabilities (default is None).

    Returns
    -------
    y_pred : ndarray
        Cleaned predicted probabilities.
    y_true : ndarray, optional
        Cleaned true labels corresponding to the non-NaN predicted probabilities (if provided).

    Notes
    -----
    - The method filters out rows where any NaN values are present in the predicted probabilities.
    - If `y_true` is provided, it is also filtered to keep only the corresponding non-NaN entries.
    """
    nan_rows = np.isnan(y_pred).any(axis=1)
    y_pred = y_pred[~nan_rows]
    if y_true is not None:
        y_true = y_true[~nan_rows]
    return y_pred, y_true


def get_entropy_y(y_true):
    """
    Computes the entropy of the true labels.
    Softmax Function:

    .. math::

        H(Y) = \\sum{y \\in \\mathcal{Y}}{(p_y \\lg (p_y))}

    where:
        - \( p_y \) is the prior probability of class \( y \), calculated as \( p_y = \frac{\text{counts}_y}{\text{total samples}} \).
    Parameters
    ----------
    y_true : ndarray
        True class labels.

    Returns
    -------
    entropy_y : float
        Entropy of the true labels.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    pys = counts / np.sum(counts)
    entropy_y = 0
    for k_class, py in zip(classes, pys):
        entropy_y += -py * np.log2(py)
    return entropy_y


def pc_softmax_estimation(y_true, p_pred):
    """
    Estimates the mutual information using predicted probabilities in the softmax and PC-Softmax functions.

    The mutual information I(X; Y) is estimated using the formula:

    .. math::

        I(X;Y) = H(Y) - H(Y|X)

    where H(Y) is the entropy of the true labels and H(Y|X) is the conditional entropy estimated from the predicted probabilities.

    Softmax Function:

    .. math::

        S(z_k) = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}

    PC-Softmax (Probability-Corrected Softmax) Function:

    .. math::

        S_{pc}(z_k) = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j} \\cdot p_j}

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    p_pred : ndarray
        Predicted probabilities.

    Returns
    -------
    estimated_mi : float
        Estimated mutual information.

    Notes
    -----
    - The PC-Softmax estimation adjusts the softmax probabilities using class priors, which can improve the robustness of the MI estimate.
    - If the input contains NaN values, they are removed before performing the estimation.
    """
    p_pred[p_pred == 0] = np.finfo(float).eps
    p_pred[p_pred == 1] = 1 - np.finfo(float).eps
    p_pred, y_true = remove_nan_values(p_pred, y_true=y_true)

    if y_true.size != 0:
        classes, counts = np.unique(y_true, return_counts=True)
        pys = counts / np.sum(counts)
        mis = []
        x_exp = np.exp(np.log(p_pred / (1 - p_pred)))
        weighted_x_exp = x_exp * pys
        x_exp_sum = np.sum(weighted_x_exp, axis=1, keepdims=True)
        own_softmax = x_exp / x_exp_sum

        for i, y_t in enumerate(y_true):
            softmax = own_softmax[i, int(y_t)]
            mis.append(np.log2(softmax))

        estimated_mi = np.nanmean(mis)
        estimated_mi = np.nanmax([estimated_mi, 0.0])
    else:
        logger.error("All rows were NaN, so cannot estimate mutual information")
        estimated_mi = 0.0
    return estimated_mi


def log_loss_estimation(y_true, y_pred):
    """
    Estimates mutual information by evaluating the log-loss of the predicted probabilities and entropy of outputs.

    Parameters
    ----------
    y_true : ndarray
        True class labels.
    y_pred : ndarray
        Predicted probabilities.

    Returns
    -------
    estimated_mi : float
        Estimated mutual information.

    Notes
    -----
    - The estimation is based on calculating the entropy H(Y) of the true labels and the average log-loss of the predictions.
    - NaN values in the input are removed before performing the estimation.
    """
    y_pred[y_pred == 0] = np.finfo(float).eps
    y_pred[y_pred == 1] = 1 - np.finfo(float).eps
    y_pred, y_true = remove_nan_values(y_pred, y_true=y_true)

    if y_true.size != 0:
        mi_pp = get_entropy_y(y_true)
        if len(y_pred.shape) == 1:
            pyx = (y_pred * np.log2(y_pred) + (1 - y_pred) * np.log2(1 - y_pred))
        else:
            pyx = (y_pred * np.log2(y_pred)).sum(axis=1)
        mi_bp = pyx.mean()
        estimated_mi = mi_bp + mi_pp
        estimated_mi = np.nanmax([estimated_mi, 0.0])
    else:
        logger.error("All rows were NaN, so cannot estimate mutual information")
        estimated_mi = 0.0
    return estimated_mi
