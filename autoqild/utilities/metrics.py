import logging

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils import normalize

__all__ = ['bin_ce', 'helmann_raviv_function', 'helmann_raviv_upper_bound', 'santhi_vardi_upper_bound',
           'fanos_lower_bound', 'fanos_adjusted_lower_bound', 'auc_score', 'pc_softmax_estimation',
           'log_loss_estimation', 'mid_point_mi', 'false_positive_rate', 'false_negative_rate']

logger = logging.getLogger("Metrics")


# logger.info(f"Nan Rows Train {nan_rows}")
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

        Parameters
        ----------
        n_classes : int
            The number of classes.
        pe : ndarray
            The error probability values.

        Returns
        -------
        hrf_values : ndarray
            The computed Hellman-Raviv function values.
    """
    hrf_values = []
    indicies = []
    num = pe.shape[0]
    for k in range(1, int(n_classes)):
        def cal_l(k, n_pe):
            T = (k + 1) / k
            T2 = (k - 1) / k
            l = np.log2(k) + k * (k + 1) * np.log2(T) * (n_pe - T2)
            # T = (k+1)/k
            # T2 = (1+1/k)
            # l = np.log2(k+1) + k**2*np.log2(T)*(pe*T2-1)
            return l

        l_mpe = (1 - 1 / k)
        u_mpe = (1 - 1 / (k + 1))
        idx = np.where((pe >= l_mpe) & (pe < u_mpe))[0]
        indicies.extend(idx)
        if len(idx) != 0:
            n_pe = pe[idx]
            l = cal_l(k, n_pe)
            hrf_values.extend(l)
        # else:
        #   print(k, l_mpe, u_mpe, mpe)

        # plt.plot(pe, l, label='Hellman-Raviv k={}-{}'.format(k ,LOWER), linewidth=2, color='tab:orange')
    idx = np.array(list(set(np.arange(num)) ^ set(indicies)))
    if len(idx) != 0:
        n_pe = pe[idx]
        l = cal_l(k, n_pe)
        hrf_values.extend(l)
    hrf_values = np.array(hrf_values)
    return hrf_values


def helmann_raviv_upper_bound(y_true, y_pred):
    """
        Computes the Hellman-Raviv upper bound for mutual information.

        Parameters
        ----------
        y_true : ndarray
            True class labels.
        y_pred : ndarray
            Predicted class labels.

        Returns
        -------
        hr_u : float
            The Hellman-Raviv upper bound.
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
    """
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    fanos_adjusted_lb = np.log2(n_classes) * (1 - pe) - bce_f(pe)
    return fanos_adjusted_lb


def mid_point_mi(y_true, y_pred):
    """
        Computes the midpoint mutual information estimate.

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
                logger.error(f"Setting Auc to nan")
                auc_roc = np.nan
    else:
        auc_roc = roc_auc_score(y_true, p_pred)
    return auc_roc


def false_positive_rate(y_true, y_pred):
    """
        Computes the false positive rate.

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
    """
    tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    fpr = fp / (fp + tn)
    return fpr


def false_negative_rate(y_true, y_pred):
    """
        Computes the false positive rate.

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
            Cleaned Predicted probabilities.
        y_true : ndarray, optional
            Corresponsindg remaining True labels corresponding to the non-NaN predicted probabilities.

    """
    # logger.info(f"y_pred shape {y_pred.shape}")
    nan_rows = np.isnan(y_pred).any(axis=1)
    # logger.info(f"nan_rows shape {nan_rows.shape} y_pred.shape {y_pred.shape}")
    y_pred = y_pred[~nan_rows]
    if y_true is not None:
        # logger.info(f"nan_rows shape {nan_rows.shape} y_true.shape {y_true.shape}")
        y_true = y_true[~nan_rows]
    # logger.info(f"Nan rows {np.sum(nan_rows)} y_pred shape {y_pred.shape}")
    return y_pred, y_true


def get_entropy_y(y_true):
    """
        Computes the entropy of the true labels.

        Parameters
        ----------
        y_true : ndarray
            True class labels.

        Returns
        -------
        mi_pp : float
            Entropy of the true labels.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    pys = counts / np.sum(counts)
    mi_pp = 0
    for k_class, py in zip(classes, pys):
        mi_pp += -py * np.log2(py)
    return mi_pp


def pc_softmax_estimation(y_true, p_pred):
    """
        Estimates the mutual information using softmax of predicted probabilities.

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
    """
    p_pred[p_pred == 0] = np.finfo(float).eps
    p_pred[p_pred == 1] = 1 - np.finfo(float).eps
    p_pred, y_true = remove_nan_values(p_pred, y_true=y_true)
    # \[z_i = \ln\left(\frac{p_i}{1 - p_i}\right)\], Score approximation
    s_pred = np.log(p_pred / (1 - p_pred))
    if y_true.size != 0:
        classes, counts = np.unique(y_true, return_counts=True)
        pys = counts / np.sum(counts)
        mis = []
        x_exp = np.exp(s_pred)
        weighted_x_exp = x_exp * pys
        # weighted_x_exp = x_exp
        x_exp_sum = np.sum(weighted_x_exp, axis=1, keepdims=True)
        own_softmax = x_exp / x_exp_sum
        for i, y_t in enumerate(y_true):
            softmax = own_softmax[i, int(y_t)]
            mis.append(np.log2(softmax))
        estimated_mi = np.nanmean(mis)
        estimated_mi = np.nanmax([estimated_mi, 0.0])
    else:
        logger.error("All rows were nan, so cannot estimate mutual information")
        estimated_mi = 0.0
    return estimated_mi


def log_loss_estimation(y_true, y_pred):
    """
        Estimates mutual information by evaluating the log-loss of the predicted probabilities and entropy of outputs
        using the class labels.

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
        logger.error("All rows were nan, so cannot estimate mutual information")
        estimated_mi = 0.0
    return estimated_mi
