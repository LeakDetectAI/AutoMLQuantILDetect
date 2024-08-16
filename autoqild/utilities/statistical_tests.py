"""Implementation of paired t-test and wilcoxon_signed_rank_test used to detect
leakage using blind classifiers."""

import logging

import numpy as np
from scipy.stats import t, wilcoxon

__all__ = ["wilcoxon_signed_rank_test", "paired_ttest"]


def wilcoxon_signed_rank_test(accuracies, accuracies2, alternative="two-sided", verbose=False):
    """Performs the Wilcoxon signed-rank test on two sets of accuracies.

    Parameters
    ----------
    accuracies : ndarray
        First set of accuracy values.
    accuracies2 : ndarray
        Second set of accuracy values.
    alternative : str, optional
        Defines the alternative hypothesis (default is "two-sided").
    verbose : bool, optional
        If True, outputs additional logging information (default is False).

    Returns
    -------
    p_value : float
        The p-value from the Wilcoxon signed-rank test.
    """
    logger = logging.getLogger("Wilcoxon-Signed_Rank")

    try:
        _, p_value = wilcoxon(accuracies, accuracies2, correction=True, alternative=alternative)
    except Exception as e:
        if verbose:
            logger.info("Accuracies are exactly same {}".format(str(e)))
        p_value = 1.0
    return p_value


def paired_ttest(
    x1,
    x2,
    n_training_folds,
    n_test_folds,
    correction=True,
    alternative="two-sided",
    verbose=False,
):
    """Performs a paired t-test on two sets of values with and without
    correction.

    Parameters
    ----------
    x1 : ndarray
        First set of values.
    x2 : ndarray
        Second set of values.
    n_training_folds : int
        Number of training folds.
    n_test_folds : int
        Number of test folds.
    correction : bool, optional
        If True, applies a correction to the variance (default is True).
    alternative : str, optional
        Defines the alternative hypothesis (default is "two-sided").
    verbose : bool, optional
        If True, outputs additional logging information (default is False).

    Returns
    -------
    p_value : float
        The p-value from the paired t-test.
    """
    logger = logging.getLogger("Paired T-Test")
    n = len(x1)
    df = n - 1
    diff = [(x1[i] - x2[i]) for i in range(n)]
    # Compute the mean of differences
    d_bar = np.mean(diff)
    # compute the variance of differences
    sigma2 = np.var(diff, ddof=1)
    if sigma2 == 0.0:
        sigma2 = 1e-30
        if verbose:
            logger.info("Correcting the sigma")

    if correction:
        if verbose:
            logger.info("With the correction option")
    if verbose:
        logger.info("D_bar {} Variance {} Sigma {}".format(d_bar, sigma2, np.sqrt(sigma2)))

    # compute the modified variance
    if correction:
        sigma2 = sigma2 * (1 / n + n_test_folds / n_training_folds)
    else:
        sigma2 = sigma2 / n

    # compute the t_static
    with np.errstate(divide="ignore", invalid="ignore"):
        t_static = np.divide(d_bar, np.sqrt(sigma2))

    # Compute p-value and plot the results
    if alternative == "less":
        p_value = t.cdf(t_static, df)
    elif alternative == "greater":
        p_value = t.sf(t_static, df)
    elif alternative == "two-sided":
        p_value = 2 * t.sf(np.abs(t_static), df)
    if verbose:
        logger.info(
            "Final Variance {} Sigma {} t_static {} p {}".format(
                sigma2, np.sqrt(sigma2), t_static, p_value
            )
        )
        logger.info(
            "np.isnan(p) {}, np.isinf {},  d_bar == 0 {}, sigma2_mod == 0 {}, np.isinf(t_static) {}, "
            "np.isnan(t_static) {}".format(
                np.isnan(p_value),
                np.isinf(p_value),
                d_bar == 0,
                sigma2 == 0,
                np.isinf(t_static),
                np.isnan(t_static),
            )
        )
    if (
        np.isnan(p_value)
        or np.isinf(p_value)
        or d_bar == 0
        or sigma2 == 0
        or np.isinf(t_static)
        or np.isnan(t_static)
    ):
        p_value = 1.0
    return p_value
