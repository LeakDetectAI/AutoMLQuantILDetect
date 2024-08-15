"""Provides utility functions for dataset handling, operations, and
preprocessing."""
import logging

import numpy as np

__all__ = ["GEN_TYPES", "FACTOR", "LABEL_COL", "generate_samples_per_class", "clean_class_label", "pdf"]


GEN_TYPES = ["single", "multiple"]
"""
    List of supported generation types for class imbalance:
    
    - `single`: Imbalance is applied to one class.
    - `multiple`: Imbalance is distributed across multiple classes.
"""

FACTOR = 1.5
"""A constant factor used for scaling or other operations."""

LABEL_COL = "label"
"""Default label column name used in datasets."""


def generate_samples_per_class(n_classes, samples=1000, imbalance=0.05, gen_type="single", logger=None, verbose=1):
    """Generate the number of samples per class with a specified imbalance.

    This function calculates the number of samples for each class based on the provided imbalance ratio and the generation type.
    It supports both binary and multi-class scenarios, allowing the user to specify whether the imbalance should be distributed
    across a single class or multiple classes.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset.

    samples : int, default=1000
        The total number of samples across all classes.

    imbalance : float, default=0.05
        The proportion of samples in the minority class (or classes if `gen_type` is "multiple"). The value must be less than or equal to 1/n_classes.

    gen_type : str, default="single"
        The type of imbalance generation:
        - "single": Imbalance is applied to one class.
        - "multiple": Imbalance is distributed across multiple classes.

    logger : logging.Logger, optional
        Logger object for logging output. If None, a default logger is created.

    verbose : int, default=1
        Verbosity level. If 1, logging information is displayed.

    Returns
    -------
    samples_per_class : dict
        A dictionary where the keys are class labels (as strings) and the values are the number of samples for each class.

    Raises
    ------
    ValueError
        If the imbalance ratio is greater than 1/n_classes or if the generation type is not recognized.
    """
    if logger is None:
        logger = logging.getLogger("Generate Samples")
    if verbose:
        logger.info("###############################################################################")
    if imbalance > 1 / n_classes:
        raise ValueError(
            f"The imbalance {np.around(imbalance, 2)} for a class cannot be more than uniform {1 / n_classes}")
    if gen_type not in GEN_TYPES:
        raise ValueError(f"Generation type {gen_type} not defined {GEN_TYPES}")
    assert (n_classes == 2) == (gen_type == "single") or n_classes > 2
    samples_per_class = {}
    n_total_instances = samples * n_classes
    if gen_type == "single":
        for n_c in range((n_classes - 1)):
            imb = ((1 - imbalance) / (n_classes - 1))
            n_samples = imb * n_total_instances
            samples_per_class[str(n_c)] = int(np.ceil(n_samples))
            if verbose:
                logger.info(f"Class {n_c + 1} calculated {n_samples / n_total_instances}")
        samples_per_class[str(n_classes - 1)] = n_total_instances - sum(samples_per_class.values())
        v = samples_per_class[str(n_classes - 1)] / n_total_instances
        if verbose:
            logger.info(f"Class {n_classes} calculated {np.around(v, 2)}")
    if gen_type == "multiple":
        for n_c in range((n_classes - 1)):
            n_samples = imbalance * n_total_instances
            samples_per_class[str(n_c)] = int(np.ceil(n_samples))
            if verbose:
                logger.info(f"Class {n_c + 1} calculated {n_samples / n_total_instances}")
        samples_per_class[str(n_classes - 1)] = n_total_instances - sum(samples_per_class.values())
        v = samples_per_class[str(n_classes - 1)] / n_total_instances
        if verbose:
            logger.info(f"Class {n_classes} calculated {np.around(v, 2)}")
    if verbose:
        logger.info(f"Imbalanced {np.around(imbalance, 2)} samples_per_class {samples_per_class}")
    return samples_per_class


def clean_class_label(string):
    """Clean and format a class label string.

    This function processes a string by replacing underscores with spaces, capitalizing each word,
    and removing any extra spaces to make the label more readable and formatted consistently.

    Parameters
    ----------
    string : str
        The input class label string to be cleaned and formatted.

    Returns
    -------
    str
        The cleaned and formatted class label string.

    Example
    -------
    >>> clean_class_label("class_label_example")
    `Class Label Example`

    Notes
    -----
    This function is useful for formatting class labels in a readable way, especially when they are
    generated automatically or retrieved from a source where they are not human-readable.
    """
    string = ' '.join(string.split('_')).title()
    string = string.replace("  ", " ")
    return string


def pdf(dist, x):
    """Compute the probability density function (PDF) for the given
    distribution and input data.

    Parameters
    ----------
    dist : scipy.stats._multivariate.multivariate_normal_frozen
        The multivariate normal distribution object.
    x : array-like of shape (n_samples, n_features)
        Input data for which the PDF is computed.

    Returns
    -------
    log_dist_samples: array-like
        Probability density values for the input data.
    """
    log_dist_samples = np.exp(dist.logpdf(x))
    return log_dist_samples
