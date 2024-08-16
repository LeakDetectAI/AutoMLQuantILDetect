"""The `bayes_search` package contains custom implementations and utilities for
Bayesian hyperparameter optimization.

This package extends functionality provided by `scikit-optimize`,
offering additional features such as logging, state-saving for optimizer
progress, and enhanced callback functions for tracking and resuming the
optimization process.
"""

from .bayes_search_cv import BayesSearchCV
from .bayes_search_utils import *
