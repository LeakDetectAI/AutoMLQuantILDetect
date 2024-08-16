"""The `automl` package provides tools for automated machine learning (AutoML),
offering pre-configured classifiers that leverage state-of-the-art AutoML
frameworks like AutoGluon and TabPFN.

These tools simplify the process of training, validating, and deploying
machine learning models with minimal manual intervention.
"""

from .autogluon_classifier import AutoGluonClassifier
from .tabpfn_classifier import AutoTabPFNClassifier
