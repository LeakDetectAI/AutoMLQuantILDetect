"""The `utilities` subpackage provides essential utility functions and
constants for streamlined operations.

It includes constants from the `_constants` module, dimensionality reduction models through
`create_dimensionality_reduction_model`, performance metrics, statistical tests for data analysis,
and general utility functions to facilitate various tasks in the package.
"""
from ._constants import *
from .dimensionality_reduction_techniques import create_dimensionality_reduction_model
from .metrics import *
from .statistical_tests import *
from ._utils import *
