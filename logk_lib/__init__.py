"""
LOGK Library: Core functionality for predicting stability constants.

This library provides the core functionality used by the LOGKPREDICT and LOGKPREDICT0
executables, while maintaining compatibility with HostDesigner.
"""

__version__ = "2.0.0"
__author__ = "Federico Zahariev, Marilú Pérez García"
__email__ = "fzahariev@iastate.edu"

from .exceptions import InvalidInputError, LogKPredictError, ModelNotFoundError
from .molecular_processing import MolecularProcessor
from .predictor import LogKPredictor

__all__ = [
    "LogKPredictor",
    "MolecularProcessor",
    "LogKPredictError",
    "ModelNotFoundError",
    "InvalidInputError",
]
