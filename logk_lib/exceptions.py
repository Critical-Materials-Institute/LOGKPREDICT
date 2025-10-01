"""Custom exceptions for LOGKPREDICT."""


class LogKPredictError(Exception):
    """Base exception for LOGKPREDICT errors."""

    pass


class ModelNotFoundError(LogKPredictError):
    """Raised when the ML model file cannot be found."""

    pass


class InvalidInputError(LogKPredictError):
    """Raised when input data is invalid or malformed."""

    pass


class ChempropError(LogKPredictError):
    """Raised when Chemprop prediction fails."""

    pass


class MolecularProcessingError(LogKPredictError):
    """Raised when molecular processing fails."""

    pass


class EnvironmentError(LogKPredictError):
    """Raised when required environment variables are missing."""

    pass
