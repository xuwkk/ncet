"""Public exceptions raised by NCET."""


class NCETError(Exception):
    """Base class for package-specific errors."""


class UnsupportedOperatorError(NCETError):
    """Raised when a graph contains an unsupported operator."""


class GraphCaptureError(NCETError):
    """Raised when a PyTorch model cannot be captured as a static graph."""


class InvalidBoundsError(NCETError):
    """Raised when bounds are missing, inconsistent, or non-finite."""


class ExactnessContractError(NCETError):
    """Raised when a model violates the exact-encoding contract."""
