"""Neural-network Constraint Embedding Toolkit."""

from .errors import (
    ExactnessContractError,
    GraphCaptureError,
    InvalidBoundsError,
    NCETError,
    UnsupportedOperatorError,
)
from .backend import MILPEncoding
from .builder import form_milp
from .passes import Bounds

__version__ = "0.1.0"

__all__ = [
    "ExactnessContractError",
    "Bounds",
    "GraphCaptureError",
    "InvalidBoundsError",
    "MILPEncoding",
    "NCETError",
    "UnsupportedOperatorError",
    "form_milp",
    "__version__",
]
