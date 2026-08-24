"""Canonical graph intermediate representation."""

from .capability import CapabilityReport, analyze_capabilities
from .graph import GraphIR, IRNode, TensorSpec
from .indexing import static_index
from .validate import validate_ir

__all__ = [
    "CapabilityReport",
    "GraphIR",
    "IRNode",
    "TensorSpec",
    "analyze_capabilities",
    "static_index",
    "validate_ir",
]
