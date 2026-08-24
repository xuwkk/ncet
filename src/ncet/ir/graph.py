"""Data structures for the canonical graph IR."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

"""
Residual block example:
[Input IRNode]
      │
      ▼
 tensor "x"
      │
      ▼
 network operations
      │
      ▼
 tensor "y"
      │
      ▼
[Output IRNode]
TensorSpec: x, y
IRNode: Input, network operations, Output
The node and its output tensor may share the same name, 
but they represent different concepts
A graph starts with an Input node and ends with an Output node
"""

@dataclass(frozen=True)
class TensorSpec:
    """Static information about one graph tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    producer: str | None   # Node name


@dataclass(frozen=True)
class IRNode:
    """One canonical operation in topological order."""

    name: str
    op_type: str
    inputs: tuple[str, ...]   # tensor names
    outputs: tuple[str, ...]  # tensor names
    attrs: Mapping[str, Any]  # e.g., module path


@dataclass
class GraphIR:
    """Canonical static computation graph."""

    nodes: list[IRNode]
    inputs: list[str]                # tensor names: x
    outputs: list[str]               # tensor names: y
    tensors: dict[str, TensorSpec]   # mapping from tensor name to its specification
    constants: dict[str, np.ndarray] = field(default_factory=dict) # fixed weights, biases, and buffers
