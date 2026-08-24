"""LAPSO-style public builder for NCET encodings."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .backend import EncodingOptions, MILPEncoding, encode_cvxpy
from .errors import InvalidBoundsError
from .frontend import capture_graph, normalize_graph, propagate_shapes
from .passes import Bounds, propagate_bounds


BoundPair = tuple[Any, Any]
BoundLike = Bounds | BoundPair # Bounds or a (lower, upper) tuple
# single bound, list of bounds, or mapping of tensor names to bounds
InputBounds = BoundLike | Sequence[Bounds] | Mapping[str, BoundLike] 


def form_milp(
    model: nn.Module,
    input_bounds: InputBounds,
    *,
    relu_binary_mode: Literal["full", "reduced"] = "reduced",
) -> MILPEncoding:
    """Capture a PyTorch model and return its exact NCET formulation."""
    traced = capture_graph(model)
    input_names = [
        node.name for node in traced.graph.nodes if node.op == "placeholder"
    ]
    
    # Check the input bounds with the graphFX input names
    ordered_bounds = _bind_input_bounds(input_names, input_bounds) # List of Bounds of the inputs
    example_inputs = _example_inputs(model, ordered_bounds) # List of input tensors
    propagate_shapes(traced, *example_inputs)               # Add tensor shape and dtype information to the graph
    graph = normalize_graph(traced)
    bound_mapping = dict(zip(graph.inputs, ordered_bounds))
    bounds = propagate_bounds(graph, bound_mapping)  # Dictionary of tensor names to Bounds
    options = EncodingOptions(relu_binary_mode=relu_binary_mode)
    return encode_cvxpy(graph, bounds, options)


def _bind_input_bounds(
    input_names: list[str],
    input_bounds: InputBounds,
) -> list[Bounds]:
    """Return a list of Bounds objects for the input names."""
    if isinstance(input_bounds, Mapping):
        # Mapping of tensor names to BoundLike
        expected = set(input_names)
        provided = set(input_bounds)
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        if missing or unknown:
            raise InvalidBoundsError(
                f"input bounds do not match model inputs: "
                f"missing={missing}, unknown={unknown}"
            )
        return [_as_bounds(input_bounds[name]) for name in input_names]

    if isinstance(input_bounds, Bounds) or isinstance(input_bounds, tuple):
        # Single Bounds or (lower, upper) tuple
        if len(input_names) != 1:
            raise InvalidBoundsError(
                f"model has {len(input_names)} inputs; provide one Bounds per input"
            )
        return [_as_bounds(input_bounds)]
    
    # List of Bounds
    # ! Make sure the bounds are in the same order as the input names in forward() method
    ordered = list(input_bounds)
    if len(ordered) != len(input_names):
        raise InvalidBoundsError(
            f"model has {len(input_names)} inputs but received "
            f"{len(ordered)} bounds"
        )
    return [_as_bounds(value) for value in ordered]


def _as_bounds(value: BoundLike) -> Bounds:
    """Return standard Bounds object"""
    if isinstance(value, Bounds):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise InvalidBoundsError(
            "each input bound must be Bounds or a (lower, upper) tuple"
        )
    return Bounds(_as_numpy(value[0]), _as_numpy(value[1]))


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _example_inputs(model: nn.Module, bounds: list[Bounds]) -> list[torch.Tensor]:
    """Create batch-size-one tensors from public per-sample bounds."""
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
    device = parameter.device if parameter is not None else torch.device("cpu")
    return [
        torch.zeros((1, *bound.lower.shape), dtype=dtype, device=device)
        for bound in bounds
    ]
