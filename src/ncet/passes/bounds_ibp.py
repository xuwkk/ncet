"""Graph-based interval bound propagation."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ..errors import InvalidBoundsError, UnsupportedOperatorError
from ..ir import GraphIR, IRNode, analyze_capabilities, static_index, validate_ir


_SUPPORTED_OPS = frozenset(
    {
        "Input",
        "Linear",
        "Conv2d",
        "BatchNorm",
        "AdaptiveAvgPool2d",
        "AvgPool2d",
        "MaxPool2d",
        "Identity",
        "ElementwiseAffine",
        "ReLU",
        "LeakyReLU",
        "Add",
        "Sub",
        "Concat",
        "ReduceMean",
        "Flatten",
        "Reshape",
        "Permute",
        "Transpose",
        "GetItem",
        "Slice",
        "Output",
    }
)


@dataclass(frozen=True)
class Bounds:
    """Elementwise lower and upper bounds for one graph tensor."""

    lower: np.ndarray
    upper: np.ndarray


def propagate_bounds(
    graph: GraphIR,
    input_bounds: Mapping[str, Bounds],
) -> dict[str, Bounds]:
    """Propagate sound interval bounds through the supported graph operators.
    
    To use this low-level inferface, you can
    name = graph.inputs[0] # for one input

    bounds = propagate_bounds(
        graph,
        {
            name: Bounds(
                lower=lower,
                upper=upper,
            )
        },
    )    
    These names normally originate from the forward() argument names captured by FX:
    def forward(self, x, y): ... so the name will automatically be captured as 'x' and 'y'.
    
    The final high-level API should automatically bind positional bounds to graph.inputs.
    """
    validate_ir(graph)
    _require_supported_graph(graph)
    bounds = _validate_input_bounds(graph, input_bounds)

    for node in graph.nodes:
        if node.op_type in {"Input", "Output"}:
            continue
        # In the current implementation, each node can have multiple inputs, but only one output.
        inputs = tuple(bounds[name] for name in node.inputs)
        output_name = node.outputs[0]
        output = _propagate_node(graph, node, inputs)
        bounds[output_name] = _validate_bounds(
            output,
            graph.tensors[output_name].shape,
            output_name,
        )

    return bounds


def _propagate_node(
    graph: GraphIR,
    node: IRNode,
    inputs: tuple[Bounds, ...],
) -> Bounds:
    if node.op_type == "Linear":
        input_bounds = inputs[0]
        weight = graph.constants[node.attrs["weight"]]
        bias = graph.constants[node.attrs["bias"]]
        positive = np.maximum(weight, 0)
        negative = np.minimum(weight, 0)
        return Bounds(
            lower=(
                input_bounds.lower @ positive.T
                + input_bounds.upper @ negative.T
                + bias
            ),
            upper=(
                input_bounds.upper @ positive.T
                + input_bounds.lower @ negative.T
                + bias
            ),
        )

    if node.op_type == "Conv2d":
        return _conv2d_bounds(graph, node, inputs[0])

    if node.op_type == "BatchNorm":
        return _batchnorm_bounds(graph, node, inputs[0])

    if node.op_type == "AdaptiveAvgPool2d":
        return _adaptive_avgpool2d_bounds(node, inputs[0])

    if node.op_type == "AvgPool2d":
        return _avgpool2d_bounds(node, inputs[0])

    if node.op_type == "MaxPool2d":
        return _maxpool2d_bounds(node, inputs[0])

    if node.op_type == "Identity":
        return inputs[0]

    if node.op_type == "ElementwiseAffine":
        scale = graph.constants[node.attrs["scale"]]
        shift = graph.constants[node.attrs["shift"]]
        positive = np.maximum(scale, 0)
        negative = np.minimum(scale, 0)
        return Bounds(
            lower=(
                positive * inputs[0].lower
                + negative * inputs[0].upper
                + shift
            ),
            upper=(
                positive * inputs[0].upper
                + negative * inputs[0].lower
                + shift
            ),
        )

    if node.op_type == "ReLU":
        return Bounds(
            lower=np.maximum(inputs[0].lower, 0),
            upper=np.maximum(inputs[0].upper, 0),
        )

    if node.op_type == "LeakyReLU":
        negative_slope = node.attrs["negative_slope"]
        return Bounds(
            lower=np.where(
                inputs[0].lower >= 0,
                inputs[0].lower,
                negative_slope * inputs[0].lower,
            ),
            upper=np.where(
                inputs[0].upper >= 0,
                inputs[0].upper,
                negative_slope * inputs[0].upper,
            ),
        )

    if node.op_type in {"Add", "Sub"}:
        # PyTorch applies alpha to the second operand: x +/- alpha * y.
        coefficient = (
            node.attrs["alpha"]
            if node.op_type == "Add"
            else -node.attrs["alpha"]
        )
        positive = max(coefficient, 0)
        negative = min(coefficient, 0)
        return Bounds(
            lower=(
                inputs[0].lower
                + positive * inputs[1].lower
                + negative * inputs[1].upper
            ),
            upper=(
                inputs[0].upper
                + positive * inputs[1].upper
                + negative * inputs[1].lower
            ),
        )

    if node.op_type == "Concat":
        dim = node.attrs["dim"]
        return Bounds(
            lower=np.concatenate([item.lower for item in inputs], axis=dim),
            upper=np.concatenate([item.upper for item in inputs], axis=dim),
        )

    if node.op_type == "ReduceMean":
        options = {
            "axis": node.attrs["dims"],
            "keepdims": node.attrs["keepdim"],
        }
        return Bounds(
            lower=np.mean(inputs[0].lower, **options),
            upper=np.mean(inputs[0].upper, **options),
        )

    if node.op_type in {"Flatten", "Reshape"}:
        shape = graph.tensors[node.outputs[0]].shape
        return Bounds(
            lower=inputs[0].lower.reshape(shape),
            upper=inputs[0].upper.reshape(shape),
        )

    if node.op_type == "Permute":
        dims = node.attrs["dims"]
        return Bounds(
            lower=np.transpose(inputs[0].lower, axes=dims),
            upper=np.transpose(inputs[0].upper, axes=dims),
        )

    if node.op_type == "Transpose":
        dim0 = node.attrs["dim0"]
        dim1 = node.attrs["dim1"]
        return Bounds(
            lower=np.swapaxes(inputs[0].lower, dim0, dim1),
            upper=np.swapaxes(inputs[0].upper, dim0, dim1),
        )

    if node.op_type in {"GetItem", "Slice"}:
        index = static_index(node)
        return Bounds(
            lower=inputs[0].lower[index],
            upper=inputs[0].upper[index],
        )

    raise UnsupportedOperatorError(
        f"bound propagation does not support {node.op_type} node '{node.name}'"
    )


def _conv2d_bounds(
    graph: GraphIR,
    node: IRNode,
    input_bounds: Bounds,
) -> Bounds:
    """Propagate per-sample bounds through one affine Conv2d operation."""
    weight = torch.tensor(graph.constants[node.attrs["weight"]])
    bias = torch.tensor(
        graph.constants[node.attrs["bias"]],
        dtype=weight.dtype,
    )
    # GraphIR bounds use (C, H, W). Add a temporary batch-size-one dimension
    # only while reusing PyTorch's NCHW convolution implementation.
    lower = torch.as_tensor(input_bounds.lower, dtype=weight.dtype).unsqueeze(0)
    upper = torch.as_tensor(input_bounds.upper, dtype=weight.dtype).unsqueeze(0)
    positive = weight.clamp(min=0)
    negative = weight.clamp(max=0)
    options = {
        "stride": node.attrs["stride"],
        "padding": node.attrs["padding"],
        "dilation": node.attrs["dilation"],
        "groups": node.attrs["groups"],
    }

    return Bounds(
        lower=(
            F.conv2d(lower, positive, bias=bias, **options)
            + F.conv2d(upper, negative, **options)
        ).squeeze(0).numpy(),
        upper=(
            F.conv2d(upper, positive, bias=bias, **options)
            + F.conv2d(lower, negative, **options)
        ).squeeze(0).numpy(),
    )


def _batchnorm_bounds(
    graph: GraphIR,
    node: IRNode,
    input_bounds: Bounds,
) -> Bounds:
    """Propagate bounds through a fixed per-channel affine transform."""
    scale = graph.constants[node.attrs["scale"]]
    shift = graph.constants[node.attrs["shift"]]
    # scale.size is the channel dimension, so we need to broadcast it to the input shape
    # with all 1s for the remaining dimensions
    broadcast_shape = (scale.size,) + (1,) * (input_bounds.lower.ndim - 1)
    scale = scale.reshape(broadcast_shape)
    shift = shift.reshape(broadcast_shape)
    positive = np.maximum(scale, 0)
    negative = np.minimum(scale, 0)
    return Bounds(
        lower=(
            positive * input_bounds.lower
            + negative * input_bounds.upper
            + shift
        ),
        upper=(
            positive * input_bounds.upper
            + negative * input_bounds.lower
            + shift
        ),
    )


def _avgpool2d_bounds(node: IRNode, input_bounds: Bounds) -> Bounds:
    """Apply the monotone average-pooling map to both interval endpoints."""
    options = {
        "kernel_size": node.attrs["kernel_size"],
        "stride": node.attrs["stride"],
        "padding": node.attrs["padding"],
        "ceil_mode": node.attrs["ceil_mode"],
        "count_include_pad": node.attrs["count_include_pad"],
        "divisor_override": node.attrs["divisor_override"],
    }
    lower = torch.as_tensor(input_bounds.lower).unsqueeze(0)  # Add a temporary batch-size-one dimension
    upper = torch.as_tensor(input_bounds.upper).unsqueeze(0)

    return Bounds(
        lower=F.avg_pool2d(lower, **options).squeeze(0).numpy(),
        upper=F.avg_pool2d(upper, **options).squeeze(0).numpy(),
    )


def _adaptive_avgpool2d_bounds(
    node: IRNode,
    input_bounds: Bounds,
) -> Bounds:
    """Apply the monotone adaptive average to both interval endpoints."""
    output_size = node.attrs["output_size"]
    lower = torch.as_tensor(input_bounds.lower).unsqueeze(0)
    upper = torch.as_tensor(input_bounds.upper).unsqueeze(0)
    return Bounds(
        lower=F.adaptive_avg_pool2d(lower, output_size).squeeze(0).numpy(),
        upper=F.adaptive_avg_pool2d(upper, output_size).squeeze(0).numpy(),
    )


def _maxpool2d_bounds(node: IRNode, input_bounds: Bounds) -> Bounds:
    """Apply the monotone maximum-pooling map to both interval endpoints.
    
    input_bounds -> node operation -> output_bounds
    """
    
    options = {
        "kernel_size": node.attrs["kernel_size"],
        "stride": node.attrs["stride"],
        "padding": node.attrs["padding"],
        "dilation": node.attrs["dilation"],
        "ceil_mode": node.attrs["ceil_mode"],
    }
    lower = torch.as_tensor(input_bounds.lower).unsqueeze(0)
    upper = torch.as_tensor(input_bounds.upper).unsqueeze(0)

    return Bounds(
        lower=F.max_pool2d(lower, **options).squeeze(0).numpy(),
        upper=F.max_pool2d(upper, **options).squeeze(0).numpy(),
    )


def _validate_input_bounds(
    graph: GraphIR,
    input_bounds: Mapping[str, Bounds], # {tensor_name: Bounds}
) -> dict[str, Bounds]:
    expected = set(graph.inputs)   # (tensor_name1, tensor_name2, ...)
    provided = set(input_bounds)   # {tensor_name1, tensor_name2, ...}
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing or unknown:
        raise InvalidBoundsError(
            f"input bounds do not match graph inputs: "
            f"missing={missing}, unknown={unknown}"
        )

    return {
        name: _validate_bounds(
            input_bounds[name],
            graph.tensors[name].shape,
            name,
        )
        for name in graph.inputs
    }


def _validate_bounds(
    bounds: Bounds,
    expected_shape: tuple[int, ...],
    tensor_name: str,
) -> Bounds:
    if not isinstance(bounds, Bounds):
        raise InvalidBoundsError(
            f"bounds for tensor '{tensor_name}' must be a Bounds instance"
        )

    lower = np.asarray(bounds.lower)
    upper = np.asarray(bounds.upper)
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise InvalidBoundsError(
            f"bounds for tensor '{tensor_name}' have shapes "
            f"{lower.shape} and {upper.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise InvalidBoundsError(
            f"bounds for tensor '{tensor_name}' must be finite"
        )
    if np.any(lower > upper):
        raise InvalidBoundsError(
            f"lower bound exceeds upper bound for tensor '{tensor_name}'"
        )

    return Bounds(lower=lower, upper=upper)


def _require_supported_graph(graph: GraphIR) -> None:
    report = analyze_capabilities(graph, _SUPPORTED_OPS)
    if report.supported:
        return

    unsupported = ", ".join(
        f"{node.name} ({node.op_type})" for node in report.unsupported_nodes
    )
    raise UnsupportedOperatorError(
        f"bound propagation does not support graph nodes: {unsupported}"
    )
