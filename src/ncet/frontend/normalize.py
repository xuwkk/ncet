"""Normalize a shape-propagated PyTorch FX graph into the canonical graph IR in NCET.

This module contains the PyTorch-specific spelling and metadata rules. Later
passes only need to understand canonical operators such as ``Linear``, ``Add``,
and ``Reshape``.
"""

import operator
from numbers import Real
from typing import Any

import numpy as np
import torch
from torch import fx, nn
from torch.nn import functional as F

from ..errors import GraphCaptureError, UnsupportedOperatorError
from ..ir import GraphIR, IRNode, TensorSpec, validate_ir
from .fx import validate_graph


def normalize_graph(graph_module: fx.GraphModule) -> GraphIR:
    """Convert a shape-propagated FX graph into a validated static ``GraphIR``.

    Each FX operation becomes an ``IRNode``. Every graph value produced by an
    input or operation receives a separate ``TensorSpec`` with its static
    per-sample shape and dtype. Raw FX metadata contains the temporary leading
    batch dimension used for shape propagation; GraphIR does not.
    """
    validate_graph(graph_module)
    # Attributes for GraphIR
    nodes: list[IRNode] = []                  # Canonical operations in topological order.
    inputs: list[str] = []                    # output tensors of Input nodes
    outputs: list[str] = []                   # input tensors of Output nodes
    tensors: dict[str, TensorSpec] = {}       # Metadata for every graph tensor
    constants: dict[str, np.ndarray] = {}     # Lifted parameters and buffers

    for node in graph_module.graph.nodes:
        # FX nodes are topologically ordered: every producer appears before its
        # consumers, even when the model contains branches or residual edges.
        # Obtain the input tensor names for the current node
        input_names = _input_names(node)  # For example, ("linear", "x").

        if node.op == "placeholder":
            inputs.append(node.name)
            # An Input operation has no incoming tensor and produces one graph
            # tensor. ``producer=None`` marks that tensor as externally supplied.
            tensors[node.name] = _tensor_spec(node, producer=None)
            nodes.append(_ir_node(node, "Input", (), (node.name,)))
            continue

        if node.op == "output":
            # The Output operation consumes existing tensors but does not create
            # another tensor. Multiple returned values produce multiple names.
            outputs.extend(input_names)  # For example, ["add", "sub"].
            nodes.append(_ir_node(node, "Output", input_names, ()))
            continue

        op_type, attrs = _canonical_operation(graph_module, node, constants)
        # An operation and its output tensor currently share the FX node name,
        # but IRNode and TensorSpec represent different concepts.
        tensors[node.name] = _tensor_spec(node, producer=node.name)
        nodes.append(
            _ir_node(node, op_type, input_names, (node.name,), attrs=attrs)
        )

    graph = GraphIR(
        nodes=nodes,           # graph nodes in topological order
        inputs=inputs,         # output tensors of Input nodes
        outputs=outputs,       # input tensors of Output nodes
        tensors=tensors,       # metadata for every graph tensor
        constants=constants,   # lifted parameters and buffers
    )
    validate_ir(graph)
    return graph


def _canonical_operation(
    graph_module: fx.GraphModule,
    node: fx.Node,
    constants: dict[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    """Map one FX operation spelling to a canonical operator and attributes
    
    Return the canonical operator name and the attributes for the operation.
    - The attributes are the canonicalized parameters for the operation such as
    start_dim, end_dim for Flatten, dims for Permute, weight name and bias name for Linear, etc.
    - For parameterized modules such as Linear and Conv2d, fixed parameters are
    also added to `constants`, while the returned attributes store their names.

    For example, ``nn.ReLU``, ``F.relu``, ``torch.relu``, and
    ``Tensor.relu()`` all become canonical ``ReLU`` operations in IR.
    """
    if node.op == "call_module":
        # A call_module target is a dotted path in GraphModule, so inspect the
        # concrete submodule type to determine its mathematical operation.
        module = graph_module.get_submodule(str(node.target))
        if isinstance(module, nn.Linear):
            return "Linear", _linear_attrs(str(node.target), module, constants)
        if isinstance(module, nn.Conv2d):
            return "Conv2d", _conv2d_attrs(
                node.name,
                str(node.target),
                module,
                constants,
            )
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return "BatchNorm", _batchnorm_attrs(
                node,
                str(node.target),
                module,
                constants,
            )
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return "AdaptiveAvgPool2d", _adaptive_avgpool2d_attrs(
                node,
                module.output_size,
            )
        if isinstance(module, nn.AvgPool2d):
            return "AvgPool2d", _avgpool2d_attrs(
                node.name,
                module.kernel_size,
                module.stride,
                module.padding,
                module.ceil_mode,
                module.count_include_pad,
                module.divisor_override,
            )
        if isinstance(module, nn.MaxPool2d):
            return "MaxPool2d", _maxpool2d_attrs(
                node.name,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.ceil_mode,
                module.return_indices,
            )
        if isinstance(module, nn.Identity):
            return "Identity", {}
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            if module.training:
                raise UnsupportedOperatorError(
                    f"Dropout node '{node.name}' must be in evaluation mode"
                )
            return "Identity", {}
        if isinstance(module, nn.ReLU):
            return "ReLU", {}
        if isinstance(module, nn.Flatten):
            return "Flatten", _flatten_attrs(
                node,
                module.start_dim,
                module.end_dim,
            )

    if node.op == "call_function":
        # A call_function target is the callable object recorded by FX.
        if node.target in {operator.add, torch.add}:
            return "Add", _unit_alpha_attrs(node, "Add")
        if node.target in {operator.sub, torch.sub, torch.subtract}:
            return "Sub", _unit_alpha_attrs(node, "Sub")
        if node.target in {F.relu, torch.relu}:
            return "ReLU", {}
        if node.target in {
            F.dropout,
            F.dropout1d,
            F.dropout2d,
            F.dropout3d,
        }:
            return "Identity", _dropout_call_attrs(node)
        if node.target is F.adaptive_avg_pool2d:
            output_size = (
                node.args[1]
                if len(node.args) > 1
                else node.kwargs["output_size"]
            )
            return "AdaptiveAvgPool2d", _adaptive_avgpool2d_attrs(
                node,
                output_size,
            )
        if node.target is F.avg_pool2d:
            return "AvgPool2d", _avgpool2d_call_attrs(node)
        if node.target is F.max_pool2d:
            return "MaxPool2d", _maxpool2d_call_attrs(node)
        if node.target in {torch.cat, torch.concat, torch.concatenate}:
            return "Concat", _concat_attrs(node)
        if node.target is torch.flatten:
            return "Flatten", _flatten_call_attrs(node)
        if node.target is torch.mean:
            return "ReduceMean", _reduce_mean_attrs(node)
        if node.target is torch.reshape:
            return "Reshape", _reshape_attrs(node)
        if node.target is torch.squeeze:
            return "Reshape", _squeeze_attrs(node)
        if node.target is torch.unsqueeze:
            return "Reshape", _unsqueeze_attrs(node)
        if node.target is torch.permute:
            return "Permute", _permute_attrs(node)
        if node.target is torch.transpose:
            return "Transpose", _transpose_attrs(node)
        if node.target is operator.getitem:
            return _index_operation(node)

    if node.op == "call_method":
        # A call_method target is a method-name string such as "add" or "view".
        if node.target == "add":
            return "Add", _unit_alpha_attrs(node, "Add")
        if node.target in {"sub", "subtract"}:
            return "Sub", _unit_alpha_attrs(node, "Sub")
        if node.target == "relu":
            return "ReLU", {}
        if node.target == "flatten":
            return "Flatten", _flatten_call_attrs(node)
        if node.target == "mean":
            return "ReduceMean", _reduce_mean_attrs(node)
        if node.target in {"reshape", "view"}:
            return "Reshape", _reshape_attrs(node)
        if node.target == "squeeze":
            return "Reshape", _squeeze_attrs(node)
        if node.target == "unsqueeze":
            return "Reshape", _unsqueeze_attrs(node)
        if node.target == "permute":
            return "Permute", _permute_attrs(node)
        if node.target == "transpose":
            return "Transpose", _transpose_attrs(node)

    raise _unsupported_node(node)


def _dropout_call_attrs(node: fx.Node) -> dict[str, Any]:
    """Accept a functional Dropout call only when it is a pure identity."""
    training = (
        node.args[2]
        if len(node.args) > 2
        else node.kwargs.get("training", True)
    )
    inplace = (
        node.args[3]
        if len(node.args) > 3
        else node.kwargs.get("inplace", False)
    )
    if training is not False:
        raise UnsupportedOperatorError(
            f"Dropout node '{node.name}' must use training=False"
        )
    if inplace is not False:
        raise UnsupportedOperatorError(
            f"Dropout node '{node.name}' must use inplace=False"
        )
    return {}


def _adaptive_avgpool2d_attrs(
    node: fx.Node,
    output_size: Any,
) -> dict[str, tuple[int, int]]:
    """Resolve AdaptiveAvgPool2d to one static per-sample output size."""
    input_shape = _sample_shape(node.all_input_nodes[0])
    output_shape = _sample_shape(node)
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise UnsupportedOperatorError(
            f"unsupported AdaptiveAvgPool2d shape at node '{node.name}': "
            f"input={input_shape}, output={output_shape}"
        )

    if type(output_size) is int:
        requested = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        requested = tuple(output_size)
    else:
        raise UnsupportedOperatorError(
            f"unsupported AdaptiveAvgPool2d output_size at node "
            f"'{node.name}': {output_size!r}"
        )

    if any(
        value is not None and (type(value) is not int or value <= 0)
        for value in requested
    ):
        raise UnsupportedOperatorError(
            f"unsupported AdaptiveAvgPool2d output_size at node "
            f"'{node.name}': {output_size!r}"
        )

    resolved = tuple(
        input_size if requested_size is None else requested_size
        for input_size, requested_size in zip(input_shape[1:], requested)
    )
    if output_shape[0] != input_shape[0] or output_shape[1:] != resolved:
        raise UnsupportedOperatorError(
            f"inconsistent AdaptiveAvgPool2d shape at node '{node.name}': "
            f"expected={(input_shape[0], *resolved)}, got={output_shape}"
        )
    return {"output_size": resolved}


def _avgpool2d_call_attrs(node: fx.Node) -> dict[str, Any]:
    """Read AvgPool2d parameters from a functional FX call."""
    kernel_size = (
        node.args[1] if len(node.args) > 1 else node.kwargs["kernel_size"]
    )
    stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride")
    padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
    ceil_mode = (
        node.args[4] if len(node.args) > 4 else node.kwargs.get("ceil_mode", False)
    )
    count_include_pad = (
        node.args[5]
        if len(node.args) > 5
        else node.kwargs.get("count_include_pad", True)
    )
    divisor_override = (
        node.args[6]
        if len(node.args) > 6
        else node.kwargs.get("divisor_override")
    )
    return _avgpool2d_attrs(
        node.name,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def _avgpool2d_attrs(
    node_name: str,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    ceil_mode: Any,
    count_include_pad: Any,
    divisor_override: Any,
) -> dict[str, Any]:
    """Canonicalize the AvgPool2d parameters supported by the exact backend."""
    if ceil_mode:
        raise UnsupportedOperatorError(
            f"unsupported AvgPool2d ceil_mode at node '{node_name}': expected False"
        )
    if divisor_override is not None:
        raise UnsupportedOperatorError(
            f"unsupported AvgPool2d divisor_override at node '{node_name}': "
            f"expected None, got {divisor_override}"
        )

    kernel = _spatial_pair(kernel_size)
    return {
        "kernel_size": kernel,
        # By default, stride is the same as kernel_size in PyTorch
        "stride": kernel if stride is None else _spatial_pair(stride), 
        "padding": _spatial_pair(padding),
        "ceil_mode": False,
        "count_include_pad": bool(count_include_pad),
        "divisor_override": None,
    }


def _maxpool2d_call_attrs(node: fx.Node) -> dict[str, Any]:
    """Read MaxPool2d parameters from a functional FX call."""
    kernel_size = (
        node.args[1] if len(node.args) > 1 else node.kwargs["kernel_size"]
    )
    stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride")
    padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
    dilation = node.args[4] if len(node.args) > 4 else node.kwargs.get("dilation", 1)
    ceil_mode = (
        node.args[5] if len(node.args) > 5 else node.kwargs.get("ceil_mode", False)
    )
    return_indices = (
        node.args[6]
        if len(node.args) > 6
        else node.kwargs.get("return_indices", False)
    )
    return _maxpool2d_attrs(
        node.name,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        return_indices,
    )


def _maxpool2d_attrs(
    node_name: str,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    ceil_mode: Any,
    return_indices: Any,
) -> dict[str, Any]:
    """Canonicalize the MaxPool2d parameters supported by NCET."""
    canonical_dilation = _spatial_pair(dilation)
    if canonical_dilation != (1, 1):
        raise UnsupportedOperatorError(
            f"unsupported MaxPool2d dilation at node '{node_name}': "
            f"expected (1, 1), got {canonical_dilation}"
        )
    if ceil_mode:
        raise UnsupportedOperatorError(
            f"unsupported MaxPool2d ceil_mode at node '{node_name}': expected False"
        )
    if return_indices:
        raise UnsupportedOperatorError(
            f"unsupported MaxPool2d return_indices at node '{node_name}': "
            "expected False"
        )

    kernel = _spatial_pair(kernel_size)
    return {
        "kernel_size": kernel,
        "stride": kernel if stride is None else _spatial_pair(stride),
        "padding": _spatial_pair(padding),
        "dilation": canonical_dilation,
        "ceil_mode": False,
        "return_indices": False,
    }


def _spatial_pair(value: Any) -> tuple[int, int]:
    """Represent an already shape-validated scalar or pair as a 2-tuple."""
    if isinstance(value, int):
        return value, value
    values = tuple(value)
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _unit_alpha_attrs(node: fx.Node, op_type: str) -> dict[str, int]:
    """Accept the unit-scaled Add/Sub semantics represented by the current IR.

    PyTorch Add/Sub can scale the second operand through ``alpha``. Canonical
    Add and Sub currently mean only ``x + y`` and ``x - y``, so other values
    are rejected explicitly.
    """
    alpha = node.kwargs.get("alpha", 1)
    if not isinstance(alpha, Real) or alpha != 1:
        raise UnsupportedOperatorError(
            f"unsupported {op_type} alpha at node '{node.name}': "
            f"expected 1, got {alpha}"
        )
    return {"alpha": 1}


def _concat_attrs(node: fx.Node) -> dict[str, int]:
    """Normalize Concat to a non-negative per-sample dimension."""
    if node.kwargs.get("out") is not None:
        # ``out`` writes into caller-provided storage, which is outside the
        # package's mutation-free graph contract.
        raise UnsupportedOperatorError(
            f"unsupported Concat out argument at node '{node.name}'"
        )

    # Accept torch.cat((x, y), 1), torch.cat((x, y), dim=1), and the
    # default-dimension form torch.cat((x, y)).
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    if type(dim) is not int:
        raise UnsupportedOperatorError(
            f"unsupported Concat dim at node '{node.name}': expected int, got {dim!r}"
        )

    raw_dim = _canonical_dim(node, dim, len(_node_shape(node)), "Concat")
    return {"dim": _sample_dim(node, raw_dim, "Concat")}


def _flatten_call_attrs(node: fx.Node) -> dict[str, int]:
    """Read Flatten parameters from a function or tensor-method FX call."""
    # torch.flatten(x, 1, -1) records positional arguments; keyword spellings
    # and omitted arguments are recovered from kwargs and PyTorch defaults.
    start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get(
        "start_dim", 0
    )
    end_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get(
        "end_dim", -1
    )
    return _flatten_attrs(node, start_dim, end_dim)


def _flatten_attrs(
    node: fx.Node,
    start_dim: Any,
    end_dim: Any,
) -> dict[str, int]:
    """Canonicalize Flatten dimensions from any PyTorch spelling."""
    # nn.Flatten supplies module attributes
    # torch.flatten and Tensor.flatten supply args/kwargs; 
    # all spellings converge in IR.
    rank = _input_rank(node)
    start = _canonical_dim(node, start_dim, rank, "Flatten start_dim")
    end = _canonical_dim(node, end_dim, rank, "Flatten end_dim")
    if start > end:
        raise UnsupportedOperatorError(
            f"unsupported Flatten dimensions at node '{node.name}': "
            f"start_dim={start}, end_dim={end}"
        )
    return {
        "start_dim": _sample_dim(node, start, "Flatten"),
        "end_dim": _sample_dim(node, end, "Flatten"),
    }


def _reshape_attrs(node: fx.Node) -> dict[str, tuple[int, ...]]:
    """Represent View and Reshape by their resolved static output shape."""
    # ShapeProp has already resolved inferred dimensions such as -1, so the
    # backend receives one explicit target shape for both View and Reshape.
    return {"shape": _sample_shape(node)}


def _reduce_mean_attrs(node: fx.Node) -> dict[str, Any]:
    """Normalize a static mean over sample dimensions."""
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        raise UnsupportedOperatorError(
            f"unsupported ReduceMean at node '{node.name}': "
            "dim must be explicit to preserve batch dimension 0"
        )

    raw_dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    if not raw_dims:
        raise UnsupportedOperatorError(
            f"unsupported ReduceMean dims at node '{node.name}': {dim!r}"
        )

    rank = _input_rank(node)
    dims = tuple(
        _canonical_dim(node, item, rank, "ReduceMean") for item in raw_dims
    )
    if len(set(dims)) != len(dims):
        raise UnsupportedOperatorError(
            f"unsupported repeated ReduceMean dims at node '{node.name}': "
            f"{dim!r}"
        )

    keepdim = (
        node.args[2]
        if len(node.args) > 2
        else node.kwargs.get("keepdim", False)
    )
    if type(keepdim) is not bool:
        raise UnsupportedOperatorError(
            f"unsupported ReduceMean keepdim at node '{node.name}': "
            f"{keepdim!r}"
        )
    if node.kwargs.get("dtype") is not None:
        raise UnsupportedOperatorError(
            f"unsupported ReduceMean dtype at node '{node.name}'"
        )
    if node.kwargs.get("out") is not None:
        raise UnsupportedOperatorError(
            f"unsupported ReduceMean out argument at node '{node.name}'"
        )

    return {
        "dims": tuple(
            _sample_dim(node, item, "ReduceMean") for item in dims
        ),
        "keepdim": keepdim,
    }


def _squeeze_attrs(node: fx.Node) -> dict[str, tuple[int, ...]]:
    """Map an explicit sample-axis Squeeze to canonical Reshape."""
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        raise UnsupportedOperatorError(
            f"unsupported Squeeze at node '{node.name}': "
            "dim must be explicit to preserve batch dimension 0"
        )

    dims = dim if isinstance(dim, tuple) else (dim,)
    rank = _input_rank(node)
    canonical_dims = tuple(
        _canonical_dim(node, item, rank, "Squeeze") for item in dims
    )
    if 0 in canonical_dims:
        raise UnsupportedOperatorError(
            f"unsupported Squeeze at node '{node.name}': "
            "batch dimension 0 must remain unchanged"
        )
    # convert the Squeeze to a Reshape based on the input and output shape
    # we can determine the transformation
    return _reshape_attrs(node)


def _unsqueeze_attrs(node: fx.Node) -> dict[str, tuple[int, ...]]:
    """Map a sample-axis Unsqueeze to canonical Reshape."""
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    rank = _input_rank(node)
    canonical_dim = _canonical_insert_dim(node, dim, rank, "Unsqueeze")
    if canonical_dim == 0:
        raise UnsupportedOperatorError(
            f"unsupported Unsqueeze at node '{node.name}': "
            "cannot insert before batch dimension 0"
        )
    return _reshape_attrs(node)


def _permute_attrs(node: fx.Node) -> dict[str, tuple[int, ...]]:
    """Read and validate a complete permutation of the input dimensions."""
    # The first argument is the input tensor. Tensor.permute may record each
    # dimension separately, whereas torch.permute records one tuple/list.
    if len(node.args) > 2:
        raw_dims = node.args[1:]
    elif len(node.args) == 2:
        raw_dims = node.args[1]
    else:
        raw_dims = node.kwargs.get("dims")

    if not isinstance(raw_dims, (tuple, list)):
        raise UnsupportedOperatorError(
            f"unsupported Permute dims at node '{node.name}': {raw_dims!r}"
        )

    rank = _input_rank(node)
    dims = tuple(
        _canonical_dim(node, dim, rank, "Permute") for dim in raw_dims
    )
    # A valid permutation mentions every input dimension exactly once.
    if len(dims) != rank or set(dims) != set(range(rank)):
        raise UnsupportedOperatorError(
            f"unsupported Permute dims at node '{node.name}': {raw_dims!r}"
        )
    if dims[0] != 0:
        raise UnsupportedOperatorError(
            f"unsupported Permute at node '{node.name}': "
            "batch dimension 0 must remain first"
        )
    return {"dims": tuple(dim - 1 for dim in dims[1:])}  # remove the batch dimension

def _transpose_attrs(node: fx.Node) -> dict[str, int]:
    """Normalize the two dimensions exchanged by Transpose."""
    # These cases differ only in where FX records dim0 and dim1:
    # x.transpose(-1, -2), torch.transpose(x, -1, -2), and keyword arguments.
    dim0 = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0")
    dim1 = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1")
    rank = _input_rank(node)
    canonical_dim0 = _canonical_dim(node, dim0, rank, "Transpose dim0")
    canonical_dim1 = _canonical_dim(node, dim1, rank, "Transpose dim1")
    return {
        "dim0": _sample_dim(node, canonical_dim0, "Transpose"),
        "dim1": _sample_dim(node, canonical_dim1, "Transpose"),
    }


def _index_operation(node: fx.Node) -> tuple[str, dict[str, Any]]:
    """Classify and normalize one static ``operator.getitem`` tensor access.

    FX records indexing against the temporary batched tensor, while GraphIR
    represents one sample. This function therefore performs three steps:

    1. verify that the original indexing syntax leaves batch axis 0 unchanged;
    2. resolve Python syntax into one explicit canonical action per FX axis;
    3. remove the leading full-batch slice before storing the sample-level IR.

    For example, batched ``x[:, 0]`` becomes the sample-level index ``x[0]``.
    Batched ``x[0]`` is rejected because it removes the batch axis. Pure
    integer sample indexing becomes ``GetItem``; any sample slice, ellipsis, or
    new axis becomes ``Slice``. Both use the same canonical index format.
    """
    # FX represents x[0], x[:, 1:3], and x[..., -1] as
    # operator.getitem(input_tensor, raw_index). The first argument is the
    # producer Node, not the concrete tensor value used during ShapeProp.
    if len(node.args) != 2 or node.kwargs:
        raise UnsupportedOperatorError(
            f"unsupported index arguments at node '{node.name}'"
        )

    input_node, raw_index = node.args
    # A Node stored in another Node's args is a producer reference: this index
    # operation consumes the tensor produced by input_node.
    if not isinstance(input_node, fx.Node):
        raise UnsupportedOperatorError(
            f"unsupported index input at node '{node.name}'"
        )

    # Use one tuple representation for both x[0] -> (0,) and
    # x[:, 1:3] -> (slice(None), slice(1, 3)). This is still the raw
    # batch-aware PyTorch syntax; no dimension has been removed yet.
    
    
    items = raw_index if isinstance(raw_index, tuple) else (raw_index,)
    input_shape = _node_shape(input_node)

    # Check the original syntax before canonicalization. With batch size 1,
    # both x[:] and x[:1] normalize numerically to slice(0, 1, 1), but only
    # x[:] preserves an arbitrary batch. Keeping this check before
    # _canonical_index avoids losing that semantic distinction.
    if not _index_preserves_batch(items, len(input_shape)):
        raise UnsupportedOperatorError(
            f"unsupported index at node '{node.name}': "
            "batch dimension 0 must remain unchanged"
        )

    # Expand ellipses and omitted suffixes, resolve negative integers, and
    # convert every raw FX axis into an explicit (kind, ...) action.
    index = _canonical_index(node, items, input_shape)

    # This is a defensive invariant after canonicalization: the first action
    # must be the complete slice of the size-one FX batch axis. Integer
    # selection, new-axis insertion, or a missing action cannot be removed
    # safely when translating to per-sample GraphIR.
    if not index or index[0] != ("slice", 0, 1, 1):
        raise UnsupportedOperatorError(
            f"unsupported index at node '{node.name}': "
            "batch dimension 0 must remain unchanged"
        )

    # GraphIR has no batch dimension, so drop exactly the validated first
    # action. For example, batched x[:, 0] becomes per-sample x[0], while all
    # later slice/index actions keep their original sample-axis meaning.
    index = index[1:]

    # Classify from the original syntax because _canonical_index also inserts
    # implicit full slices. After ignoring the required leading batch slice,
    # syntax containing only integers is GetItem; other static syntax is Slice.
    pure_integer = _pure_sample_integer_index(items)
    op_type = "GetItem" if pure_integer else "Slice"
    return op_type, {"index": index}


def _pure_sample_integer_index(items: tuple[Any, ...]) -> bool:
    """Return whether syntax after an explicit full batch slice is integral."""
    if not items or items[0] != slice(None):
        return False
    sample_items = items[1:]
    return bool(sample_items) and all(type(item) is int for item in sample_items)


def _index_preserves_batch(items: tuple[Any, ...], rank: int) -> bool:
    """Return whether indexing leaves the leading batch axis untouched."""
    consumed = sum(item is not None and item is not Ellipsis for item in items)
    missing = rank - consumed

    for item in items:
        if item is None:
            return False
        if item is Ellipsis:
            if missing > 0:
                return True
            continue
        if type(item) is int:
            return False
        if isinstance(item, slice):
            return (
                item.start is None
                and item.stop is None
                and (item.step is None or item.step == 1)
            )
        # Let canonical index validation report unsupported dynamic objects.
        return True

    return True


def _canonical_index(
    node: fx.Node,
    items: tuple[Any, ...],
    input_shape: tuple[int, ...],
) -> tuple[tuple[Any, ...], ...]:
    """Resolve Python indexing syntax against a static input shape.
    
    A slice action stores ``(kind, start, stop, step)``. Integer indexing removes
    one output dimension, while ``("newaxis",)`` introduces one.
    
    For ``x.shape == (1, 4, 5, 6)``, the expression
    ``x[:, 1:3, ::2, -1]`` becomes::

        (("slice", 0, 1, 1),
         ("slice", 1, 3, 1),
         ("slice", 0, 5, 2),
         ("index", 5))
    """
    # Step 1: determine how many full slices an ellipsis (...) or omitted suffix must
    # represent. Python permits at most one ellipsis in a tensor index.
    ellipsis_count = sum(item is Ellipsis for item in items)
    if ellipsis_count > 1:
        raise UnsupportedOperatorError(
            f"unsupported index at node '{node.name}': multiple ellipses"
        )
    rank = len(input_shape)
    # Integer and slice items each consume one input axis. None adds an output
    # axis, and Ellipsis stands for the as-yet-unmentioned input axes.
    # For x[..., -1], consumed == 1 and a rank-4 input has missing == 3.
    consumed = sum(item is not None and item is not Ellipsis for item in items)
    if consumed > rank:
        raise UnsupportedOperatorError(
            f"unsupported index at node '{node.name}': too many dimensions"
        )

    missing = rank - consumed

    # Step 2: make all implicit full slices explicit. For a rank-4 input,
    # x[..., -1] becomes x[:, :, :, -1], while x[0] becomes x[0, :, :, :].
    expanded: list[Any] = []
    for item in items:
        if item is Ellipsis:
            expanded.extend([slice(None)] * missing)
        else:
            expanded.append(item)
    if ellipsis_count == 0:
        # If there is no ellipsis, fill the missing dimensions with full slices.
        expanded.extend([slice(None)] * missing)

    # Step 3: resolve every item into a backend-independent dimension action.
    # ``axis`` tracks consumed input axes; it does not advance for newaxis.
    canonical = [] # a list of tuple, each tuple is the type and the index/slice information
    axis = 0
    for item in expanded:
        if item is None:
            canonical.append(("newaxis",))
            continue

        size = input_shape[axis]
        if type(item) is int:
            # Resolve negative integers now: index -1 on an axis of size 6 is 5.
            value = item + size if item < 0 else item
            if value < 0 or value >= size:
                raise UnsupportedOperatorError(
                    f"unsupported index at node '{node.name}': "
                    f"{item} is out of bounds for dimension {axis}"
                )
            canonical.append(("index", value))
        elif isinstance(item, slice):
            _validate_static_slice(node, item)
            # slice.indices(size) fills None, convert negative indices, 
            # cut the out of bounds indices, add step=1 by default.
            start, stop, step = item.indices(size)
            canonical.append(("slice", start, stop, step))
        else:
            raise UnsupportedOperatorError(
                f"unsupported dynamic index at node '{node.name}': {item!r}"
            )
        # Both integer selection and slicing consume exactly one input axis.
        axis += 1

    return tuple(canonical) # tuple of tuple, each tuple is the type and the index/slice information


def _validate_static_slice(node: fx.Node, item: slice) -> None:
    """Reject data-dependent bounds and slice steps unsupported by PyTorch."""
    for value in (item.start, item.stop, item.step):
        if value is not None and type(value) is not int:
            raise UnsupportedOperatorError(
                f"unsupported dynamic slice at node '{node.name}': {item!r}"
            )
    if item.step is not None and item.step <= 0:
        raise UnsupportedOperatorError(
            f"unsupported slice step at node '{node.name}': {item.step}"
        )


def _canonical_dim(
    node: fx.Node,
    dim: Any,
    rank: int,
    operation: str,
) -> int:
    """Convert one static dimension to its non-negative form.

    For a rank-4 tensor, dimensions ``-1`` and ``-2`` become ``3`` and ``2``.
    """
    if type(dim) is not int:
        raise UnsupportedOperatorError(
            f"unsupported {operation} dimension at node '{node.name}': {dim!r}"
        )

    canonical = dim + rank if dim < 0 else dim
    if canonical < 0 or canonical >= rank:
        raise UnsupportedOperatorError(
            f"unsupported {operation} dimension at node '{node.name}': "
            f"{dim} for rank {rank}"
        )
    return canonical


def _canonical_insert_dim(
    node: fx.Node,
    dim: Any,
    rank: int,
    operation: str,
) -> int:
    """Resolve a static dimension that inserts into a rank-plus-one result."""
    if type(dim) is not int:
        raise UnsupportedOperatorError(
            f"unsupported {operation} dimension at node '{node.name}': {dim!r}"
        )
    
    # plus 1 means new dimension is added
    canonical = dim + rank + 1 if dim < 0 else dim
    if canonical < 0 or canonical > rank:
        raise UnsupportedOperatorError(
            f"unsupported {operation} dimension at node '{node.name}': "
            f"{dim} for rank {rank}"
        )
    return canonical


def _sample_dim(node: fx.Node, dim: int, operation: str) -> int:
    """Convert one raw FX dimension into its per-sample IR dimension."""
    if dim == 0:
        raise UnsupportedOperatorError(
            f"unsupported {operation} at node '{node.name}': "
            "batch dimension 0 must remain unchanged"
        )
    return dim - 1


def _input_rank(node: fx.Node) -> int:
    """Return the rank of the tensor input to a supported unary shape op."""
    if not node.all_input_nodes:
        raise GraphCaptureError(f"node '{node.name}' has no tensor input")
    # Current callers are unary tensor operations such as Flatten and Permute,
    # so all_input_nodes[0] is their sole tensor producer.
    return len(_node_shape(node.all_input_nodes[0]))


def _node_shape(node: fx.Node) -> tuple[int, ...]:
    """Read the static output shape attached to an FX node by ShapeProp."""
    tensor_meta = node.meta.get("tensor_meta")
    shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        raise GraphCaptureError(
            f"node '{node.name}' has no tensor metadata; run propagate_shapes() first"
        )
    return tuple(shape)


def _sample_shape(node: fx.Node) -> tuple[int, ...]:
    """Remove the leading batch-size-one dimension from an FX tensor shape."""
    shape = _node_shape(node)
    if not shape or shape[0] != 1:
        raise UnsupportedOperatorError(
            f"node '{node.name}' does not preserve the leading "
            f"batch dimension of size 1: shape={shape}"
        )
    return shape[1:]


def _linear_attrs(
    module_path: str,
    module: nn.Linear,
    constants: dict[str, np.ndarray],
) -> dict[str, str]:
    """Lift Linear parameters and return their names in ``GraphIR.constants``."""
    return _weight_bias_attrs(
        module_path,
        module,
        module.out_features,
        constants,
    )


def _conv2d_attrs(
    node_name: str,
    module_path: str,
    module: nn.Conv2d,
    constants: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validate version 0.1 Conv2d semantics and lift its fixed parameters."""
    # The first exact Conv2d formulation intentionally covers only a standard
    # dense kernel with unit dilation and zero padding.
    if module.groups != 1:
        raise UnsupportedOperatorError(
            f"unsupported Conv2d groups at node '{node_name}': "
            f"expected 1, got {module.groups}"
        )
    if module.dilation != (1, 1):
        raise UnsupportedOperatorError(
            f"unsupported Conv2d dilation at node '{node_name}': "
            f"expected (1, 1), got {module.dilation}"
        )
    if module.padding_mode != "zeros" or isinstance(module.padding, str):
        raise UnsupportedOperatorError(
            f"unsupported Conv2d padding at node '{node_name}': "
            f"padding={module.padding!r}, mode={module.padding_mode!r}"
        )

    attrs = _weight_bias_attrs(
        module_path,
        module,
        module.out_channels,
        constants,
    )
    attrs.update(
        stride=tuple(module.stride),
        padding=tuple(module.padding),
        dilation=tuple(module.dilation),
        groups=module.groups,
    )
    return attrs


def _batchnorm_attrs(
    node: fx.Node,
    module_path: str,
    module: nn.BatchNorm1d | nn.BatchNorm2d,
    constants: dict[str, np.ndarray],
) -> dict[str, str]:
    """Convert evaluation-mode BatchNorm into fixed scale and shift constants."""
    if module.training:
        raise UnsupportedOperatorError(
            f"BatchNorm node '{node.name}' must be in evaluation mode"
        )
    if (
        not module.track_running_stats
        or module.running_mean is None
        or module.running_var is None
    ):
        raise UnsupportedOperatorError(
            f"BatchNorm node '{node.name}' requires fixed running statistics"
        )

    input_shape = _sample_shape(node.all_input_nodes[0])
    expected_ranks = (1, 2) if isinstance(module, nn.BatchNorm1d) else (3,)
    if (
        len(input_shape) not in expected_ranks
        or input_shape[0] != module.num_features  # num_features is the channel dimension
    ):
        raise UnsupportedOperatorError(
            f"unsupported BatchNorm input shape at node '{node.name}': "
            f"shape={input_shape}, num_features={module.num_features}"
        )

    scale_name = f"{module_path}.scale"
    shift_name = f"{module_path}.shift"
    if scale_name not in constants:
        mean = module.running_mean
        variance = module.running_var
        weight = (
            module.weight
            if module.weight is not None
            else torch.ones_like(mean)
        )
        bias = module.bias if module.bias is not None else torch.zeros_like(mean)
        scale = weight / torch.sqrt(variance + module.eps)
        shift = bias - scale * mean
        constants[scale_name] = _readonly_numpy(scale)
        constants[shift_name] = _readonly_numpy(shift)

    return {"scale": scale_name, "shift": shift_name}


def _weight_bias_attrs(
    module_path: str,
    module: nn.Module,
    output_size: int,
    constants: dict[str, np.ndarray],
) -> dict[str, str]:
    """Return weight/bias constant references, synthesizing zero bias if needed.

    Node attributes store names such as ``"linear.weight"``; the corresponding
    immutable NumPy arrays live once in ``GraphIR.constants``. This indirection
    lets repeated calls to a shared module reuse the same parameter arrays.
    """
    state = _lift_module_state(module_path, module, constants)
    if "bias" not in state:
        # A missing PyTorch bias is mathematically equivalent to a fixed zero
        # vector, so materialize that vector to simplify later formulations.
        bias_name = f"{module_path}.bias"
        if bias_name not in constants:
            constants[bias_name] = _readonly_numpy(
                module.weight.new_zeros(output_size)
            )
        state["bias"] = bias_name

    return {"weight": state["weight"], "bias": state["bias"]}


def _lift_module_state(
    module_path: str,
    module: nn.Module,
    constants: dict[str, np.ndarray],
) -> dict[str, str]:
    """Copy a module's direct parameters and buffers into IR constants.

    ``recurse=False`` is important: each call_module node owns only the state of
    its target module, rather than duplicating parameters of nested children.
    """
    state = dict(module.named_parameters(recurse=False))
    state.update(module.named_buffers(recurse=False))

    references = {}
    for name, tensor in state.items():
        constant_name = f"{module_path}.{name}"
        if constant_name not in constants:
            constants[constant_name] = _readonly_numpy(tensor)
        references[name] = constant_name

    return references


def _readonly_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach fixed PyTorch state into an independent read-only NumPy array."""
    array = tensor.detach().cpu().numpy().copy()
    array.setflags(write=False)
    return array


def _ir_node(
    node: fx.Node,
    op_type: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    attrs: dict[str, Any] | None = None,
) -> IRNode:
    """Build one immutable canonical operation while preserving its FX name."""
    return IRNode(
        name=node.name,
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
        attrs=attrs or {},
    )


def _input_names(node: fx.Node) -> tuple[str, ...]:
    """Return producer-node names referenced anywhere in args or kwargs.

    ``fx.map_arg`` recursively visits Node references inside tuples, lists, and
    dictionaries. This is why ``torch.cat((left, right), dim=1)`` produces the
    two tensor inputs ``("left", "right")`` while ignoring the integer dim.
    """
    names = []
    fx.map_arg(
        (node.args, node.kwargs),
        lambda input_node: names.append(input_node.name),
    )
    return tuple(names)


def _tensor_spec(node: fx.Node, producer: str | None) -> TensorSpec:
    """Create per-sample IR metadata from batched FX ShapeProp output."""
    tensor_meta = node.meta.get("tensor_meta")
    dtype = getattr(tensor_meta, "dtype", None)
    if dtype is None:
        raise GraphCaptureError(
            f"node '{node.name}' has no tensor metadata; run propagate_shapes() first"
        )

    return TensorSpec(
        name=node.name,
        shape=_sample_shape(node),
        dtype=str(dtype).removeprefix("torch."),
        producer=producer,  # Node name
    )


def _unsupported_node(node: fx.Node) -> UnsupportedOperatorError:
    """Build an error containing the FX target, output shape, and source module."""
    tensor_meta = node.meta.get("tensor_meta")
    shape = getattr(tensor_meta, "shape", None)
    module_stack = node.meta.get("nn_module_stack")
    module_path = next(reversed(module_stack.values()))[0] if module_stack else None
    return UnsupportedOperatorError(
        f"unsupported node '{node.name}': op={node.op}, target={node.target}, "
        f"shape={tuple(shape) if shape is not None else None}, "
        f"module={module_path}"
    )
