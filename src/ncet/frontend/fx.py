"""PyTorch FX graph capture and shape propagation."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from ..errors import GraphCaptureError, UnsupportedOperatorError


@dataclass(frozen=True)
class FXNodeInfo:
    """Readable information extracted from one FX node."""

    name: str              # Unique FX node name, also the name of the output tensor
    op: str                # FX operation type, e.g. "call_module", "call_function", "call_method", "placeholder", "output", "get_attr"
    target: Any            # What this FX node actually calls. e.g. block.linear or operator.add
    args: tuple[Any, ...]  # Positional augments for the producer nodes and/or static arguments e.g. (x, y) for add(x, y)
    kwargs: Mapping[str, Any]   # Keyword arguments, e.g. {"inplace": True} for add_(x, y)
    users: tuple[str, ...] # Names of the consumer nodes
    output_shape: tuple[int, ...] | None
    output_dtype: torch.dtype | None
    # Derived from optional FX metadata; this is not an fx.Node attribute.
    module_path: str | None  # Where this operation originated in the original model. e.g. block.linear or block


def capture_graph(model: nn.Module) -> fx.GraphModule:
    """Capture an evaluation-mode model as an executable FX graph."""
    if model.training:
        raise GraphCaptureError("model must be in evaluation mode")
    _validate_batchnorm_modules(model)
    _validate_dropout_modules(model)

    try:
        graph_module = fx.symbolic_trace(model)
    except Exception as error:
        raise GraphCaptureError(
            f"failed to capture {type(model).__name__}: {error}"
        ) from error

    validate_graph(graph_module) # reject graph operations that mutate tensor values in place
    return graph_module


def _validate_batchnorm_modules(model: nn.Module) -> None:
    """Require fixed running statistics for exact BatchNorm encoding."""
    supported = (nn.BatchNorm1d, nn.BatchNorm2d)
    for path, module in model.named_modules():
        if not isinstance(module, supported):
            continue
        name = path or type(module).__name__
        if module.training:
            raise UnsupportedOperatorError(
                f"BatchNorm module '{name}' must be in evaluation mode"
            )
        if (
            not module.track_running_stats
            or module.running_mean is None
            or module.running_var is None
        ):
            raise UnsupportedOperatorError(
                f"BatchNorm module '{name}' requires fixed running statistics"
            )


def _validate_dropout_modules(model: nn.Module) -> None:
    """Reject stochastic Dropout while allowing its evaluation identity."""
    supported = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
    for path, module in model.named_modules():
        if isinstance(module, supported) and module.training:
            name = path or type(module).__name__
            raise UnsupportedOperatorError(
                f"Dropout module '{name}' must be in evaluation mode"
            )


def validate_graph(graph_module: fx.GraphModule) -> None:
    """Reject graph operations that mutate tensor values in place."""
    for node in graph_module.graph.nodes:
        if _is_mutating_node(graph_module, node):
            raise GraphCaptureError(
                f"mutation is not supported at node '{node.name}' "
                f"({node.op}: {node.target})"
            )


def propagate_shapes(
    graph_module: fx.GraphModule,
    *example_inputs: torch.Tensor,
) -> fx.GraphModule:
    """The graph module does not have shape and dtype information.
    Attach tensor metadata to the fx.node.meta including shape and dtype 
    using imaginary inputs."""
    try:
        with torch.no_grad():
            ShapeProp(graph_module).propagate(*example_inputs)
    except Exception as error:
        raise GraphCaptureError(
            f"failed to propagate graph shapes: {error}"
        ) from error

    return graph_module  # the shape and dtype will be added to the fx.node.meta


def describe_graph(graph_module: fx.GraphModule) -> tuple[FXNodeInfo, ...]:
    """Return node information in graph execution order."""
    return tuple(_describe_node(node) for node in graph_module.graph.nodes)


def format_graph(graph_module: fx.GraphModule) -> str:
    """Format graph nodes as clean, one-line records."""
    lines = []
    for node in describe_graph(graph_module):
        # Each FXNodeInfo object
        dtype = str(node.output_dtype).removeprefix("torch.")
        parts = [
            node.name,
            node.op,
            f"target={_format_target(node.target)}",
            f"args={node.args}",
        ]
        if node.kwargs:
            parts.append(f"kwargs={dict(node.kwargs)}")
        parts.extend(
            [
                f"users=[{', '.join(node.users)}]",
                f"shape={node.output_shape}",
                f"dtype={dtype}",
                f"module={node.module_path or '-'}",
            ]
        )
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _describe_node(node: fx.Node) -> FXNodeInfo:
    tensor_meta = node.meta.get("tensor_meta")
    shape = getattr(tensor_meta, "shape", None)
    return FXNodeInfo(
        name=node.name,
        op=node.op,
        target=node.target,
        args=node.args,
        kwargs=node.kwargs,
        users=tuple(user.name for user in node.users), # node name, can be more than one
        output_shape=(tuple(shape) if shape is not None else None),
        output_dtype=getattr(tensor_meta, "dtype", None),
        module_path=_module_path(node),  # locates the source module of the operation
    )


def _format_target(target: Any) -> str:
    if isinstance(target, str):
        return target

    module = getattr(target, "__module__", "")
    if module == "_operator":
        module = "operator"
    name = getattr(target, "__name__", str(target))
    return f"{module}.{name}" if module else name


def _is_mutating_node(graph_module: fx.GraphModule, node: fx.Node) -> bool:
    if node.op == "call_module":
        # self.relu = nn.ReLU(inplace=True)
        module = graph_module.get_submodule(str(node.target))
        return bool(getattr(module, "inplace", False))

    if node.op not in {"call_function", "call_method"}:
        return False

    if node.kwargs.get("inplace", False) is True:
        # F.relu(x, inplace=True)
        return True
    
    # reject relu_() or add_() such as x.add_(1)
    target_name = getattr(node.target, "__name__", str(node.target))
    return target_name.endswith("_") and not target_name.endswith("__")


def _module_path(node: fx.Node) -> str | None:
    """Return the innermost source module recorded during FX tracing.

    This provenance comes from ``node.meta["nn_module_stack"]``. For a
    ``call_module`` node it usually matches ``node.target``; for a functional
    node it identifies the enclosing module and may be absent.
    """
    module_stack = node.meta.get("nn_module_stack")
    if not module_stack:
        return None

    path, _ = next(reversed(module_stack.values())) # get the innermost source module
    return path
