"""Validation for canonical graph IR."""

from ..errors import ExactnessContractError
from .graph import GraphIR, IRNode


def validate_ir(graph: GraphIR) -> None:
    """Validate invariants that could make encoding incorrect or ambiguous."""
    available: set[str] = set()
    node_names: set[str] = set()

    for node in graph.nodes:
        if node.name in node_names:
            _fail(f"duplicate node name '{node.name}'")
        node_names.add(node.name)

        for tensor_name in node.inputs:
            # Input node has no inputs.
            # Input tensor should be available.
            if tensor_name not in available:
                _fail(
                    f"node '{node.name}' references unavailable input "
                    f"'{tensor_name}'"
                )

        for tensor_name in node.outputs:
            # Output node should not have multiple producers.
            if tensor_name in available:
                _fail(f"tensor '{tensor_name}' has multiple producers")
            
            # Each node output becomes a new tensor
            tensor = graph.tensors.get(tensor_name)
            if tensor is None:
                _fail(f"node '{node.name}' produces unknown tensor '{tensor_name}'")
            
            # TensorSpec.name == producer IRNode.outputs[0] == producer IRNode.name
            expected = None if node.op_type == "Input" else node.name
            if tensor.producer != expected:
                _fail(
                    f"tensor '{tensor_name}' has producer {tensor.producer!r}; "
                    f"expected {expected!r}"
                )

            static_shape = all(
                type(dimension) is int and dimension >= 0
                for dimension in tensor.shape
            )
            if not static_shape:
                _fail(f"tensor '{tensor_name}' has non-static shape {tensor.shape}")

            available.add(tensor_name)
        
        # IRNode attr contains the constants name which should be stored in the graph.constants
        _validate_constants(graph, node)

    for tensor_name in [*graph.inputs, *graph.outputs]:
        if tensor_name not in available:
            _fail(f"graph boundary references unavailable tensor '{tensor_name}'")


def _validate_constants(graph: GraphIR, node: IRNode) -> None:
    constant_attributes = {
        "Linear": ("weight", "bias"),
        "Conv2d": ("weight", "bias"),
        "BatchNorm": ("scale", "shift"),
        "ElementwiseAffine": ("scale", "shift"),
    }
    attributes = constant_attributes.get(node.op_type)
    if attributes is None:
        return

    for attribute in attributes:
        constant_name = node.attrs.get(attribute)
        # The node constants are stored separately from the graph.tensors
        if constant_name not in graph.constants:
            _fail(
                f"{node.op_type} node '{node.name}' references missing {attribute} "
                f"constant {constant_name!r}"
            )


def _fail(message: str) -> None:
    raise ExactnessContractError(f"invalid IR: {message}")
