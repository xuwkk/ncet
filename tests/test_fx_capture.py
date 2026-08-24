import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.fixtures.models import ResidualMLP
from ncet.errors import GraphCaptureError
from ncet.frontend import (
    capture_graph,
    describe_graph,
    propagate_shapes,
)


class InplaceReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


class FunctionalInplaceReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=True)


class DynamicBranch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x
        return -x


def test_capture_preserves_residual_connection(residual_mlp: ResidualMLP) -> None:
    traced = capture_graph(residual_mlp)
    x = torch.tensor([[0.2, -0.4, 0.7, 0.1]])

    with torch.no_grad():
        torch.testing.assert_close(traced(x), residual_mlp(x))

    add_nodes = [
        node
        for node in traced.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    ]
    assert len(add_nodes) == 1

    main, shortcut = add_nodes[0].args
    assert main.op == "call_module" and main.target == "linear"
    assert shortcut.op == "placeholder"


def test_shape_propagation_records_residual_tensor_metadata(
    residual_mlp: ResidualMLP,
) -> None:
    traced = capture_graph(residual_mlp)

    propagate_shapes(traced, torch.zeros(2, 4))

    metadata = {
        node.name: (
            tuple(node.meta["tensor_meta"].shape),
            node.meta["tensor_meta"].dtype,
        )
        for node in traced.graph.nodes
    }
    assert metadata == {
        "x": ((2, 4), torch.float32),
        "linear": ((2, 4), torch.float32),
        "add": ((2, 4), torch.float32),
        "output": ((2, 4), torch.float32),
    }


def test_describe_graph_records_residual_dataflow(residual_mlp: ResidualMLP) -> None:
    traced = capture_graph(residual_mlp)
    propagate_shapes(traced, torch.zeros(2, 4))

    nodes = {node.name: node for node in describe_graph(traced)}

    assert nodes["linear"].module_path == "linear"
    assert nodes["add"].op == "call_function"
    assert nodes["add"].target is operator.add
    assert tuple(node.name for node in nodes["add"].args) == ("linear", "x")
    assert nodes["add"].users == ("output",)
    assert nodes["add"].output_shape == (2, 4)
    assert nodes["add"].output_dtype == torch.float32


@pytest.mark.parametrize("model", [InplaceReLU(), FunctionalInplaceReLU()])
def test_capture_rejects_inplace_mutation(model: nn.Module) -> None:
    with pytest.raises(GraphCaptureError, match="mutation is not supported"):
        capture_graph(model.eval())


def test_capture_rejects_tensor_dependent_control_flow() -> None:
    with pytest.raises(GraphCaptureError, match="failed to capture DynamicBranch"):
        capture_graph(DynamicBranch().eval())
