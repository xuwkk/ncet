from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.fixtures.models import (
    CNN_INPUT_SHAPE,
    IdentityDropout,
    make_batchnorm_residual_cnn,
    make_branch_concat_cnn,
    make_multiple_input_output,
    make_residual_mlp,
    make_sequential_mlp,
    make_shared_linear,
)
from ncet.errors import ExactnessContractError, UnsupportedOperatorError
from ncet.frontend import capture_graph, normalize_graph, propagate_shapes
from ncet.ir import GraphIR, validate_ir


class FunctionalOps(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.relu(x + y)


class TorchOps(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.add(x, y))


class MethodOps(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.add(y).relu()


class NoBiasLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NoRunningStatsBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(3, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


class ScaledAdd(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y, alpha=2)


class GroupedConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ShapeOps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = x.view(1, 2, 2, 2)
        x = x.permute(0, 2, 3, 1)
        return x.transpose(-1, -2)


class StaticIndexing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0], x[:, 1:3, ::2, -1]


class BatchIndexing(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[0]


class BatchSqueeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)


class ImplicitSqueeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class BatchUnsqueeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)


class BatchMean(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)


class ImplicitMean(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class DynamicIndexing(nn.Module):
    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return x[index]


def _normalize(model: nn.Module, *inputs: torch.Tensor) -> GraphIR:
    traced = capture_graph(model.eval())
    propagate_shapes(traced, *inputs)
    return normalize_graph(traced)


def test_normalize_sequential_mlp() -> None:
    graph = _normalize(make_sequential_mlp(), torch.zeros(1, 4))

    assert [node.op_type for node in graph.nodes] == [
        "Input",
        "Linear",
        "ReLU",
        "Linear",
        "Output",
    ]
    assert graph.inputs == ["input_1"]
    assert graph.outputs == ["_2"]
    assert graph.tensors[graph.inputs[0]].shape == (4,)
    assert graph.tensors["_1"].shape == (3,)


def test_normalize_preserves_residual_inputs() -> None:
    graph = _normalize(make_residual_mlp(), torch.zeros(1, 4))
    add = next(node for node in graph.nodes if node.op_type == "Add")

    assert add.inputs == ("linear", "x")
    assert graph.tensors["add"].producer == "add"


def test_normalize_lifts_linear_parameters() -> None:
    model = make_residual_mlp()
    graph = _normalize(model, torch.zeros(1, 4))
    linear = next(node for node in graph.nodes if node.op_type == "Linear")

    assert linear.attrs == {
        "weight": "linear.weight",
        "bias": "linear.bias",
    }
    np.testing.assert_array_equal(
        graph.constants["linear.weight"],
        model.linear.weight.detach().numpy(),
    )
    assert not graph.constants["linear.weight"].flags.writeable
    assert not graph.constants["linear.bias"].flags.writeable


def test_normalize_reuses_shared_module_constants() -> None:
    graph = _normalize(make_shared_linear(), torch.zeros(1, 4))
    linear_nodes = [node for node in graph.nodes if node.op_type == "Linear"]

    assert linear_nodes[0].attrs == linear_nodes[1].attrs
    assert set(graph.constants) == {"linear.weight", "linear.bias"}


def test_normalize_creates_zero_bias() -> None:
    graph = _normalize(NoBiasLinear(), torch.zeros(1, 4))
    linear = next(node for node in graph.nodes if node.op_type == "Linear")
    bias = graph.constants[linear.attrs["bias"]]

    np.testing.assert_array_equal(bias, np.zeros(3, dtype=np.float32))
    assert bias.dtype == graph.constants[linear.attrs["weight"]].dtype
    assert not bias.flags.writeable


def test_normalize_batchnorm_to_scale_and_shift() -> None:
    model = make_batchnorm_residual_cnn()
    graph = _normalize(model, torch.zeros(1, 1, 3, 3))
    node = next(node for node in graph.nodes if node.op_type == "BatchNorm")

    scale = graph.constants[node.attrs["scale"]]
    shift = graph.constants[node.attrs["shift"]]
    expected_scale_tensor = (
        model.batch_norm.weight
        / torch.sqrt(model.batch_norm.running_var + model.batch_norm.eps)
    )
    expected_shift = (
        model.batch_norm.bias
        - expected_scale_tensor * model.batch_norm.running_mean
    ).detach().numpy()
    expected_scale = expected_scale_tensor.detach().numpy()

    assert node.attrs == {
        "scale": "batch_norm.scale",
        "shift": "batch_norm.shift",
    }
    np.testing.assert_allclose(scale, expected_scale)
    np.testing.assert_allclose(shift, expected_shift)
    assert not scale.flags.writeable
    assert not shift.flags.writeable


def test_capture_rejects_batchnorm_without_running_statistics() -> None:
    with pytest.raises(UnsupportedOperatorError, match="fixed running statistics"):
        capture_graph(NoRunningStatsBatchNorm().eval())


def test_capture_rejects_training_dropout_submodule() -> None:
    model = IdentityDropout().eval()
    model.dropout.train()
    with pytest.raises(UnsupportedOperatorError, match="evaluation mode"):
        capture_graph(model)


def test_normalize_conv2d_and_concat_branches() -> None:
    model = make_branch_concat_cnn()
    graph = _normalize(model, torch.zeros(CNN_INPUT_SHAPE))

    assert [node.op_type for node in graph.nodes] == [
        "Input",
        "Conv2d",
        "Conv2d",
        "Concat",
        "Output",
    ]

    left = next(node for node in graph.nodes if node.name == "left")
    assert left.attrs == {
        "weight": "left.weight",
        "bias": "left.bias",
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
    }
    np.testing.assert_array_equal(
        graph.constants["left.weight"],
        model.left.weight.detach().numpy(),
    )

    concat = next(node for node in graph.nodes if node.op_type == "Concat")
    assert concat.inputs == ("left", "right")
    assert concat.attrs == {"dim": 0}
    assert graph.tensors[concat.outputs[0]].shape == (3, 4, 4)


def test_normalize_sub_with_multiple_inputs_and_outputs() -> None:
    graph = _normalize(
        make_multiple_input_output(),
        torch.zeros(1, 4),
        torch.ones(1, 4),
    )

    assert [node.op_type for node in graph.nodes] == [
        "Input",
        "Input",
        "Add",
        "Sub",
        "Output",
    ]
    assert graph.inputs == ["x", "y"]
    assert graph.outputs == ["add", "sub"]
    sub = next(node for node in graph.nodes if node.op_type == "Sub")
    assert sub.inputs == ("x", "y")
    assert sub.attrs == {"alpha": 1}


def test_normalize_shape_operations() -> None:
    graph = _normalize(ShapeOps(), torch.zeros(1, 2, 2, 2))
    operations = {node.op_type: node for node in graph.nodes}

    assert operations["Flatten"].attrs == {"start_dim": 0, "end_dim": 2}
    assert operations["Reshape"].attrs == {"shape": (2, 2, 2)}
    assert operations["Permute"].attrs == {"dims": (1, 2, 0)}
    assert operations["Transpose"].attrs == {"dim0": 2, "dim1": 1}


def test_normalize_static_indexing() -> None:
    graph = _normalize(StaticIndexing(), torch.zeros(1, 4, 5, 6))
    getitem, sliced = graph.nodes[1:3]

    assert getitem.op_type == "GetItem"
    assert getitem.attrs["index"] == (
        ("index", 0),
        ("slice", 0, 5, 1),
        ("slice", 0, 6, 1),
    )
    assert sliced.op_type == "Slice"
    assert sliced.attrs["index"] == (
        ("slice", 1, 3, 1),
        ("slice", 0, 5, 2),
        ("index", 5),
    )
    assert graph.tensors[sliced.outputs[0]].shape == (2, 3)


def test_normalize_rejects_batch_indexing() -> None:
    with pytest.raises(UnsupportedOperatorError, match="batch dimension 0"):
        _normalize(BatchIndexing(), torch.zeros(1, 4, 5, 6))


def test_normalize_rejects_shape_ops_that_change_batch_axis() -> None:
    models = (
        BatchSqueeze(),
        ImplicitSqueeze(),
        BatchUnsqueeze(),
        BatchMean(),
        ImplicitMean(),
    )
    for model in models:
        with pytest.raises(UnsupportedOperatorError, match="batch dimension"):
            _normalize(model, torch.zeros(1, 2, 1, 3))


def test_normalize_rejects_dynamic_index() -> None:
    with pytest.raises(UnsupportedOperatorError, match="dynamic index"):
        _normalize(
            DynamicIndexing(),
            torch.zeros(1, 3),
            torch.tensor([0]),
        )


def test_normalize_rejects_grouped_conv2d() -> None:
    with pytest.raises(UnsupportedOperatorError, match="Conv2d groups"):
        _normalize(GroupedConv(), torch.zeros(1, 2, 4, 4))


@pytest.mark.parametrize("model", [FunctionalOps(), TorchOps(), MethodOps()])
def test_normalize_equivalent_operator_spellings(model: nn.Module) -> None:
    inputs = (torch.zeros(1, 4), torch.ones(1, 4))
    graph = _normalize(model, *inputs)

    assert [node.op_type for node in graph.nodes] == [
        "Input",
        "Input",
        "Add",
        "ReLU",
        "Output",
    ]
    add = next(node for node in graph.nodes if node.op_type == "Add")
    assert add.attrs == {"alpha": 1}


def test_normalize_rejects_unsupported_add_alpha() -> None:
    inputs = (torch.zeros(1, 4), torch.ones(1, 4))
    with pytest.raises(UnsupportedOperatorError, match="Add alpha"):
        _normalize(ScaledAdd(), *inputs)


def test_validate_ir_rejects_unknown_output() -> None:
    graph = _normalize(make_residual_mlp(), torch.zeros(1, 4))
    graph.outputs = ["missing"]

    with pytest.raises(ExactnessContractError, match="graph boundary"):
        validate_ir(graph)


def test_validate_ir_rejects_non_static_shape() -> None:
    graph = _normalize(make_residual_mlp(), torch.zeros(1, 4))
    graph.tensors["x"] = replace(graph.tensors["x"], shape=(-1, 4))

    with pytest.raises(ExactnessContractError, match="non-static shape"):
        validate_ir(graph)
