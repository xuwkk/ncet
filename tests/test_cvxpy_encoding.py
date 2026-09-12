import cvxpy as cp
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.fixtures.models import (
    make_batchnorm_residual_cnn,
    make_branch_concat_cnn,
)
from ncet import Bounds, form_milp


class ResidualReLUMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))
            self.linear.bias.copy_(torch.tensor([0.0, 2.0, -2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + x


class StridedConv2d(nn.Module):
    """Exercise channels, rectangular kernels, stride, padding, and bias."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            3,
            kernel_size=(2, 3),
            stride=(2, 1),
            padding=(1, 1),
        )
        with torch.no_grad():
            values = torch.linspace(-0.3, 0.4, self.conv.weight.numel())
            self.conv.weight.copy_(values.reshape_as(self.conv.weight))
            self.conv.bias.copy_(torch.tensor([-0.2, 0.1, 0.3]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AvgPoolVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            count_include_pad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        without_padding = self.pool(x)
        with_padding = F.avg_pool2d(
            x,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            count_include_pad=True,
        )
        return without_padding, with_padding


class AdaptiveAvgPoolVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 3))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pool(x), F.adaptive_avg_pool2d(x, (1, 2))


class MaxPoolVariants(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        overlapping = self.pool(x)
        default_stride = F.max_pool2d(x, kernel_size=2)
        return overlapping, default_stride


class ResidualReLUCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [[[[0.2, -0.1, 0.3],
                       [0.4, 0.5, -0.2],
                       [-0.3, 0.1, 0.2]]]]
                )
            )
            self.conv.bias.fill_(-0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x)) + x


class FlattenReshape(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.flatten(start_dim=1)
        return value.reshape(1, 2, 4)


class PermuteTranspose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.permute(0, 2, 3, 1)
        return value.transpose(-1, -2)


class StaticIndexing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, 0], x[:, 1:3, ::2, -1]


@pytest.mark.parametrize(
    ("mode", "expected_binary_count"),
    [("reduced", 1), ("full", 3)],
)
def test_residual_mlp_encoding_matches_pytorch(
    mode: str,
    expected_binary_count: int,
) -> None:
    model = ResidualReLUMLP().eval()
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(3, -1.0, dtype=np.float32),
            upper=np.full(3, 1.0, dtype=np.float32),
        ),
        relu_binary_mode=mode,
    )
    sample = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    constraints = [
        *encoding.constraints,
        encoding.inputs["x"] == sample,
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    assert encoding.inputs["x"].shape == (3,)
    assert encoding.outputs[0].shape == (3,)
    assert encoding.values["x"].shape == (3,)
    assert encoding.values["x"].bounds.lower.shape == (3,)
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-6)
    assert encoding.stats.binary_variables == expected_binary_count
    binary = next(iter(encoding.binaries.values()))
    expected_indices = [0] if mode == "reduced" else [0, 1, 2]
    assert binary.flat_indices.tolist() == expected_indices
    assert binary.original_tensor_shape == (3,)


def test_conv2d_encoding_matches_pytorch() -> None:
    model = StridedConv2d().eval()
    shape = (2, 4, 5)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(-0.8, 0.9, np.prod(shape), dtype=np.float32).reshape(shape)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    assert encoding.outputs[0].shape == (3, 3, 5)
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-6)


def test_avgpool2d_encoding_matches_pytorch() -> None:
    model = AvgPoolVariants().eval()
    shape = (2, 4, 5)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 2.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(
        -0.8,
        1.7,
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0))

    pool_nodes = [
        node for node in encoding.graph.nodes if node.op_type == "AvgPool2d"
    ]
    assert problem.status == cp.OPTIMAL
    assert [node.attrs["count_include_pad"] for node in pool_nodes] == [False, True]
    assert encoding.outputs[0].shape == (2, 5, 3)
    np.testing.assert_allclose(
        encoding.outputs[0].value,
        expected[0].squeeze(0).numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        encoding.outputs[1].value,
        expected[1].squeeze(0).numpy(),
        atol=1e-6,
    )
    assert encoding.stats.binary_variables == 0


def test_adaptive_avgpool2d_encoding_matches_pytorch() -> None:
    model = AdaptiveAvgPoolVariants().eval()
    shape = (2, 5, 7)
    lower = np.linspace(-2.0, -0.5, np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    upper = lower + 3.0
    encoding = form_milp(model, Bounds(lower=lower, upper=upper))
    sample = lower + 0.4 * (upper - lower)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        sample_outputs = model(torch.from_numpy(sample).unsqueeze(0))

    nodes = [
        node
        for node in encoding.graph.nodes
        if node.op_type == "AdaptiveAvgPool2d"
    ]
    assert problem.status == cp.OPTIMAL
    assert [node.attrs["output_size"] for node in nodes] == [(2, 3), (1, 2)]
    for index, node in enumerate(nodes):
        value = encoding.values[node.outputs[0]]
        expected_lower = F.adaptive_avg_pool2d(
            torch.from_numpy(lower).unsqueeze(0),
            node.attrs["output_size"],
        ).squeeze(0)
        expected_upper = F.adaptive_avg_pool2d(
            torch.from_numpy(upper).unsqueeze(0),
            node.attrs["output_size"],
        ).squeeze(0)
        np.testing.assert_allclose(value.bounds.lower, expected_lower.numpy())
        np.testing.assert_allclose(value.bounds.upper, expected_upper.numpy())
        np.testing.assert_allclose(
            encoding.outputs[index].value,
            sample_outputs[index].squeeze(0).numpy(),
            atol=1e-6,
        )
    assert encoding.stats.binary_variables == 0


def test_maxpool2d_full_encoding_matches_pytorch() -> None:
    model = MaxPoolVariants().eval()
    shape = (2, 2, 3)
    lower = np.linspace(-2.0, -0.5, np.prod(shape), dtype=np.float32).reshape(shape)
    upper = np.linspace(0.5, 3.0, np.prod(shape), dtype=np.float32).reshape(shape)
    encoding = form_milp(model, Bounds(lower=lower, upper=upper))
    sample = (lower + upper) / 2
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0))

    pool_nodes = [
        node for node in encoding.graph.nodes if node.op_type == "MaxPool2d"
    ]
    binaries = [encoding.binaries[node.name] for node in pool_nodes]
    assert problem.status == cp.OPTIMAL
    assert [node.attrs["stride"] for node in pool_nodes] == [(1, 1), (2, 2)]
    assert [binary.variable.size for binary in binaries] == [48, 8]
    assert binaries[0].output_indices.shape == (48, 3)
    assert binaries[0].input_indices.shape == (48, 3)
    assert encoding.stats.binary_variables == 56
    np.testing.assert_allclose(
        encoding.outputs[0].value,
        expected[0].squeeze(0).numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        encoding.outputs[1].value,
        expected[1].squeeze(0).numpy(),
        atol=1e-6,
    )


def test_residual_conv_relu_encoding_matches_pytorch() -> None:
    model = ResidualReLUCNN().eval()
    shape = (1, 3, 3)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(-0.9, 0.7, np.prod(shape), dtype=np.float32).reshape(shape)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        sample_tensor = torch.from_numpy(sample).unsqueeze(0)
        expected_conv = model.conv(sample_tensor)
        expected_relu = torch.relu(expected_conv)
        expected = (expected_relu + sample_tensor).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(
        encoding.values["conv"].expression.value,
        expected_conv.squeeze(0).numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        encoding.values["relu"].expression.value,
        expected_relu.squeeze(0).numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-6)


def test_batchnorm_residual_encoding_matches_pytorch() -> None:
    model = make_batchnorm_residual_cnn()
    shape = (1, 3, 3)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(
        -0.8,
        0.7,
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        sample_tensor = torch.from_numpy(sample).unsqueeze(0)
        expected_batch_norm = model.batch_norm(model.conv(sample_tensor))
        expected = model(sample_tensor).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(
        encoding.values["batch_norm"].expression.value,
        expected_batch_norm.squeeze(0).numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-5)
    assert "batch_norm" not in encoding.binaries


def test_branch_concat_encoding_matches_pytorch() -> None:
    model = make_branch_concat_cnn()
    shape = (1, 4, 4)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(-0.8, 0.9, np.prod(shape), dtype=np.float32).reshape(shape)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    assert encoding.outputs[0].shape == (3, 4, 4)
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-6)
    assert encoding.stats.binary_variables == 0


def test_flatten_reshape_encoding_preserves_c_order() -> None:
    model = FlattenReshape().eval()
    shape = (2, 2, 2)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -4.0, dtype=np.float32),
            upper=np.full(shape, 4.0, dtype=np.float32),
        ),
    )
    sample = np.arange(8, dtype=np.float32).reshape(shape) - 3.5
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0).numpy()

    assert problem.status == cp.OPTIMAL
    assert encoding.values["flatten"].expression.shape == (8,)
    assert encoding.outputs[0].shape == (2, 4)
    np.testing.assert_allclose(encoding.outputs[0].value, expected, atol=1e-6)


def test_permute_transpose_encoding_matches_pytorch() -> None:
    model = PermuteTranspose().eval()
    shape = (2, 3, 4)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -20.0, dtype=np.float32),
            upper=np.full(shape, 20.0, dtype=np.float32),
        ),
    )
    sample = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 12
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        sample_tensor = torch.from_numpy(sample).unsqueeze(0)
        expected_permute = sample_tensor.permute(0, 2, 3, 1)
        expected = expected_permute.transpose(-1, -2)

    assert problem.status == cp.OPTIMAL
    assert encoding.values["permute"].expression.shape == (3, 4, 2)
    assert encoding.outputs[0].shape == (3, 2, 4)
    np.testing.assert_array_equal(
        encoding.values["permute"].expression.value,
        expected_permute.squeeze(0).numpy(),
    )
    np.testing.assert_array_equal(
        encoding.outputs[0].value,
        expected.squeeze(0).numpy(),
    )


def test_static_index_encoding_matches_pytorch() -> None:
    model = StaticIndexing().eval()
    shape = (4, 5, 6)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -60.0, dtype=np.float32),
            upper=np.full(shape, 60.0, dtype=np.float32),
        ),
    )
    sample = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 60
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0))

    assert problem.status == cp.OPTIMAL
    np.testing.assert_array_equal(
        encoding.outputs[0].value,
        expected[0].squeeze(0).numpy(),
    )
    np.testing.assert_array_equal(
        encoding.outputs[1].value,
        expected[1].squeeze(0).numpy(),
    )
    assert encoding.stats.binary_variables == 0
