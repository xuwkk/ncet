import cvxpy as cp
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.fixtures.models import (
    IdentityDropout,
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


class SqueezeUnsqueeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.squeeze(2)
        value = torch.unsqueeze(value, 2)
        value = torch.squeeze(value, 2)
        return value.unsqueeze(-1)


class ReduceMeanCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[-0.5]]]]))
            self.conv.bias.copy_(torch.tensor([0.1, 0.2]))
            self.linear.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.25]]))
            self.linear.bias.copy_(torch.tensor([0.0, -0.1]))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.conv(x))
        pooled = torch.mean(features, dim=(-2, -1))
        logits = self.linear(pooled)
        channel_mean = pooled.mean(dim=1, keepdim=True)
        return logits, channel_mean


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


class ElementwiseConstantAffine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([1.0, -2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.add(x, 2.0)
        value = value.sub(1.0)
        value = value * self.scale
        value = torch.divide(value, 2.0)
        return 3.0 - value


class ScaledTensorAddSub(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.add(x, y, alpha=-2), torch.sub(x, y, alpha=-3)


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


def test_identity_dropout_encoding_matches_pytorch() -> None:
    model = IdentityDropout().eval()
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(4, -1.0, dtype=np.float32),
            upper=np.full(4, 1.0, dtype=np.float32),
        ),
    )
    sample = np.array([-0.8, -0.2, 0.4, 0.9], dtype=np.float32)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0)

    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(encoding.outputs[0].value, expected.numpy())
    assert [node.op_type for node in encoding.graph.nodes] == [
        "Input",
        "Identity",
        "Identity",
        "Identity",
        "Output",
    ]
    assert encoding.stats.binary_variables == 0


def test_elementwise_constant_affine_encoding_matches_pytorch() -> None:
    model = ElementwiseConstantAffine().eval()
    lower = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    upper = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    encoding = form_milp(model, Bounds(lower=lower, upper=upper))
    sample = np.array([-0.5, 2.0, 1.0], dtype=np.float32)
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0)

    nodes = [
        node
        for node in encoding.graph.nodes
        if node.op_type == "ElementwiseAffine"
    ]
    effective_scale = -model.scale.numpy() / 2
    effective_shift = 3 - model.scale.numpy() / 2
    expected_lower = np.where(
        effective_scale >= 0,
        effective_scale * lower,
        effective_scale * upper,
    ) + effective_shift
    expected_upper = np.where(
        effective_scale >= 0,
        effective_scale * upper,
        effective_scale * lower,
    ) + effective_shift
    output_bounds = encoding.values[encoding.graph.outputs[0]].bounds

    assert problem.status == cp.OPTIMAL
    assert len(nodes) == 5
    assert all(len(node.inputs) == 1 for node in nodes)
    assert all(
        name in encoding.graph.constants
        for node in nodes
        for name in node.attrs.values()
    )
    np.testing.assert_allclose(encoding.outputs[0].value, expected.numpy())
    np.testing.assert_allclose(output_bounds.lower, expected_lower)
    np.testing.assert_allclose(output_bounds.upper, expected_upper)
    assert encoding.stats.binary_variables == 0


def test_scaled_tensor_add_sub_encoding_matches_pytorch() -> None:
    model = ScaledTensorAddSub().eval()
    x_bounds = Bounds(
        lower=np.array([-1.0, 0.0]),
        upper=np.array([2.0, 4.0]),
    )
    y_bounds = Bounds(
        lower=np.array([-2.0, 1.0]),
        upper=np.array([3.0, 2.0]),
    )
    encoding = form_milp(model, {"x": x_bounds, "y": y_bounds})
    x = np.array([0.5, 1.5])
    y = np.array([-1.0, 1.25])
    problem = cp.Problem(
        cp.Minimize(0),
        [
            *encoding.constraints,
            encoding.inputs["x"] == x,
            encoding.inputs["y"] == y,
        ],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(x), torch.from_numpy(y))

    add_bounds = encoding.values[encoding.graph.outputs[0]].bounds
    sub_bounds = encoding.values[encoding.graph.outputs[1]].bounds
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(encoding.outputs[0].value, expected[0].numpy())
    np.testing.assert_allclose(encoding.outputs[1].value, expected[1].numpy())
    np.testing.assert_allclose(
        add_bounds.lower,
        x_bounds.lower - 2 * y_bounds.upper,
    )
    np.testing.assert_allclose(
        add_bounds.upper,
        x_bounds.upper - 2 * y_bounds.lower,
    )
    np.testing.assert_allclose(
        sub_bounds.lower,
        x_bounds.lower + 3 * y_bounds.lower,
    )
    np.testing.assert_allclose(
        sub_bounds.upper,
        x_bounds.upper + 3 * y_bounds.upper,
    )
    assert encoding.stats.binary_variables == 0


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


def test_squeeze_unsqueeze_encoding_preserves_values() -> None:
    model = SqueezeUnsqueeze().eval()
    shape = (2, 1, 3)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(-0.8, 0.7, np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0)).squeeze(0)

    assert problem.status == cp.OPTIMAL
    assert [node.op_type for node in encoding.graph.nodes] == [
        "Input",
        "Reshape",
        "Reshape",
        "Reshape",
        "Reshape",
        "Output",
    ]
    assert encoding.outputs[0].shape == (2, 3, 1)
    np.testing.assert_allclose(encoding.outputs[0].value, expected.numpy())


def test_reduce_mean_encoding_matches_pytorch() -> None:
    model = ReduceMeanCNN().eval()
    shape = (1, 2, 3)
    encoding = form_milp(
        model,
        Bounds(
            lower=np.full(shape, -1.0, dtype=np.float32),
            upper=np.full(shape, 1.0, dtype=np.float32),
        ),
    )
    sample = np.linspace(-0.8, 0.7, np.prod(shape), dtype=np.float32).reshape(
        shape
    )
    problem = cp.Problem(
        cp.Minimize(0),
        [*encoding.constraints, encoding.inputs["x"] == sample],
    )
    problem.solve(solver=cp.SCIPY)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample).unsqueeze(0))

    reduce_nodes = [
        node for node in encoding.graph.nodes if node.op_type == "ReduceMean"
    ]
    assert problem.status == cp.OPTIMAL
    assert [node.attrs for node in reduce_nodes] == [
        {"dims": (1, 2), "keepdim": False},
        {"dims": (0,), "keepdim": True},
    ]
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
    assert encoding.stats.binary_variables == 12


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
