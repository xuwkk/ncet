import numpy as np
import pytest
import torch
from torch import nn

from tests.fixtures.models import make_identity_residual_cnn
from ncet.errors import InvalidBoundsError
from ncet.frontend import capture_graph, normalize_graph, propagate_shapes
from ncet.ir import GraphIR
from ncet.passes import Bounds, propagate_bounds


class ResidualReLUMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 0.25]]))
            self.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + x


class TensorRouting(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        value = torch.cat((x, y), dim=1)
        value = value.permute(0, 2, 3, 1)
        value = value[:, :, 1:, :]
        value = value.transpose(-1, -2)
        value = value.flatten(start_dim=1)
        value = value.reshape(1, 2, 2, 2)
        return value


def _graph() -> tuple[nn.Module, GraphIR]:
    model = ResidualReLUMLP().eval()
    traced = capture_graph(model)
    propagate_shapes(traced, torch.zeros(1, 2))
    return model, normalize_graph(traced)


def test_residual_relu_bounds_contain_intermediate_values() -> None:
    model, graph = _graph()
    input_bounds = Bounds(
        lower=np.array([-1.0, 0.0], dtype=np.float32),
        upper=np.array([2.0, 3.0], dtype=np.float32),
    )
    bounds = propagate_bounds(graph, {graph.inputs[0]: input_bounds})
    node_names = {
        node.op_type: node.outputs[0]
        for node in graph.nodes
        if node.outputs
    }

    samples = torch.tensor(
        [[-1.0, 0.0], [-1.0, 3.0], [2.0, 0.0], [2.0, 3.0], [0.5, 1.5]]
    )
    with torch.no_grad():
        linear = model.linear(samples)
        relu = torch.relu(linear)
        actual = {
            graph.inputs[0]: samples,
            node_names["Linear"]: linear,
            node_names["ReLU"]: relu,
            node_names["Add"]: relu + samples,
        }

    assert set(bounds) == set(graph.tensors)
    for tensor_name, values in actual.items():
        assert np.all(values.numpy() >= bounds[tensor_name].lower - 1e-6)
        assert np.all(values.numpy() <= bounds[tensor_name].upper + 1e-6)


def test_propagate_bounds_rejects_reversed_input_interval() -> None:
    _, graph = _graph()

    with pytest.raises(InvalidBoundsError, match="lower bound exceeds"):
        propagate_bounds(
            graph,
            {
                graph.inputs[0]: Bounds(
                    lower=np.ones(2),
                    upper=np.zeros(2),
                )
            },
        )


def test_tensor_routing_bounds_follow_the_static_mapping() -> None:
    model = TensorRouting().eval()
    example_inputs = (torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3))
    traced = capture_graph(model)
    propagate_shapes(traced, *example_inputs)
    graph = normalize_graph(traced)

    lower_x = np.arange(6, dtype=np.float32).reshape(1, 2, 3) - 6
    lower_y = np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 10
    upper_x = lower_x + 1
    upper_y = lower_y + 2
    bounds = propagate_bounds(
        graph,
        {
            graph.inputs[0]: Bounds(lower_x, upper_x),
            graph.inputs[1]: Bounds(lower_y, upper_y),
        },
    )

    with torch.no_grad():
        expected_lower = model(
            torch.from_numpy(lower_x).unsqueeze(0),
            torch.from_numpy(lower_y).unsqueeze(0),
        ).squeeze(0)
        expected_upper = model(
            torch.from_numpy(upper_x).unsqueeze(0),
            torch.from_numpy(upper_y).unsqueeze(0),
        ).squeeze(0)

    output_bounds = bounds[graph.outputs[0]]
    np.testing.assert_array_equal(output_bounds.lower, expected_lower.numpy())
    np.testing.assert_array_equal(output_bounds.upper, expected_upper.numpy())


def test_conv_relu_residual_bounds_contain_sampled_outputs() -> None:
    model = make_identity_residual_cnn()
    traced = capture_graph(model)
    propagate_shapes(traced, torch.zeros(1, 1, 4, 4))
    graph = normalize_graph(traced)
    input_bounds = Bounds(
        lower=np.full((1, 4, 4), -1.0, dtype=np.float32),
        upper=np.full((1, 4, 4), 2.0, dtype=np.float32),
    )
    bounds = propagate_bounds(graph, {graph.inputs[0]: input_bounds})

    samples = torch.stack(
        (
            torch.full((1, 4, 4), -1.0),
            torch.zeros(1, 4, 4),
            torch.full((1, 4, 4), 2.0),
            torch.linspace(-1.0, 2.0, 16).reshape(1, 4, 4),
        )
    )
    with torch.no_grad():
        outputs = model(samples).numpy()

    output_bounds = bounds[graph.outputs[0]]
    assert output_bounds.lower.shape == (1, 4, 4)
    assert output_bounds.upper.shape == (1, 4, 4)
    assert np.all(outputs >= output_bounds.lower - 1e-6)
    assert np.all(outputs <= output_bounds.upper + 1e-6)
