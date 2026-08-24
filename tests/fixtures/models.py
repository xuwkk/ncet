"""Small deterministic PyTorch models used by regression tests."""

from typing import TypeVar

import torch
from torch import nn

MLP_INPUT_SHAPE = (1, 4)
CNN_INPUT_SHAPE = (1, 1, 4, 4)
ModelT = TypeVar("ModelT", bound=nn.Module)


class ResidualMLP(nn.Module):
    """A single affine branch with an identity shortcut."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + x


class IdentityResidualCNN(nn.Module):
    """A convolutional branch with an identity shortcut."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x))) + x


class ProjectionResidualCNN(nn.Module):
    """A convolutional branch with a channel projection shortcut."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv2(self.relu(self.conv1(x)))
        return main + self.shortcut(x)


class BranchConcatCNN(nn.Module):
    """Two convolutional branches concatenated along the channel axis."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Conv2d(1, 1, kernel_size=1)
        self.right = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.left(x), self.right(x)), dim=1)


class SharedLinear(nn.Module):
    """Call the same Linear module at two distinct graph nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # two nodes but same target
        return self.linear(x) + self.linear(x)


class MultipleInputOutput(nn.Module):
    """Return two values computed from two graph inputs."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + y, x - y


def make_sequential_mlp() -> nn.Sequential:
    """Return a small Linear-ReLU-Linear network in evaluation mode."""
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    )
    return _initialize(model)


def make_sequential_cnn() -> nn.Sequential:
    """Return a small Conv-ReLU-Pool-Linear network in evaluation mode."""
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    return _initialize(model)


def make_residual_mlp() -> ResidualMLP:
    """Return a small residual network in evaluation mode."""
    return _initialize(ResidualMLP())


def make_identity_residual_cnn() -> IdentityResidualCNN:
    return _initialize(IdentityResidualCNN())


def make_projection_residual_cnn() -> ProjectionResidualCNN:
    return _initialize(ProjectionResidualCNN())


def make_branch_concat_cnn() -> BranchConcatCNN:
    return _initialize(BranchConcatCNN())


def make_shared_linear() -> SharedLinear:
    return _initialize(SharedLinear())


def make_multiple_input_output() -> MultipleInputOutput:
    return MultipleInputOutput().eval()


def _initialize(model: ModelT) -> ModelT:
    """Set deterministic parameters without changing the global random state."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(-0.2, 0.2, parameter.numel())
            parameter.copy_(values.reshape_as(parameter) + 0.01 * index)

    return model.eval()
