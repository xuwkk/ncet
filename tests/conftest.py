import pytest
from torch import nn

from tests.fixtures.models import (
    ResidualMLP,
    make_residual_mlp,
    make_sequential_cnn,
    make_sequential_mlp,
)


@pytest.fixture
def sequential_mlp() -> nn.Sequential:
    return make_sequential_mlp()


@pytest.fixture
def sequential_cnn() -> nn.Sequential:
    return make_sequential_cnn()


@pytest.fixture
def residual_mlp() -> ResidualMLP:
    return make_residual_mlp()
