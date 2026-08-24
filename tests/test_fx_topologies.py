import operator

import torch

from tests.fixtures.models import (
    CNN_INPUT_SHAPE,
    make_projection_residual_cnn,
)
from ncet.frontend import capture_graph, propagate_shapes


def test_capture_preserves_projection_residual_shortcut() -> None:
    model = make_projection_residual_cnn()
    traced = capture_graph(model)
    propagate_shapes(traced, torch.zeros(CNN_INPUT_SHAPE))
    add = next(node for node in traced.graph.nodes if node.target is operator.add)
    main, shortcut = add.args

    assert main.target == "conv2"
    assert shortcut.op == "call_module"
    assert shortcut.target == "shortcut"
