# NCET

NCET (Neural-network Constraint Embedding Toolkit) converts supported PyTorch
neural networks into
[exact mixed-integer linear constraints](https://xuwkk.github.io/ncet/exactness_contract/).
It preserves graph connectivity, including branches and residual/skip
connections, rather than restricting models to sequential structures.

NCET is partly supported by Engineering and Physical Sciences Research Council grant number [EP/Y025946/1].

**Documentation:** [https://xuwkk.github.io/ncet/](https://xuwkk.github.io/ncet/)

> Declarations: Codex has been used to refine the codebase, generate the documentation and the pytest cases.

## Installation

Install NCET from PyPI:

```bash
pip install ncet
```

For a local editable installation, run from the repository root:

```bash
pip install -e .
```

## Quick start

This example exactly encodes a small residual network and maximizes its first
output over a bounded input box:

```python
import cvxpy as cp
import numpy as np
import torch
from torch import nn

from ncet import Bounds, form_milp

# Define a simple residual MLP
class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[1.0, -1.0], [0.5, 1.0]])
            )
            self.linear.bias.copy_(torch.tensor([0.0, -0.25]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + x


model = ResidualMLP().eval()
# Define the bounds of the input. Bounds describe one sample; do not include a batch dimension.
bounds = Bounds(
    lower=np.array([-1.0, -1.0]),
    upper=np.array([1.0, 1.0]),
)

# Convert the model to a MILP encoding
encoding = form_milp(model, bounds, relu_binary_mode="reduced")

# Connect the encoding to a CVXPY problem
x = encoding.inputs["x"]
y = encoding.outputs[0]
problem = cp.Problem(cp.Maximize(y[0]), encoding.constraints)
problem.solve(solver=cp.SCIPY)

print(problem.status, problem.value)  # optimal 3.0
print(x.value)                        # [ 1. -1.]
```

See the complete [`form_milp()` user interface](https://xuwkk.github.io/ncet/user_interface/)
for all arguments, accepted bound forms, return fields, and public exceptions.

## Modeling scope

NCET captures static PyTorch FX graphs and normalizes the supported operations
into a canonical graph intermediate representation (IR). The current operator
set is Linear, Conv2d, BatchNorm1d/2d, AdaptiveAvgPool2d, AvgPool2d,
MaxPool2d, ReLU, Add, Sub, fixed-constant Add/Sub/Mul/Div, Concat, Flatten,
ReduceMean, Reshape/View/Squeeze/Unsqueeze, Permute, Transpose, Identity/evaluation-mode
Dropout, and static GetItem/Slice. Graph-based interval
bound propagation and the CVXPY/MILP encoder support this operator set. See the
[current operator boundary](https://xuwkk.github.io/ncet/supported_operators/)
for the accepted semantics and restrictions of each operator.

- [Input bounds](https://xuwkk.github.io/ncet/user_interface/#input_bounds) use
  a single sample's shape, such as `(features,)` or
  `(channels, height, width)`, without a batch dimension.
- `lower` and `upper` define elementwise box bounds
  $\mathrm{lower} \leq x \leq \mathrm{upper}$. The symmetric box
  $[x_0-\epsilon, x_0+\epsilon]$ is an $L_\infty$ ball. Coupled $L_1$,
  $L_2$, and other norm bounds are not currently accepted.
- `relu_binary_mode="reduced"` (the default) introduces binaries only for
  unstable ReLU elements. `"full"` introduces one binary per ReLU element.
  Both modes are exact; see
  [ReLU binary handling](https://xuwkk.github.io/ncet/user_interface/#relu_binary_mode).
- MaxPool2d uses the
  [full exact formulation](https://xuwkk.github.io/ncet/knowledge/maxpool2d_exact_encoding/),
  with one binary selector for every valid candidate in each pooling window.

## Comparison with OMLT

[OMLT](https://github.com/cog-imperial/omlt) is a broader Pyomo-based package
for embedding trained machine-learning models in optimization problems. NCET
focuses on direct, graph-preserving encoding of supported PyTorch networks as
exact CVXPY LP/MILP constraints.

| Aspect | NCET | OMLT |
|---|---|---|
| Model input | PyTorch module via FX | Primarily ONNX or Keras model import |
| Optimization interface | CVXPY variables and constraints | Pyomo blocks and formulations |
| Network connectivity | Preserves branches, fan-out, and residual/skip connections, including supported `Add` and `Concat` merges | Stores network graphs, but built-in neural formulations primarily expect one predecessor per layer and do not provide general tensor `Add`/`Concat` merge layers |
| Built-in neural operators | Broader coverage of common PyTorch graph operations, including normalization, average/adaptive pooling, arithmetic, concatenation, reduction, shape, axis, and static indexing operations | Core neural layers include dense, convolution, max pooling, and GNN layers; also provides smooth activation formulations not currently covered by NCET |
| Main scope | Exact LP/MILP encoding of supported affine, piecewise-linear, pooling, reduction, and tensor-shape operations | Neural networks plus gradient-boosted trees, linear trees, and graph neural networks; also includes nonlinear activation formulations |
| ReLU handling | Exact big-M encoding with full or stable-unit-reduced binaries | Multiple formulations, including big-M, complementarity, and partition-based formulations |
| Best fit | PyTorch models with modern graph connectivity used in CVXPY optimization | Broader model/formulation choices in Pyomo workflows |

The packages are therefore complementary: choose NCET when direct PyTorch and
graph-preserving CVXPY encoding are central, and consider OMLT when Pyomo or
its broader formulation ecosystem is the priority.

## Requirements

NCET requires Python 3.10+, CVXPY 1.7.5+, NumPy 1.26+, SciPy 1.13+, and
PyTorch 2.2+. See `pyproject.toml` for the supported upper bounds.

## Examples

| Notebook | Description |
|---|---|
| [Representative operator test](examples/artificial_test_on_operators.ipynb) | Compares PyTorch and NCET outputs for a branched CNN using representative supported operators. |
| [MNIST adversarial attack](examples/mnist_cnn_adversarial_attack.ipynb) | Trains a small CNN and solves targeted and worst-case adversarial attacks with NCET. |
| [Common failures and exceptions](examples/common_failures_and_exceptions.ipynb) | Demonstrates common invalid inputs and models, their public exceptions, and supported fixes. |

## License

NCET is licensed under the [Apache License 2.0](LICENSE).


## Citation

If you use NCET in your research, please cite:

```bibtex
@ARTICLE{xu2026learning,
  author={Xu, Wangkun and Chu, Zhongda and Teng, Fei},
  journal={IEEE Transactions on Power Systems}, 
  title={Learning-Augmented Power System Operations: A Unified Optimization View}, 
  year={2026},
  volume={},
  number={},
  pages={1-21},
  doi={10.1109/TPWRS.2026.3726363}}
```
