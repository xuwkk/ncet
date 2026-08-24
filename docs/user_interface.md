# `form_milp()` user interface

`form_milp()` is NCET's high-level public interface. It captures a supported
PyTorch model, propagates interval bounds, and returns an exact CVXPY
mixed-integer formulation. It creates the neural-network constraints but does
not select an optimization objective or solve the problem.

## Signature

```python
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from torch import nn

from ncet import Bounds, MILPEncoding


BoundPair = tuple[Any, Any]
BoundLike = Bounds | BoundPair
InputBounds = BoundLike | Sequence[Bounds] | Mapping[str, BoundLike]


def form_milp(
    model: nn.Module,
    input_bounds: InputBounds,
    *,
    relu_binary_mode: Literal["full", "reduced"] = "reduced",
) -> MILPEncoding:
    ...
```

The `*` makes `relu_binary_mode` keyword-only.

## Arguments

### `model`

The PyTorch `nn.Module` to encode.

The model must:

- be in evaluation mode, normally by calling `model.eval()`;
- have fixed parameters and buffers;
- be traceable as a static PyTorch FX graph;
- contain only [supported operations](supported_operators.md);
- avoid in-place mutation and data-dependent Python control flow;
- accept a leading batch dimension and compute each sample independently.

Branches, shared tensors, residual/skip connections, multiple inputs, and
multiple outputs are supported when all operations remain inside NCET's
supported boundary.

### `input_bounds`

> **Important — no batch dimension:** `lower` and `upper` describe exactly one
> sample. Use `(features,)` for an MLP or `(channels, height, width)` for an
> image, not `(batch, features)` or `(batch, channels, height, width)`.

Finite elementwise lower and upper bounds for every model input. Bounds use
the shape of one sample and must not contain the batch dimension. For example,
use `(features,)` for an MLP input or `(channels, height, width)` for an image.

Each pair defines a box

$$
\mathrm{lower} \leq x \leq \mathrm{upper}.
$$

The lower and upper arrays must have identical shapes, contain only finite
values, and satisfy `lower <= upper` elementwise.

`Bounds` is the public container for one pair:

```python
Bounds(lower: np.ndarray, upper: np.ndarray)
```

NumPy arrays are recommended when constructing `Bounds` directly. The
`(lower, upper)` tuple form also accepts PyTorch tensors and other array-like
values, which `form_milp()` converts to NumPy arrays.

The accepted forms are:

| Model inputs | Accepted form | Example | Binding rule |
|---|---|---|---|
| One | `Bounds` | `Bounds(lower, upper)` | Bound to the only input |
| One | `(lower, upper)` tuple | `(lower, upper)` | Converted to `Bounds` |
| One or more | Name-to-bound mapping | `{"x": x_bounds}` | Keys match `forward()` argument names |
| Multiple | Non-tuple sequence of `Bounds` | `[x_bounds, y_bounds]` | Order must match `forward()` arguments |

A mapping value may be either `Bounds` or a `(lower, upper)` tuple. Named
bounds are reordered automatically according to the model's FX placeholder
order. A positional list cannot be reordered, so the caller must supply it in
the same order as the arguments of `forward()`. Tuples are reserved for the
single-input `(lower, upper)` form; use a list for multiple positional inputs.

For a model defined as

```python
def forward(self, x, y):
    ...
```

the following two calls are equivalent:

```python
form_milp(model, [x_bounds, y_bounds])
form_milp(model, {"y": y_bounds, "x": x_bounds})
```

`input_bounds` represents elementwise box bounds. A symmetric box
`[x0 - epsilon, x0 + epsilon]` is an $L_\infty$ neighborhood, but NCET does
not directly accept coupled $L_1$, $L_2$, or other norm domains.

### `relu_binary_mode`

Controls the number of ReLU binary variables without changing exactness.

| Value | Behavior |
|---|---|
| `"reduced"` | Default. Introduces binaries only for unstable elements whose bounds satisfy $L < 0 < U$. |
| `"full"` | Introduces one binary for every ReLU element, including elements already known to be active or inactive. |

Both modes describe the same exact ReLU graph. `"reduced"` generally produces
a smaller formulation.

## Return value

`form_milp()` returns a `MILPEncoding` with these public fields:

| Field | Type | Meaning |
|---|---|---|
| `constraints` | `list[cp.Constraint]` | All exact neural-network and input-bound constraints |
| `inputs` | `dict[str, cp.Expression]` | Per-sample input variables keyed by `forward()` argument name |
| `outputs` | `list[cp.Expression]` | Per-sample output expressions in model return order |
| `values` | `dict[str, EncodedTensor]` | Every graph tensor's expression, propagated bounds, and shape |
| `binaries` | `dict[str, ReLUBinaries | MaxPoolBinaries]` | Binary-variable metadata keyed by the corresponding IR node name |
| `graph` | `GraphIR` | Canonical NCET graph used by the encoder |
| `stats` | `EncodingStats` | Counts of variables, constraints, and ReLU states |

Even a single model output is stored in a list, so access it with
`encoding.outputs[0]`. A single model input is still stored in a dictionary,
for example `encoding.inputs["x"]`.

## Using the encoding

```python
import cvxpy as cp

encoding = form_milp(
    model.eval(),
    Bounds(lower=lower, upper=upper),
    relu_binary_mode="reduced",
)

x = encoding.inputs["x"]
y = encoding.outputs[0]

problem = cp.Problem(
    cp.Maximize(y[0]),
    encoding.constraints,
)
problem.solve(solver=cp.SCIPY)
```

External decision variables can be connected to `x`, and additional
constraints can be added alongside `encoding.constraints`. The selected solver
must support mixed-integer problems whenever the encoding contains binaries.

## Errors

| Exception | Typical cause |
|---|---|
| `GraphCaptureError` | Training mode, failed FX tracing, in-place mutation, or dynamic Python control flow |
| `InvalidBoundsError` | Missing/unknown input names, wrong input count or shape, non-finite values, or `lower > upper` |
| `UnsupportedOperatorError` | Unsupported operation or unsupported parameter option |
| `ExactnessContractError` | Captured graph violates NCET's exact graph contract |
| `ValueError` | `relu_binary_mode` is not `"reduced"` or `"full"` |

NCET fails instead of silently approximating unsupported behavior. See the
[exactness contract](exactness_contract.md) for the semantic guarantee and
[current operator support](supported_operators.md) for the capability boundary.
