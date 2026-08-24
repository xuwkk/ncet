# NCET

NCET (Neural-network Constraint Embedding Toolkit) converts supported PyTorch
neural networks into exact mixed-integer linear constraints. Its graph-aware
frontend preserves branches, shared tensors, and residual/skip connections
instead of restricting models to sequential structures.

## Installation

Install the published package from PyPI:

```bash
pip install ncet
```

For a local editable installation, run from the repository root:

```bash
pip install -e .
```

## Minimal use

```python
import cvxpy as cp
import numpy as np

from ncet import Bounds, form_milp


model.eval()
bounds = Bounds(
    lower=np.array([-1.0, -1.0]),
    upper=np.array([1.0, 1.0]),
)
encoding = form_milp(model, bounds, relu_binary_mode="reduced")

y = encoding.outputs[0]
problem = cp.Problem(cp.Maximize(y[0]), encoding.constraints)
problem.solve(solver=cp.SCIPY)
```

!!! important "Input bounds do not include a batch dimension"

    Bounds describe exactly one sample. Use `(features,)` for an MLP or
    `(channels, height, width)` for an image, not `(batch, features)` or
    `(batch, channels, height, width)`.

See the [`form_milp()` user interface](user_interface.md) for every argument,
accepted bound form, return field, and public exception.

## Documentation map

For regular users, please refer to [User interface](user_interface.md), [Examples](examples.md), and [Supported operators](supported_operators.md) for detailed usage and examples.

For developers, please refer to [Developer reference](developer_note.md) for detailed implementation details.

Mathematical background is available in [Knowledge](knowledge/index.md).

| Section | Purpose |
|---|---|
| [User interface](user_interface.md) | Build and consume an exact NCET encoding |
| [Examples](examples.md) | Representative notebooks and applications |
| [Supported operators](supported_operators.md) | Authoritative current capability boundary |
| [Exactness contract](exactness_contract.md) | Model assumptions and semantic guarantee |
| [Mathematical background](knowledge/index.md) | Bound propagation and exact formulations |
| [Developer reference](developer_note.md) | FX, GraphIR, propagation, and backend internals |
