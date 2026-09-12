# Mathematical knowledge

This folder collects the mathematical knowledge needed to understand NCET's
bound propagation and exact optimization encoding. The notes describe the
formulations and their connection to the current implementation; broader
frontend and GraphIR design details are documented elsewhere under `docs/`.

## Recommended reading order

1. [Interval Bound Propagation](bound_propagation.md) explains how NCET obtains
   finite tensor bounds throughout a graph.
2. [Exact Encoding](exact_encoding.md) explains how a bounded GraphIR becomes
   an LP- or MILP-compatible CVXPY constraint system.
3. The operator notes derive the sparse matrices or exact formulations used by
   the backend:
   - [Conv2d](conv2d_exact_encoding.md)
   - [BatchNorm](batchnorm_exact_encoding.md)
   - [AvgPool2d](avgpool2d_exact_encoding.md)
   - [MaxPool2d](maxpool2d_exact_encoding.md)

## Scope of each note

| Note | Main topic |
|---|---|
| [Interval Bound Propagation](bound_propagation.md) | Bound rules for affine, monotone, arithmetic, and structural operators |
| [Exact Encoding](exact_encoding.md) | Graph-wide LP/MILP variables, constraints, and exactness |
| [Conv2d](conv2d_exact_encoding.md) | Sparse affine matrix for convolution |
| [BatchNorm](batchnorm_exact_encoding.md) | Inference-mode per-channel affine map |
| [AvgPool2d](avgpool2d_exact_encoding.md) | Sparse averaging matrix and pooling divisor |
| [MaxPool2d](maxpool2d_exact_encoding.md) | Full one-hot exact maximum formulation |
