# Current operator support

This document is the authoritative capability boundary for the current NCET
implementation. It describes canonical GraphIR operators and their accepted
semantics, not future roadmap targets and not every operation that PyTorch FX
can trace.

The accepted PyTorch spellings that produce these canonical operators are
listed separately in
[PyTorch-to-GraphIR operator mapping](pytorch_to_ir_operator_mapping.md).
The semantic guarantee for a graph inside this boundary is defined by the
[exactness contract](exactness_contract.md).

## Meaning of supported

A canonical operator is supported only when all of the following exist for its
documented parameter range:

1. frontend normalization into GraphIR;
2. graph-based interval bound propagation;
3. an exact CVXPY formulation;
4. essential operator and network-integration tests.

An FX operation is not supported merely because `torch.fx.symbolic_trace()`
can capture it. Any canonical operator or parameter case outside the table
below is outside the current exact boundary.

## Current canonical operator boundary

| Category | Canonical operator | Exact formulation | Current restrictions |
|---|---|---|---|
| Graph | `Input` | Continuous tensor variable with elementwise box bounds | One tensor per FX placeholder; multiple model inputs are supported |
| Graph | `Output` | Reference to existing graph tensor variables | One or more tensor outputs; creates no new tensor or variable |
| Affine | `Linear` | Linear equality | Fixed `nn.Linear` parameters; acts on the last tensor dimension; bias may be present or absent |
| Affine | `Conv2d` | Sparse affine equality | Per-sample shape `(C,H,W)`; fixed `nn.Conv2d`; `groups=1`; `dilation=(1,1)`; numeric padding with `padding_mode="zeros"`; bias optional |
| Affine | `BatchNorm` | Per-channel affine equality | `nn.BatchNorm1d` on `(C,)` or `(C,L)` and `nn.BatchNorm2d` on `(C,H,W)`; evaluation mode; fixed running statistics; affine or non-affine modules |
| Pooling | `AdaptiveAvgPool2d` | Sparse linear equality | Per-sample shape `(C,H,W)`; module or functional form; static scalar or length-2 output size; each entry is a positive integer or `None`, where `None` preserves that input dimension |
| Pooling | `AvgPool2d` | Sparse linear equality | Per-sample shape `(C,H,W)`; scalar or 2-D kernel, stride, and padding; `stride=None` uses the kernel size; `ceil_mode=False`; `divisor_override=None`; either value of `count_include_pad` |
| Pooling | `MaxPool2d` | Full exact one-hot formulation | Per-sample shape `(C,H,W)`; scalar or 2-D kernel, stride, and padding; `stride=None` uses the kernel size; `dilation=(1,1)`; `ceil_mode=False`; `return_indices=False` |
| Structural | `Identity` | Elementwise equality | `nn.Identity`; `nn.Dropout`, `nn.Dropout1d/2d/3d`, or corresponding functional calls only with `training=False`; in-place forms unsupported |
| Activation | `ReLU` | Exact big-M or stable equality | Elementwise ReLU; `relu_binary_mode` may be `"full"` or `"reduced"`; in-place forms are unsupported |
| Arithmetic | `Add`, `Sub` | Linear equality | Exactly two tensor operands; `alpha=1`; scalar or constant operands are not canonicalized |
| Composition | `Concat` | Exact output-slice equalities | Static tensor inputs and dimension; tracing batch axis cannot be concatenated; `out` must be absent or `None` |
| Shape | `Flatten` | C-order element-preserving equality | Static `start_dim` and `end_dim`; flattened range cannot include the tracing batch axis |
| Shape | `Reshape` | C-order element-preserving equality | Statically resolved output shape; leading singleton tracing batch axis must remain present |
| Shape | `Permute` | Native N-D axis permutation equality | Complete static permutation; tracing batch axis remains first |
| Shape | `Transpose` | Native N-D axis permutation equality | Static dimensions; tracing batch axis cannot be exchanged |
| Indexing | `GetItem`, `Slice` | Exact static index selection | Static integer/slice/ellipsis/new-axis syntax; positive slice steps; tracing batch axis remains unchanged |

The graph may contain branches, residual connections, concatenation, shared
modules, and multiple graph inputs or outputs. Each non-boundary canonical
operation currently produces one tensor.

## Outside the current boundary

Standalone `Constant` operators, constant arithmetic, and other canonical
operators not shown above are not currently supported. Linear,
Conv2d, and BatchNorm fixed arrays are lifted into `GraphIR.constants`, but
this does not create standalone Constant operators.

Planned operators and formulations remain outside this published capability
boundary; appearing in a development roadmap does not imply current support.
