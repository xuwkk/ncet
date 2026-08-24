# Exactness contract

This document defines NCET's semantic guarantee. It does not define which
PyTorch spellings or canonical operators are supported; those boundaries are
maintained in
[PyTorch-to-GraphIR operator mapping](pytorch_to_ir_operator_mapping.md) and
[current operator support](supported_operators.md).

## 1. Input domain

NCET encodes one evaluation of a supported inference graph over finite
elementwise input bounds. For one input tensor, the encoded domain is the box

$$
\mathcal X=\{x\mid L\leq x\leq U\}.
$$

For multiple inputs, $x$, $L$, and $U$ denote the corresponding collection of
input tensors. The bounds must have the exact per-sample input shapes, contain
only finite values, and satisfy $L\leq U$ elementwise.

This is a box-bound contract. A symmetric box may represent an
$L_\infty$ neighborhood, but NCET does not directly represent coupled
$L_1$- or $L_2$-norm input domains.

## 2. Required model conditions

The current exactness guarantee requires:

- `model.eval()` and fixed parameters and buffers;
- deterministic execution and static tensor ranks and dimensions;
- successful capture by `torch.fx.symbolic_trace()` without data-dependent
  Python control flow;
- no in-place mutation or unsupported tensor aliasing;
- only PyTorch spellings, canonical operators, and parameter cases inside the
  documented current support boundary;
- a model that accepts a leading batch dimension, preserves batch axis 0, and
  computes samples independently.

The public bounds, GraphIR tensors, propagated bounds, and CVXPY variables are
per-sample and contain no batch dimension. NCET adds a leading singleton batch
dimension only for PyTorch shape propagation. The detailed internal shape
lifecycle is documented in [developer_note.md](developer_note.md).

## 3. Formulation guarantee

Let $f$ be the supported PyTorch inference graph and let $C$ denote the
constraints returned by NCET. Up to floating-point parameter conversion and
solver feasibility tolerances, NCET guarantees

$$
\left\{(x,y)\;\middle|\;
\begin{aligned}
&x\in\mathcal X,\\
&\text{there exist intermediate and binary variables satisfying }C
\end{aligned}
\right\}
=
\left\{(x,f(x))\mid x\in\mathcal X\right\}.
$$

The same statement applies componentwise to multiple graph inputs and outputs.
All encoded intermediate tensor variables are constrained to the values
produced by the corresponding supported graph operations.

In particular:

- affine and structural operators use exact linear equalities;
- ReLU and MaxPool2d use exact integer formulations rather than convex
  approximations;
- interval bound propagation may be conservative, but its bounds must be
  sound; looser sound bounds can weaken performance without changing the
  integer feasible set;
- formulation-size changes such as reduced ReLU binaries must preserve the
  same feasible set.

## 4. Solver boundary

Exactness is a property of the generated mixed-integer formulation. Binary
variables must retain their integer domains; solving only its LP relaxation
does not preserve the neural-network graph. Numerical feasibility and
optimality are additionally subject to the selected solver and its tolerances.

NCET creates variables and constraints but does not choose the surrounding
optimization objective. Adding external constraints or an objective changes
the user's larger optimization problem, not the meaning of the encoded neural
network.

## 5. Failure boundary

NCET does not silently approximate an unsupported operation. Capture,
normalization, bound propagation, or encoding fails instead of returning a
partial `MILPEncoding`. Operators and dynamic behaviors outside the current
boundary remain non-goals unless an exact formulation is added explicitly.
