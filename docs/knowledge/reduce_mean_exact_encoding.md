# Exact ReduceMean encoding in NCET

`ReduceMean` averages a tensor over one or more fixed sample dimensions. NCET
normalizes both `torch.mean(x, dim=...)` and `x.mean(dim=...)` to this canonical
operator.

## Mathematical operation

Let $\mathcal D$ be the reduced axes and let

$$
K=\prod_{d\in\mathcal D}n_d
$$

be the number of elements contributing to each output. For every fixed tuple
of coordinates $\mathbf j$ on the remaining axes,

$$
Y_{\mathbf j}
=
\frac{1}{K}
\sum_{\mathbf i_{\mathcal D}}
X_{\mathbf j,\mathbf i_{\mathcal D}}.
$$

For example, global spatial averaging of a per-sample CNN tensor
$X\in\mathbb R^{C\times H\times W}$ is

$$
Y_c
=
\frac{1}{HW}
\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}X_{c,h,w}.
$$

`keepdim=False` removes the reduced axes, while `keepdim=True` retains them
with size one. It does not change the coefficients or values.

## Batch and GraphIR dimensions

Shape propagation temporarily represents a sample tensor `(C, H, W)` as
`(1, C, H, W)`. A PyTorch reduction over dimensions `(-2, -1)` therefore
becomes GraphIR sample dimensions `(1, 2)` after NCET removes the temporary
batch axis. Reducing dimension 0, or omitting `dim`, is rejected because it
would remove that axis during shape propagation.

The canonical attributes are:

```python
{"dims": (1, 2), "keepdim": False}
```

## Interval bounds

Every averaging coefficient is nonnegative, so ReduceMean is monotone. Given

$$
L_X\le X\le U_X,
$$

NCET propagates

$$
L_Y=\operatorname{mean}(L_X,\mathcal D),
\qquad
U_Y=\operatorname{mean}(U_X,\mathcal D).
$$

## Exact CVXPY equality

NCET creates a continuous output variable with the shape recorded by its
`TensorSpec` and enforces

$$
Y=\frac{1}{K}\sum_{\mathcal D}X.
$$

The backend sums axes in descending order. When `keepdim=False`, removing a
higher axis then leaves every lower axis number unchanged. This implementation
also avoids relying on solver canonicalization of a tuple-valued reduction
axis.

ReduceMean introduces no binary variable and no relaxation. Subject to the
documented static-dimension restrictions, the equality represents the
PyTorch operation exactly.

## Supported boundary

- `torch.mean` and `Tensor.mean`;
- one or more explicit static integer dimensions;
- static `keepdim=True` or `False`;
- floating-point tensors without a `dtype` conversion;
- no reduction of tracing batch dimension 0;
- no caller-provided `out` tensor.

