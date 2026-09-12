# Interval Bound Propagation in NCET

NCET groups operators by the rule used to propagate their tensor bounds. This
avoids overlapping mathematical labels: for example, AveragePool2d is both
linear and monotone, but its bounds are most simply propagated by monotonicity.

For one operation over an independent input box, the formulas below give exact
elementwise output ranges. Across a deeper network, IBP can become
conservative because it does not preserve correlations between tensor
elements or graph branches.

## 1. Signed affine propagation

Linear, Conv2d, and BatchNorm may contain both positive and negative fixed
coefficients, so their bounds require a positive/negative coefficient split.

### Linear layer

Consider a linear layer

$$
y = Wx + b,
$$

where $x \in \mathbb{R}^{n}$, $W \in \mathbb{R}^{m \times n}$,
$b \in \mathbb{R}^{m}$, and the input is bounded elementwise by

$$
L_x \leq x \leq U_x.
$$

Split the weight matrix into its positive and negative parts:

$$
W^+ = \max(W,0), \qquad W^- = \min(W,0).
$$

The propagated bounds are then

$$
L_y = W^+L_x + W^-U_x + b,
$$

$$
U_y = W^+U_x + W^-L_x + b.
$$

### Conv2d layer

A Conv2d layer is affine with respect to its input:

$$
Y = \operatorname{conv2d}(X,W)+b,
$$

where the input bounds satisfy

$$
L_X \leq X \leq U_X.
$$

As for a Linear layer, split the fixed kernel into positive and negative
parts:

$$
W^+ = \max(W,0), \qquad W^- = \min(W,0).
$$

Using the same stride, padding, dilation, and groups settings as the original
layer, the propagated bounds are

$$
L_Y
= \operatorname{conv2d}(L_X,W^+)
+ \operatorname{conv2d}(U_X,W^-)+b,
$$

$$
U_Y
= \operatorname{conv2d}(U_X,W^+)
+ \operatorname{conv2d}(L_X,W^-)+b.
$$

The bound propagation exactly follows the above formulas and uses `torch.nn.functional.conv2d`
directly.

NCET applies these formulas to per-sample tensors of shape `(C, H, W)`; a
temporary batch dimension is used only for the PyTorch call and is removed
from the returned bounds. Padded positions are fixed at zero rather than
treated as uncertain inputs.

See [Exact Conv2d encoding in NCET](conv2d_exact_encoding.md) for the
corresponding optimization constraint and sparse-matrix construction.

### BatchNorm layer

In evaluation mode, BatchNorm is a fixed per-channel affine map:

$$
Y_{c,\ldots}=a_cX_{c,\ldots}+d_c,
$$

where

$$
a_c=\frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}},
\qquad
d_c=\beta_c-a_c\mu_c.
$$

Here, $\mu_c$ and $\sigma_c^2$ are the stored running statistics. For a
non-affine module, $\gamma_c=1$ and $\beta_c=0$. Define
$a_c^+=\max(a_c,0)$ and $a_c^-=\min(a_c,0)$. The propagated bounds are

$$
L_Y=a^+\odot L_X+a^-\odot U_X+d,
$$

$$
U_Y=a^+\odot U_X+a^-\odot L_X+d,
$$

with the channel vectors broadcast over spatial or sequence dimensions. See
[Exact BatchNorm encoding in NCET](batchnorm_exact_encoding.md) for the
normalization and exact equality.

## 2. Order-preserving unary operations

ReLU, AvgPool2d, AdaptiveAvgPool2d, and MaxPool2d are order-preserving unary
operations. Let $f$ denote any one of them. Given

$$
L_X \leq X \leq U_X,
$$

monotonicity gives

$$
f(L_X) \leq f(X) \leq f(U_X).
$$

Therefore, all three operators use the same IBP rule:

$$
L_Y=f(L_X), \qquad U_Y=f(U_X).
$$

### ReLU

ReLU is elementwise and nondecreasing:

$$
y=\max(0,x).
$$

Its bounds are

$$
L_y=\max(0,L_x), \qquad U_y=\max(0,U_x).
$$

### AveragePool2d layer

AveragePool2d is a linear operation whose averaging coefficients are all
nonnegative. Consequently, there is no need to split its coefficients into
positive and negative parts.

NCET uses the original kernel size, stride, padding, and
`count_include_pad` setting. Zero-padded positions are fixed constants, and
`count_include_pad` determines whether they contribute to the divisor.

See [Exact AveragePool2d encoding in NCET](avgpool2d_exact_encoding.md) for
the corresponding averaging-matrix construction.

### AdaptiveAvgPool2d layer

AdaptiveAvgPool2d also averages with nonnegative coefficients, but computes
each pooling window from the input shape and requested output shape. Its bounds
are therefore

$$
L_Y=\operatorname{adaptive\_avgpool2d}(L_X),
\qquad
U_Y=\operatorname{adaptive\_avgpool2d}(U_X).
$$

See [Exact AdaptiveAvgPool2d encoding in NCET](adaptive_avgpool2d_exact_encoding.md)
for the window boundaries and sparse averaging matrix.

### MaxPool2d layer

MaxPool2d is nonlinear but monotone: increasing any candidate in a pooling
window cannot decrease the window maximum. Its bounds are therefore

$$
L_Y=\operatorname{maxpool2d}(L_X), \qquad
U_Y=\operatorname{maxpool2d}(U_X).
$$

NCET uses the original kernel size, stride, and padding. Padded positions are
not valid maximum candidates, matching PyTorch's implicit negative-infinity
padding.

For both pooling operators, an omitted stride defaults to the kernel size.
NCET temporarily adds a batch-size-one dimension for the corresponding
PyTorch functional call and removes it from the returned bounds.

See [Exact MaxPool2d encoding in NCET](maxpool2d_exact_encoding.md) for
the corresponding one-hot exact formulation.

## 3. Multi-input arithmetic operations

These rules combine the bounds of two input tensors. Given

$$
L_X\leq X\leq U_X, \qquad L_Z\leq Z\leq U_Z,
$$

Add is increasing in both inputs:

$$
L_Y=L_X+L_Z, \qquad U_Y=U_X+U_Z.
$$

Sub is increasing in its first input and decreasing in its second input:

$$
L_Y=L_X-U_Z, \qquad U_Y=U_X-L_Z.
$$

These ranges are exact when $X$ and $Z$ independently span their input boxes.
If they are correlated graph branches, as in a residual connection, the
formulas remain sound but can be conservative.

## 4. Structural operations

Structural operators only assemble, select, or reorder tensor elements, so
NCET applies the same structural transformation to the lower and upper
bounds.

For Concat,

$$
L_Y=\operatorname{concat}(L_{X_1},\ldots,L_{X_k}), \qquad
U_Y=\operatorname{concat}(U_{X_1},\ldots,U_{X_k}).
$$

For Flatten, Reshape, Permute, Transpose, GetItem, and Slice,

$$
L_Y=g(L_X), \qquad U_Y=g(U_X),
$$

where $g$ is the same shape transformation, axis reordering, or static index
operation recorded in the GraphIR node. These operations do not introduce
additional interval relaxation.

## Related knowledge

- [Knowledge notes index](index.md)
- [Exact Encoding](exact_encoding.md)
- [Exact Conv2d encoding](conv2d_exact_encoding.md)
- [Exact BatchNorm encoding](batchnorm_exact_encoding.md)
- [Exact AdaptiveAvgPool2d encoding](adaptive_avgpool2d_exact_encoding.md)
- [Exact AvgPool2d encoding](avgpool2d_exact_encoding.md)
- [Exact MaxPool2d encoding](maxpool2d_exact_encoding.md)
