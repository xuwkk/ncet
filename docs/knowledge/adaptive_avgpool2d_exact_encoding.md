# Exact AdaptiveAvgPool2d encoding in NCET

`AdaptiveAvgPool2d` chooses its pooling windows from the input shape and a
requested output shape. Once those shapes are static, it is a fixed linear
operator and can be encoded exactly without binary variables.

## 1. Adaptive windows

Let the per-sample input and output shapes be `(C,H_in,W_in)` and
`(C,H_out,W_out)`. For output row $i$ and column $j$, PyTorch uses

$$
h_{\mathrm{start}}(i)
=\left\lfloor\frac{iH_{\mathrm{in}}}{H_{\mathrm{out}}}\right\rfloor,
\qquad
h_{\mathrm{end}}(i)
=\left\lceil\frac{(i+1)H_{\mathrm{in}}}{H_{\mathrm{out}}}\right\rceil,
$$

$$
w_{\mathrm{start}}(j)
=\left\lfloor\frac{jW_{\mathrm{in}}}{W_{\mathrm{out}}}\right\rfloor,
\qquad
w_{\mathrm{end}}(j)
=\left\lceil\frac{(j+1)W_{\mathrm{in}}}{W_{\mathrm{out}}}\right\rceil.
$$

The valid window is the Cartesian product of these half-open intervals. When
an input dimension is not divisible by its output dimension, adjacent windows
may overlap.

## 2. Scalar formula

Define

$$
D_{i,j}
=\left(h_{\mathrm{end}}(i)-h_{\mathrm{start}}(i)\right)
 \left(w_{\mathrm{end}}(j)-w_{\mathrm{start}}(j)\right).
$$

Each channel is pooled independently:

$$
Y_{c,i,j}
=\frac{1}{D_{i,j}}
\sum_{h=h_{\mathrm{start}}(i)}^{h_{\mathrm{end}}(i)-1}
\sum_{w=w_{\mathrm{start}}(j)}^{w_{\mathrm{end}}(j)-1}
X_{c,h,w}.
$$

For output size `(1,1)`, this reduces to global average pooling over each
channel.

## 3. Sparse matrix formulation

Using C-order vectorization, NCET constructs a sparse matrix
$A_{\mathrm{adaptive}}$ with

$$
A_{
\operatorname{row}(c,i,j),
\operatorname{col}(c',h,w)
}
=
\begin{cases}
\dfrac{1}{D_{i,j}},
&c'=c\text{ and }(h,w)\text{ belongs to window }(i,j),\\[6pt]
0,&\text{otherwise}.
\end{cases}
$$

The exact CVXPY equality is

$$
\operatorname{vec}_C(Y)
=A_{\mathrm{adaptive}}\operatorname{vec}_C(X).
$$

## 4. Interval bounds

All matrix coefficients are nonnegative, so the operation is monotone. Given
$L_X\le X\le U_X$, NCET propagates

$$
L_Y=\operatorname{AdaptiveAvgPool2d}(L_X),
\qquad
U_Y=\operatorname{AdaptiveAvgPool2d}(U_X).
$$

These are the exact elementwise output ranges over the input box. Correlations
between overlapping output windows are not retained by IBP.

## 5. Supported forms

NCET supports `nn.AdaptiveAvgPool2d` and
`torch.nn.functional.adaptive_avg_pool2d` on per-sample `(C,H,W)` tensors.
The output size must be static and may be a positive integer or a length-two
sequence whose entries are positive integers or `None`. A `None` entry keeps
the corresponding input dimension. Normalization resolves every accepted form
to a fixed `(H_out,W_out)` GraphIR attribute.
