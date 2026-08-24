# Exact AvgPool2d encoding in NCET

This note explains how NCET can represent a fixed PyTorch `AvgPool2d`
operation as an exact sparse linear equality and propagate interval bounds
through it.

**Logic**:
1. Output index: $(c_o,i,j)$
2. Kernel position: $(r,t)$
3. 1 and 2 together determine the corresponding input index to be weighted: $(c_i, h, w)$ with $h=iS_h+r-P_h$ and $w=jS_w+t-P_w$. Note that $c_o = c_i$.
4. Meanwhile, $h$ and $w$ must satisfy $0\le h<H_{\mathrm{in}}$ and $0\le w<W_{\mathrm{in}}$; $r$ and $t$ must satisfy $0\le r<K_h$ and $0\le t<K_w$.
5. Then for all valid input indices by varying $r$ and $t$ belong to the output position $(i,j)$, the coefficient is $\frac{1}{D_{i,j}}$.
6. Transform the output and input indices into the position of the vectorized output and input tensors, which are also the row and column indices of the sparse matrix $A$.

## 1. Main idea

Average pooling computes a weighted sum of the input elements in each pooling
window. The weights are fixed, nonnegative averaging coefficients. Therefore,
`AvgPool2d` is a linear operator:

$$
\operatorname{vec}_C(Y)=A\operatorname{vec}_C(X).
$$

Here:

- $X$ and $Y$ are the input and output tensors;
- $A$ is a sparse matrix containing the pooling connections and averaging
  coefficients;
- $\operatorname{vec}_C$ flattens a tensor in C-order: channel, row, and then
  column.

This equality is exact. `AvgPool2d` requires continuous output variables but
no binary variables.

## 2. Tensor shapes

NCET represents one sample without a batch dimension:

$$
X\in\mathbb{R}^{C\times H_{\mathrm{in}}\times W_{\mathrm{in}}},
$$

$$
Y\in\mathbb{R}^{C\times H_{\mathrm{out}}\times W_{\mathrm{out}}}.
$$

Average pooling operates independently on every channel. It changes the
spatial dimensions but does not mix channels.

Let:

- $(K_h,K_w)$ be the kernel size;
- $(S_h,S_w)$ be the stride;
- $(P_h,P_w)$ be the padding.

For `ceil_mode=False`, the output dimensions are

$$
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+2P_h-K_h}{S_h}
\right\rfloor+1,
$$

$$
W_{\mathrm{out}}
=
\left\lfloor
\frac{W_{\mathrm{in}}+2P_w-K_w}{S_w}
\right\rfloor+1.
$$

PyTorch FX shape propagation determines these shapes before GraphIR and the
optimization backend are constructed.

## 3. Pooling-window coordinates

Consider output element $Y_{c,i,j}$. For kernel position $(r,t)$, the
corresponding coordinate in the original, unpadded input is

$$
h=iS_h+r-P_h,
$$

$$
w=jS_w+t-P_w,
$$

where

$$
0\le r<K_h,\qquad 0\le t<K_w.
$$

Only coordinates satisfying

$$
0\le h<H_{\mathrm{in}},\qquad
0\le w<W_{\mathrm{in}}
$$

refer to input tensor elements. Coordinates outside this range are padding
positions whose values are fixed to zero.

Define the valid kernel positions for output location $(i,j)$ as

$$
\mathcal V_{i,j}
=
\left\{
(r,t)\ \middle|\
\begin{aligned}
&0\le r<K_h,\quad 0\le t<K_w,\\
&0\le iS_h+r-P_h<H_{\mathrm{in}},\\
&0\le jS_w+t-P_w<W_{\mathrm{in}}
\end{aligned}
\right\}.
$$

## 4. Scalar AvgPool2d formula

When `count_include_pad=True`, padding zeros are included in the denominator:

$$
Y_{c,i,j}
=
\frac{1}{K_hK_w}
\sum_{(r,t)\in\mathcal V_{i,j}}
X_{c,\,iS_h+r-P_h,\,jS_w+t-P_w}.
$$

The padding terms do not appear in the sum because their values are zero, but
they are still counted by the denominator $K_hK_w$.

When `count_include_pad=False`, only valid input elements are counted:

$$
Y_{c,i,j}
=
\frac{1}{|\mathcal V_{i,j}|}
\sum_{(r,t)\in\mathcal V_{i,j}}
X_{c,\,iS_h+r-P_h,\,jS_w+t-P_w}.
$$

It is convenient to define

$$
D_{i,j}
=
\begin{cases}
K_hK_w,
&\text{if `count\_include\_pad=True`},\\
|\mathcal V_{i,j}|,
&\text{if `count\_include\_pad=False`}.
\end{cases}
$$

Then both cases share the formula

$$
Y_{c,i,j}
=
\frac{1}{D_{i,j}}
\sum_{(r,t)\in\mathcal V_{i,j}}
X_{c,\,iS_h+r-P_h,\,jS_w+t-P_w}.
$$

## 5. C-order flattening

The input element $(c,h,w)$ becomes column index

$$
\operatorname{col}(c,h,w)
=
(cH_{\mathrm{in}}+h)W_{\mathrm{in}}+w.
$$

The output element $(c,i,j)$ becomes row index

$$
\operatorname{row}(c,i,j)
=
(cH_{\mathrm{out}}+i)W_{\mathrm{out}}+j.
$$

Therefore,

$$
x=\operatorname{vec}_C(X)
\in\mathbb{R}^{CH_{\mathrm{in}}W_{\mathrm{in}}},
$$

$$
y=\operatorname{vec}_C(Y)
\in\mathbb{R}^{CH_{\mathrm{out}}W_{\mathrm{out}}}.
$$

## 6. Constructing the sparse matrix

The pooling matrix has shape

$$
A\in
\mathbb{R}^{
(CH_{\mathrm{out}}W_{\mathrm{out}})
\times
(CH_{\mathrm{in}}W_{\mathrm{in}})
}.
$$

To state each coefficient precisely, use $c_o$ for an output channel and
$c_i$ for an input channel. Matrix entries are defined only for valid output
and input tensor coordinates:

$$
0\le c_o<C,\qquad 0\le i<H_{\mathrm{out}},\qquad
0\le j<W_{\mathrm{out}},
$$

$$
0\le c_i<C,\qquad 0\le h<H_{\mathrm{in}},\qquad
0\le w<W_{\mathrm{in}}.
$$

Within these coordinate domains, each coefficient is

$$
A_{
\operatorname{row}(c_o,i,j),
\operatorname{col}(c_i,h,w)
}
=
\begin{cases}
\dfrac{1}{D_{i,j}},
&
\begin{aligned}
&c_i=c_o,\\
&0\le h<H_{\mathrm{in}},\\
&0\le w<W_{\mathrm{in}},\\
&0\le h-(iS_h-P_h)<K_h,\\
&0\le w-(jS_w-P_w)<K_w,
\end{aligned}
\\[6pt]
0,
&\text{otherwise}.
\end{cases}
$$

The two window-membership conditions are equivalent to

$$
iS_h-P_h\le h<iS_h-P_h+K_h,
$$

$$
jS_w-P_w\le w<jS_w-P_w+K_w.
$$

They test whether input position $(h,w)$ belongs to the pooling window for
output position $(i,j)$. The condition $c_i=c_o$ expresses that average
pooling does not mix channels. The explicit input-size restrictions exclude
padding coordinates: padding positions have fixed value zero and therefore do
not receive columns in $A$. When `count_include_pad=True`, those omitted zero
positions still contribute to $D_{i,j}$; when it is `False`, they contribute
to neither the matrix nor the divisor.

The scalar and matrix forms are connected by

$$
Y_{c,i,j}
=
\sum_{h=0}^{H_{\mathrm{in}}-1}
\sum_{w=0}^{W_{\mathrm{in}}-1}
A_{
\operatorname{row}(c,i,j),
\operatorname{col}(c,h,w)
}
X_{c,h,w}.
$$

Each row of $A$ therefore represents one output element. Its nonzero columns
identify the valid input elements in that element's pooling window.

## 7. Small example without padding

Suppose `X.shape == (1, 3, 3)`, the kernel is `2 x 2`, stride is one, padding
is zero, and `count_include_pad=True`. The flattened input and output are

$$
x=
[x_{00},x_{01},x_{02},x_{10},x_{11},x_{12},x_{20},x_{21},x_{22}]^\top,
$$

$$
y=[y_{00},y_{01},y_{10},y_{11}]^\top.
$$

The matrix is

$$
A=
\frac{1}{4}
\begin{bmatrix}
1&1&0&1&1&0&0&0&0\\
0&1&1&0&1&1&0&0&0\\
0&0&0&1&1&0&1&1&0\\
0&0&0&0&1&1&0&1&1
\end{bmatrix}.
$$

For example, the first row gives

$$
y_{00}=\frac{x_{00}+x_{01}+x_{10}+x_{11}}{4}.
$$

## 8. Effect of padding on the denominator

Consider a `2 x 2` kernel at a boundary where its window contains one valid
input value $x$ and three padding zeros.

For `count_include_pad=True`,

$$
y=\frac{x+0+0+0}{4}=\frac{x}{4}.
$$

For `count_include_pad=False`,

$$
y=\frac{x}{1}=x.
$$

Thus, padding coordinates never become optimization variables. They only
affect the denominator when padding is included in the average.

## 9. Interval bound propagation

All entries of $A$ are nonnegative:

$$
A_{pq}\ge 0.
$$

Consequently, AvgPool2d is monotone in every input element. Given input
bounds

$$
\underline x\le x\le\overline x,
$$

the exact interval propagation rule is

$$
\underline y=A\underline x,
$$

$$
\overline y=A\overline x.
$$

Equivalently, the same AvgPool2d operation can be applied independently to the
lower and upper tensors:

```python
output_lower = avg_pool2d(input_lower)
output_upper = avg_pool2d(input_upper)
```

Unlike a general affine operator with negative coefficients, there is no need
to split the matrix into positive and negative parts.

## 10. Exact optimization constraint

The backend flattens the three-dimensional CVXPY expressions only for applying
the sparse matrix:

```python
input_vector = cp.reshape(input_value, (input_value.size,), order="C")
output_vector = cp.reshape(output_value, (output_value.size,), order="C")

constraint = output_vector == matrix @ input_vector
```

This adds the exact equality

$$
\operatorname{vec}_C(Y)=A\operatorname{vec}_C(X).
$$

The reshape operations create views of existing expressions, not additional
decision variables. AvgPool2d introduces one continuous output tensor and no
binary variables.

## 11. Current NCET support boundary

The exact formulation supports:

- per-sample tensors with shape `(C, H, W)`;
- integer or two-dimensional kernel size, stride, and padding;
- `ceil_mode=False`;
- both values of `count_include_pad`;
- `divisor_override=None`.

Supporting `ceil_mode=True` requires matching PyTorch's additional boundary
window rules. A non-`None` `divisor_override` changes $D_{i,j}$ to the supplied
constant. The frontend explicitly rejects these two cases.

## Related knowledge

- [Knowledge notes index](index.md)
- [Interval Bound Propagation](bound_propagation.md)
- [Exact Encoding](exact_encoding.md)
