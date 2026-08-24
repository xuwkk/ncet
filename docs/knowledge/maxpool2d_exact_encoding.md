# Exact MaxPool2d encoding in NCET

This note explains the current full one-hot formulation used to encode a
PyTorch `MaxPool2d` operation exactly.

## 1. Tensor shapes

NCET represents one sample without a batch dimension:

$$
X\in\mathbb{R}^{C\times H_{\mathrm{in}}\times W_{\mathrm{in}}},
$$

$$
Y\in\mathbb{R}^{C\times H_{\mathrm{out}}\times W_{\mathrm{out}}}.
$$

For kernel $(K_h,K_w)$, stride $(S_h,S_w)$, padding $(P_h,P_w)$,
unit dilation, and `ceil_mode=False`,

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

MaxPool2d operates independently on each channel.

## 2. Valid window candidates

For output position $(c,i,j)$, kernel position $(r,t)$ refers to the original
input coordinate

$$
h=iS_h+r-P_h,
$$

$$
w=jS_w+t-P_w.
$$

The valid kernel-position set is

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

PyTorch conceptually pads MaxPool inputs with negative infinity. Padding
positions can never win the maximum, so NCET omits them instead of creating
fixed padding values or selectors.

## 3. Scalar operation

One output element is

$$
Y_{c,i,j}
=
\max_{(r,t)\in\mathcal V_{i,j}}
X_{c,\,iS_h+r-P_h,\,jS_w+t-P_w}.
$$

For a fixed output window, write its valid candidates as $x_q$ for
$q\in\mathcal V$ and its output as $y$:

$$
y=\max_{q\in\mathcal V}x_q.
$$

Suppose interval propagation provides

$$
L_q\le x_q\le U_q.
$$

The output interval is

$$
L_y=\max_{q\in\mathcal V}L_q,
$$

$$
U_y=\max_{q\in\mathcal V}U_q.
$$

## 4. Full one-hot formulation

The current formulation creates one binary selector for every valid candidate:

$$
z_q\in\{0,1\},
\qquad q\in\mathcal V.
$$

Exactly one candidate is selected:

$$
\sum_{q\in\mathcal V}z_q=1.
$$

The complete formulation is

$$
L_y\le y\le U_y,
$$

$$
y\ge x_q,
\qquad q\in\mathcal V,
$$

$$
y
\le
x_q+(U_y-L_q)(1-z_q),
\qquad q\in\mathcal V,
$$

$$
\sum_{q\in\mathcal V}z_q=1,
$$

$$
z_q\in\{0,1\},
\qquad q\in\mathcal V.
$$

No stable-window or dominated-candidate elimination is currently applied.

## 5. Exactness

The one-hot equality selects some $q^\star$ with $z_{q^\star}=1$. Its upper
constraint becomes

$$
y\le x_{q^\star}.
$$

The lower constraint for the same candidate gives

$$
y\ge x_{q^\star},
$$

so $y=x_{q^\star}$. Since the formulation also requires $y\ge x_q$ for every
candidate,

$$
y=x_{q^\star}=\max_q x_q.
$$

If several candidates tie, any one of them may be selected without changing
the output value.

## 6. Candidate-specific big-M

Each upper constraint uses

$$
M_q=U_y-L_q.
$$

When $z_q=0$,

$$
x_q+M_q
=
x_q+U_y-L_q
\ge U_y,
$$

because $x_q\ge L_q$. The inactive constraint therefore does not restrict an
output already bounded above by $U_y$. When $z_q=1$, the big-M term disappears
and forces $y\le x_q$.

## 7. Interval bound propagation

MaxPool2d is monotone in every input, so NCET applies the same pooling operation
to both interval endpoints:

$$
\underline Y=\operatorname{MaxPool2d}(\underline X),
$$

$$
\overline Y=\operatorname{MaxPool2d}(\overline X).
$$

The implementation temporarily adds a batch-size-one dimension only while
calling PyTorch's pooling function.

## 8. Vectorized backend representation

The backend stores all selectors of one MaxPool node in a flat binary vector
$z$. For every selector it records:

- its output tensor coordinate $(c,i,j)$;
- its valid input coordinate $(c,h,w)$.

A sparse incidence matrix $S$ contains one row per output element and one
column per candidate. Its entries are

$$
S_{o,q}
=
\begin{cases}
1,&\text{if candidate }q\text{ belongs to output }o,\\
0,&\text{otherwise}.
\end{cases}
$$

All one-hot equalities are then expressed together as

$$
Sz=\mathbf 1.
$$

The candidate inequalities are also vectorized. This changes only the CVXPY
construction, not the mathematical formulation.

## 9. Binary count

The full formulation introduces

$$
N_{\mathrm{binary}}
=
\sum_{c=0}^{C-1}
\sum_{i=0}^{H_{\mathrm{out}}-1}
\sum_{j=0}^{W_{\mathrm{out}}-1}
|\mathcal V_{i,j}|
$$

binary variables. Boundary windows can have fewer selectors because padding
positions are not candidates.

## 10. Current support boundary

The exact formulation supports:

- per-sample tensors with shape `(C, H, W)`;
- integer or two-dimensional kernel size, stride, and numeric padding;
- `dilation == (1, 1)`;
- `ceil_mode == False`;
- `return_indices == False`;
- full one-hot selection for all valid candidates.

The frontend explicitly rejects non-unit dilation, ceiling-mode output shapes,
and returned pooling indices.

## Related knowledge

- [Knowledge notes index](index.md)
- [Interval Bound Propagation](bound_propagation.md)
- [Exact Encoding](exact_encoding.md)
