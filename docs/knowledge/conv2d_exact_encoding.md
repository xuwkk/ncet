# Exact Conv2d encoding in NCET

This note explains how NCET converts a fixed PyTorch `Conv2d` operation into
an exact sparse linear equality for an optimization model.

## 1. Main idea

Once a neural network has been trained, the convolution weights and biases are
constants. Therefore, `Conv2d` is an affine operation with respect to its input:

$$
\operatorname{vec}_C(Y)
=
A\operatorname{vec}_C(X)+\bar b.
$$

Here:

- $X$ and $Y$ are the input and output tensors;
- $A$ is a sparse matrix constructed from the convolution kernel;
- $\bar b$ is the channel bias repeated over all spatial positions;
- $\operatorname{vec}_C$ flattens a tensor in C-order (vectorization in sequence of channel, rows, columns).

This is an exact linear equality. Conv2d itself requires no binary variables.
Binary variables may still be needed by a following nonlinear operator such as an unstable ReLU.

## 2. Tensor shapes in NCET

The public NCET interface represents one sample and does not include a batch
dimension:

$$
X\in\mathbb{R}^{C_{\mathrm{in}}\times H_{\mathrm{in}}\times W_{\mathrm{in}}},
$$

$$
W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w},
$$

$$
Y\in\mathbb{R}^{C_{\mathrm{out}}\times H_{\mathrm{out}}\times W_{\mathrm{out}}}.
$$

PyTorch FX shape propagation temporarily uses a batch-size-one tensor with
shape `(1, C, H, W)`. Normalization removes that leading dimension before
creating GraphIR tensors and CVXPY variables.

## 3. Scalar convolution formula

PyTorch `Conv2d` uses cross-correlation: the kernel is not flipped. With unit
dilation, first define the zero-extended input

$$
\widetilde X_{c,h,w}
=
\begin{cases}
X_{c,h,w},
&0\le h<H_{\mathrm{in}},\ 0\le w<W_{\mathrm{in}},\\
0,
&\text{otherwise}.
\end{cases}
$$

One output element is then

$$
Y_{o,i,j}
=
b_o+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{r=0}^{K_h-1}
\sum_{s=0}^{K_w-1}
W_{o,c,r,s}\widetilde X_{c,h,w},
$$

where the corresponding original-input coordinates are

$$
h=iS_h+r-P_h,
$$

$$
w=jS_w+s-P_w.
$$

The symbols mean:

| Symbol | Meaning |
|---|---|
| $o$ | output channel |
| $i,j$ | output row and column |
| $c$ | input channel |
| $r,s$ | kernel row and column |
| $S_h,S_w$ | stride |
| $P_h,P_w$ | padding |

Only coordinates satisfying

$$
0\le h<H_{\mathrm{in}},\qquad 0\le w<W_{\mathrm{in}}
$$

refer to input variables. At all other coordinates, $\widetilde X_{c,h,w}=0$.
The sparse matrix therefore omits those zero-padding coefficients instead of
creating variables for padded positions.

For example, at the upper-left output position with `padding=1`, a kernel
position can produce $h=-1$. NCET does not access `X[-1]`; it recognizes
that the position belongs to the zero-padding region and omits the term.

An alternative implementation could explicitly construct a padded tensor. In
that representation the coordinates would not subtract padding, but the added
padding entries would still be fixed zeros. NCET avoids creating this larger
intermediate object.

## 4. Output dimensions

For the currently supported unit-dilation case,

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

PyTorch shape propagation determines these shapes before the CVXPY backend is
called. The backend uses the resulting GraphIR input and output shapes.

## 5. C-order flattening

NCET flattens tensors in C-order: the last dimension changes fastest. Input
position $(c,h,w)$ becomes column index

$$
\operatorname{col}(c,h,w)
= cH_{\mathrm{in}} W_{\mathrm{in}} + 
hW_{\mathrm{in}} + w.
$$

Output position $(o,i,j)$ becomes row index

$$
\operatorname{row}(o,i,j)
=
oH_{\mathrm{out}} W_{\mathrm{out}} + iW_{\mathrm{out}} + j.
$$

Both input and output positions determine a non-zero entry of the sparse $A$ matrix. They both first go through the channel dimension, then the row dimension, then the column dimension.

Thus,

$$
x=\operatorname{vec}_C(X)
\in\mathbb{R}^{C_{\mathrm{in}}H_{\mathrm{in}}W_{\mathrm{in}}},
$$

$$
y=\operatorname{vec}_C(Y)
\in\mathbb{R}^{C_{\mathrm{out}}H_{\mathrm{out}}W_{\mathrm{out}}}.
$$

In CVXPY, the input vectorization can be achieved as:
```python
input_vector = cp.reshape(
    input_value,
    (input_value.size,),
    order="C",
)
```

changes an expression with shape `(C_in, H_in, W_in)` into an expression with
shape `(C_in * H_in * W_in,)`. It does not create another decision variable.

## 6. Constructing the sparse matrix

The affine matrix has shape

$$
A\in\mathbb{R}^{
(C_{\mathrm{out}}H_{\mathrm{out}}W_{\mathrm{out}})
\times
(C_{\mathrm{in}}H_{\mathrm{in}}W_{\mathrm{in}})
}.
$$

For every valid convolution connection, NCET inserts

$$
A[
\operatorname{row}(o,i,j),
\operatorname{col}(c,h,w)
]
\mathrel{+}=
W_{o,c,r,s}.
$$

Intuitively:

- one row of $A$ represents one output tensor element;
- its nonzero columns identify the receptive-field input elements;
- the values in those columns are the corresponding kernel weights.

Each output depends only on a local receptive field, so most entries of $A$
are zero. NCET records only `(row, column, coefficient)` triples, constructs a
SciPy COO matrix, and converts it to CSC format for CVXPY.

## 7. Bias vector

PyTorch shares one bias value across all spatial positions in an output
channel:

$$
Y_{o,i,j}\leftarrow Y_{o,i,j}+b_o.
$$

Under C-order flattening, all spatial entries of one channel are contiguous.
The flattened bias is therefore

$$
\bar b=
[\underbrace{b_0,\ldots,b_0}_{H_{\mathrm{out}}W_{\mathrm{out}}},
\underbrace{b_1,\ldots,b_1}_{H_{\mathrm{out}}W_{\mathrm{out}}},\ldots]^\top.
$$

The implementation uses

```python
bias_vector = np.repeat(
    bias,
    output_height * output_width,
)
```

If the PyTorch layer has `bias=False`, frontend normalization stores an
equivalent zero-bias constant, so the backend needs no special case.

## 8. Small one-channel example

Suppose the input is `X.shape == (1, 3, 3)`, the kernel is `2 x 2`, stride is
one, and padding is zero. Write

$$
K=
\begin{bmatrix}
k_{00}&k_{01}\\
k_{10}&k_{11}
\end{bmatrix}.
$$

The input and output vectors are

$$
x=[x_{00},x_{01},x_{02},x_{10},x_{11},x_{12},x_{20},x_{21},x_{22}]^\top,
$$

$$
y=[y_{00},y_{01},y_{10},y_{11}]^\top.
$$

The convolution matrix is

$$
A=
\begin{bmatrix}
k_{00}&k_{01}&0&k_{10}&k_{11}&0&0&0&0\\
0&k_{00}&k_{01}&0&k_{10}&k_{11}&0&0&0\\
0&0&0&k_{00}&k_{01}&0&k_{10}&k_{11}&0\\
0&0&0&0&k_{00}&k_{01}&0&k_{10}&k_{11}
\end{bmatrix}.
$$

Every row is the kernel placed over the input positions belonging to one
receptive field.

## 9. CVXPY formulation

The GraphIR input and output keep their three-dimensional sample shapes. The
backend creates one-dimensional expressions only to apply the sparse matrix:

```python
input_vector = cp.reshape(input_value, (input_value.size,), order="C")
output_vector = cp.reshape(output_value, (output_value.size,), order="C")

constraint = output_vector == matrix @ input_vector + bias_vector
```

This constraint is exactly equivalent to the supported PyTorch convolution:

$$
\operatorname{vec}_C(Y)=A\operatorname{vec}_C(X)+\bar b.
$$


## 10. Current support boundary

The first NCET Conv2d formulation supports:

- per-sample tensors with shape `(C, H, W)`;
- multiple input and output channels;
- integer or two-dimensional kernel size, stride, and numeric zero padding;
- `groups == 1`;
- `dilation == (1, 1)`;
- `padding_mode == "zeros"`;
- layers with or without bias.

Grouped/depthwise convolution, non-unit dilation, string padding modes, and
nonzero padding modes are rejected by frontend normalization rather than being
silently approximated.

## 11. Implementation map

The relevant backend functions are:

- `_conv2d_constraint()`: obtains GraphIR constants and CVXPY tensors, flattens
  the tensors, creates the bias vector, and returns the affine equality;
- `_conv2d_affine_matrix()`: maps scalar convolution connections into the
  sparse matrix rows, columns, and coefficients.

The essential equivalence tests cover:

- multiple input and output channels;
- rectangular kernels;
- non-unit stride;
- zero padding and bias;
- intermediate Conv2d and ReLU values;
- a complete Conv2d-ReLU-residual graph.

## Related knowledge

- [Knowledge notes index](index.md)
- [Interval Bound Propagation](bound_propagation.md)
- [Exact Encoding](exact_encoding.md)
