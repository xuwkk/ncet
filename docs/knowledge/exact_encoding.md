# Exact Encoding in NCET

NCET converts a bounded `GraphIR` into a system of CVXPY constraints. The
result is a linear constraint system when the graph contains
only continuous linear operators, and an MILP when exact ReLU or MaxPool2d
formulations introduce binary variables.

NCET creates constraints but does not choose an optimization objective. A user
can combine the returned constraints and neural-network variables with a
larger optimization model and its objective.

## 1. Tensor variables and input bounds

For every entry in `graph.tensors`, NCET creates one continuous CVXPY variable
with the same per-sample shape:

```python
variables[name] = cp.Variable(spec.shape)
```

This includes graph inputs, intermediate tensors, and final output tensors.
Weights, biases, and other fixed arrays in `graph.constants` remain constants
and do not create decision variables.

For each graph input $X$, the supplied box bounds become linear constraints:

$$
L_X \leq X \leq U_X.
$$

The `Output` IR node is only a graph boundary. It introduces no new variable;
`encoding.outputs` returns the existing variables named by `graph.outputs`.

## 2. LP-compatible operator constraints

All affine and structural operators below are represented by linear
equalities and require no binary variables.

### Linear

For a Linear layer with fixed weight $W$ and bias $b$:

$$
Y=XW^\mathsf{T}+b.
$$

PyTorch Linear acts on the last input dimension. NCET temporarily reshapes all
preceding dimensions into rows, applies this equality, and preserves the
original output shape.

### Conv2d

NCET constructs a sparse matrix $A_{\mathrm{conv}}$ containing every valid
kernel-to-input connection. Using C-order vectorization:

$$
\operatorname{vec}_C(Y)
=A_{\mathrm{conv}}\operatorname{vec}_C(X)+\bar b.
$$

$\bar b$ repeats each output-channel bias across its spatial positions.
Kernel positions lying in zero padding are omitted from the sparse matrix.

More details can be found in [conv2d_exact_encoding.md](conv2d_exact_encoding.md).

### BatchNorm

Evaluation-mode BatchNorm is normalized into fixed per-channel scale and shift
vectors $a$ and $d$. NCET broadcasts them to the input shape and enforces

$$
Y=a\odot X+d.
$$

This is an exact affine equality and introduces no binary variables. See
[batchnorm_exact_encoding.md](batchnorm_exact_encoding.md) for the derivation
and supported shapes.

### ElementwiseAffine

Arithmetic between one graph tensor and one fixed constant is normalized to

$$
Y=A\odot X+D.
$$

The fixed scale $A$ and shift $D$ are stored in `graph.constants` and
broadcast to the tensor shape. This exact affine equality covers supported
fixed-constant Add/Sub/Mul/Div forms and introduces no binary variables.

### AveragePool2d

AveragePool2d is a fixed linear map. NCET builds a sparse averaging matrix
$A_{\mathrm{avg}}$ and enforces:

$$
\operatorname{vec}_C(Y)
=A_{\mathrm{avg}}\operatorname{vec}_C(X).
$$

Its coefficients reflect the kernel, stride, padding, and
`count_include_pad` semantics of the canonical operator.

### AdaptiveAvgPool2d

AdaptiveAvgPool2d is also a fixed linear map once its input and output shapes
are known. NCET derives each adaptive window, builds a sparse matrix
$A_{\mathrm{adaptive}}$, and enforces

$$
\operatorname{vec}_C(Y)
=A_{\mathrm{adaptive}}\operatorname{vec}_C(X).
$$

Non-divisible input sizes may produce overlapping windows; the matrix records
each connection exactly. The operation introduces no binary variables.

### Add and Sub

For two input tensors:

$$
Y=X_1+X_2
$$

or

$$
Y=X_1-X_2.
$$

These equalities directly preserve residual and other branch connections.

### ReduceMean

For fixed sample axes $\mathcal D$, let $K$ be the product of their sizes.
NCET enforces

$$
Y=\frac{1}{K}\sum_{d\in\mathcal D}X.
$$

The implementation performs the reductions from the highest axis to the
lowest so axis numbers remain valid when `keepdim=False`. This is one exact
linear equality and introduces no binary variables. See
[reduce_mean_exact_encoding.md](reduce_mean_exact_encoding.md) for details.

### Identity and evaluation-mode Dropout

Identity is represented by

$$
Y=X.
$$

Evaluation-mode Dropout is normalized to the same canonical operation because
its random mask is disabled. This equality introduces no binary variables.

### Concat

For concatenation along dimension $d$, NCET maps each input to its contiguous
slice in the output:

$$
Y[s_k]=X_k,
$$

where $s_k$ is the output slice assigned to input $X_k$ along dimension $d$.

### Shape, axis, and index operations

Flatten and Reshape preserve C-order element positions. View, Squeeze, and
Unsqueeze are normalized to the same Reshape semantics:

$$
\operatorname{vec}_C(Y)=\operatorname{vec}_C(X).
$$

Permute and Transpose enforce the corresponding axis reordering:

$$
Y=\operatorname{permute}(X).
$$

Static GetItem and Slice operations directly select the recorded index:

$$
Y=X[\mathrm{index}].
$$

If a network contains only these operators, its encoding consists entirely of
continuous variables and linear constraints. Combined with a linear
objective, it is an LP.

## 3. Exact ReLU encoding

For one ReLU element

$$
y=\max(0,x), \qquad L\leq x\leq U,
$$

IBP determines one of three cases.

If $L\geq0$, the ReLU is always active:

$$
y=x.
$$

If $U\leq0$, it is always inactive:

$$
y=0.
$$

If $L<0<U$, it is unstable. NCET introduces $z\in\{0,1\}$ and adds:

$$
y\geq x,
$$

$$
y\geq0,
$$

$$
y\leq x-L(1-z),
$$

$$
y\leq Uz.
$$

These constraints describe exactly the graph of ReLU over $[L,U]$.

In `relu_binary_mode="reduced"`, stable elements use the direct equalities and
only unstable elements receive binaries. In `"full"` mode, every ReLU element
uses the binary formulation. Both modes are exact, but reduced mode normally
produces a smaller MILP and is considered as default.

## 4. Exact MaxPool2d encoding

For one output position $p$, let $\mathcal V(p)$ contain its valid input
candidates. MaxPool2d requires:

$$
y_p=\max_{q\in\mathcal V(p)}x_q.
$$

IBP provides sound output bounds $L_{y,p}$ and $U_{y,p}$. NCET explicitly
enforces them:

$$
L_{y,p}\leq y_p\leq U_{y,p}.
$$

NCET creates one binary selector $z_{pq}$ for every valid candidate
connection $q\rightarrow p$ and imposes:

$$
y_p\geq x_q,
\qquad \forall q\in\mathcal V(p),
$$

$$
y_p\leq x_q+
(U_{y,p}-L_{x,q})(1-z_{pq}),
\qquad \forall q\in\mathcal V(p),
$$

$$
\sum_{q\in\mathcal V(p)}z_{pq}=1,
$$

$$
z_{pq}\in\{0,1\}.
$$

The selected candidate has $z_{pq}=1$, forcing $y_p\leq x_q$. Together with
$y_p\geq x_q$ for every candidate, this makes the output equal to a maximum.
Ties remain valid because any tied maximum may be selected. The current
MaxPool2d formulation always uses the full set of candidate binaries. The
explicit output bounds do not change the exact feasible set; they strengthen
the MILP relaxation.

## 5. When the result is an LP or MILP

The encoding is continuous and LP-compatible when it contains no binary
variables. Binary variables arise from:

- every ReLU element in `"full"` mode;
- unstable ReLU elements in `"reduced"` mode;
- every valid MaxPool2d candidate connection.

Consequently, a graph with only affine and structural operators is an LP
constraint system. A reduced-mode graph whose ReLUs are all stable also
remains continuous. A graph with unstable ReLUs or MaxPool2d is encoded as an
MILP.

## 6. Meaning of exactness

Over the supplied input box, the feasible solutions of the NCET constraints
project onto exactly the input, intermediate, and output values permitted by
the supported PyTorch graph. The encoding does not approximate ReLU or
MaxPool2d.

The propagated bounds used in big-M coefficients must be finite and sound.
Loose bounds do not change mathematical exactness, but they weaken the MILP
relaxation and can make the optimization problem substantially slower.

## Related knowledge

- [Knowledge notes index](index.md)
- [Interval Bound Propagation](bound_propagation.md)
- [Exact Conv2d encoding](conv2d_exact_encoding.md)
- [Exact BatchNorm encoding](batchnorm_exact_encoding.md)
- [Exact AdaptiveAvgPool2d encoding](adaptive_avgpool2d_exact_encoding.md)
- [Exact ReduceMean encoding](reduce_mean_exact_encoding.md)
- [Exact AvgPool2d encoding](avgpool2d_exact_encoding.md)
- [Exact MaxPool2d encoding](maxpool2d_exact_encoding.md)
