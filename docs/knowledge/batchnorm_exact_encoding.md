# Exact BatchNorm encoding in NCET

NCET supports `nn.BatchNorm1d` and `nn.BatchNorm2d` as fixed affine operators
for inference. Training-mode BatchNorm is intentionally unsupported because
its mean and variance depend on the current batch rather than fixed model
state.

## 1. Inference formula

For channel $c$, PyTorch evaluation-mode BatchNorm computes

$$
Y_{c,\ldots}
=
\gamma_c\frac{X_{c,\ldots}-\mu_c}
{\sqrt{\sigma_c^2+\epsilon}}+\beta_c,
$$

where $\mu_c$ and $\sigma_c^2$ are the stored running mean and variance. When
`affine=False`, NCET uses $\gamma_c=1$ and $\beta_c=0$.

Define the fixed scale and shift

$$
a_c=\frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}},
\qquad
d_c=\beta_c-a_c\mu_c.
$$

The operation then becomes

$$
Y_{c,\ldots}=a_cX_{c,\ldots}+d_c.
$$

During normalization, NCET stores the read-only vectors $a$ and $d$ in
`GraphIR.constants`; the `BatchNorm` IR node refers to them through its
`scale` and `shift` attributes.

## 2. Supported sample shapes

NCET variables have no batch dimension. The supported per-sample shapes are:

- `BatchNorm1d`: `(C,)` or `(C,L)`;
- `BatchNorm2d`: `(C,H,W)`.

The channel vectors are broadcast across the remaining dimensions. The module
must be in evaluation mode, use fixed running statistics, and have
`num_features == C`.

## 3. Interval bounds

Because $a_c$ can be negative, define

$$
a_c^+=\max(a_c,0),
\qquad
a_c^-=\min(a_c,0).
$$

For input bounds $L_X\leq X\leq U_X$, NCET propagates

$$
L_Y=a^+\odot L_X+a^-\odot U_X+d,
$$

$$
U_Y=a^+\odot U_X+a^-\odot L_X+d.
$$

This gives the exact elementwise range of the BatchNorm operation over the
input box.

## 4. Optimization constraint

The CVXPY backend creates the output tensor and adds the elementwise equality

$$
Y=a\odot X+d,
$$

with $a$ and $d$ broadcast along non-channel axes. BatchNorm therefore adds
one continuous output tensor, no binary variables, and no relaxation.
