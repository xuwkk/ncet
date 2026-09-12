# NCET Developer Documentation

## Overview

This document is a developer reference for NCET's internal objects and data flow. It explains what each object represents, where it is created and how later stages consume it.

It describes the current implementation rather than the complete roadmap.
It is explanatory rather than normative: current operator capability is
defined in [`supported_operators.md`](supported_operators.md), and semantic
exactness is defined in [`exactness_contract.md`](exactness_contract.md).

## 1. Pipeline at a glance

```text
PyTorch nn.Module
    │ capture_graph()
    ▼
torch.fx.GraphModule
    └── fx.Graph
         └── fx.Node
    │ propagate_shapes()
    ▼
FX nodes with shape/dtype in node.meta
    └── temporary leading batch dimension of size 1
    │ normalize_graph()
    ▼
GraphIR
    ├── IRNode: canonical operations
    ├── TensorSpec: static tensor metadata of the model
    └── constants: fixed parameters and buffers
    │ propagate_bounds()
    ▼
dict[tensor_name, Bounds]
    │ encode_cvxpy()
    ▼
MILPEncoding
    ├── CVXPY variables and constraints
    ├── graph inputs and outputs
    ├── ReLU and MaxPool binary variable summary
    └── formulation statistics
```

The high-level `form_milp()` builder performs the entire pipeline.

Each of the stages are introduced in the following sections.

### 1.1 Key objects and their lifecycle

| Object | Created by | Main consumers | Purpose |
|---|---|---|---|
| `fx.GraphModule` | `capture_graph()` | `propagate_shapes()`, `normalize_graph()` | Executable PyTorch graph captured from the model |
| FX `tensor_meta` | `propagate_shapes()` | `normalize_graph()` | Batched shape and dtype metadata of all tensors in the graph |
| `FXNodeInfo` | `describe_graph()` | Developer diagnostics only | Print entries in the FX node metadata dictionary |
| `GraphIR` | `normalize_graph()` | `validate_ir()`, `propagate_bounds()`, `encode_cvxpy()` | NCET's structured representation of the model's computation graph |
| `dict[str, Bounds]` | `propagate_bounds()` | `encode_cvxpy()` | Elementwise interval bounds for every graph tensor |
| `MILPEncoding` | `encode_cvxpy()` | The calling optimization model | Contains CVXPY variables, constraints, metadata, and statistics |

**Distinctions**: The GraphModule is a PyTorch representation of the model's computation graph; GraphIR is a more structured representation by NCET.

### 1.2 High-level builder control flow

Internally, `form_milp()` performs the following operations in order:

```python
# 1. Capture the PyTorch graph
traced = capture_graph(model)
# 2. Obtain the list of standardized bounds of the input tensors
input_names = [
    node.name for node in traced.graph.nodes if node.op == "placeholder"
]
ordered_bounds = _bind_input_bounds(input_names, input_bounds)
# 3. Obtain the shape of each tensor
example_inputs = _example_inputs(model, ordered_bounds)
propagate_shapes(traced, *example_inputs)
# 4. Obtain the GraphIR representation
graph = normalize_graph(traced)
# 5. Propagate the bounds of each tensor
bound_mapping = dict(zip(graph.inputs, ordered_bounds))
bounds = propagate_bounds(graph, bound_mapping)
# 6. Convert the public mode into the backend's internal options
options = EncodingOptions(relu_binary_mode=relu_binary_mode)
# 7. Encode the CVXPY model
encoding = encode_cvxpy(graph, bounds, options)
```

The representative inputs `example_inputs` are zero tensors whose shapes come from the public input bounds. Their dtype and device come from the model's first parameter, or from PyTorch's default dtype on CPU for a parameter-free model. 


### 1.3 Shape and batch convention

The normative user-facing batch requirement is part of the
[`exactness contract`](exactness_contract.md). This section shows how that
requirement is represented internally.

NCET asks the user to provide data (usually when defining the input bounds) **without** a batch dimension. It also constructs the CVXPY variables and constraints **without** a batch dimension.

The leading singleton dimension exists only while PyTorch executes the model
for FX shape propagation. `normalize_graph()` removes it and converts
dimension-valued operator attributes to per-sample coordinates. 

> When defining the NN model, the user must keep batch axis 0 unchanged and compute each sample independently.

| Object | Example image shape |
|---|---|
| Public `Bounds` (user-provided) | `(C, H, W)` |
| FX representative input | `(1, C, H, W)` |
| Raw FX `tensor_meta.shape` | `(1, C, H, W)` |
| GraphIR `TensorSpec.shape` | `(C, H, W)` |
| Propagated `Bounds` | `(C, H, W)` |
| CVXPY expression | `(C, H, W)` |

## 2. PyTorch FX stage

### 2.1 `fx.GraphModule`

`capture_graph(model)` returns a `torch.fx.GraphModule`. It is both:

- an executable `nn.Module`, so `traced(x)` runs the generated `forward()`;
- a container for the symbolic computation graph.

`capture_graph()` requires `model.training == False`. It also rejects captured
in-place mutation before returning the graph.

Common entries:

| Entry | Meaning |
|---|---|
| `graph_module.graph` | The `fx.Graph` containing nodes |
| `graph_module.code` | Generated Python source for `forward()` |
| `graph_module.forward()` | Executable generated forward method |
| `graph_module.get_submodule(path)` | Locates the actual layer using its module path and returns a reference to that layer |
| `graph_module.named_modules()` | Modules retained in the GraphModule |
| `graph_module.state_dict()` | Parameters and buffers retained by the graph |

### 2.2 `fx.Node`

Iterate through FX nodes with:

```python
for node in traced.graph.nodes:
    ...
```

Common node entries:

| Entry | Type | Meaning |
|---|---|---|
| `node.name` | `str` | Unique node name; also its output-value name when the node produces a value |
| `node.op` | `str` | FX operation category |
| `node.target` | varies | Identifies what the FX node refers to or invokes; its exact meaning depends on node.op |
| `node.args` | `tuple` | Contains producer nodes and/or static arguments; producer references are `Node` objects |
| `node.kwargs` | mapping | Keyword arguments |
| `node.users` | `dict[fx.Node, None]` | An ordered set-like mapping whose keys are the unique consumer nodes that reference this node in their args or kwargs |
| `node.all_input_nodes` | `list[Node]` | All producer nodes found recursively in args/kwargs |
| `node.meta` | `dict` | Optional metadata such as shape and module provenance (The location in the original PyTorch model where an FX operation came from) |

Important: `node.args` does not normally store producer names as strings. It
stores producer `Node` objects. NCET later converts those references into tensor
names.


#### 2.2.1 `op` and `target`

- `op`: Operation category;
- `target`: Tells FX what to execute, retrieve, or represent;
- `node.meta["nn_module_stack"]`: Tells developers which original module scope produced the operation.

Only call_module and get_attr targets are paths. A call_function target is a callable object, while a call_method target is a method-name string.

Consider the following example:
```python
import operator

import torch
from torch import fx, nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        # Used by call_module.
        self.linear = nn.Linear(4, 4)
        # Accessing this buffer produces get_attr.
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x):
        y = self.linear(x)      # call_module
        y = y + self.offset     # get_attr + call_function
        y = y.reshape(-1, 2, 2) # call_method
        return torch.relu(y)    # call_function

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block(x)
```

The graph should have approximately these nodes:
| `node.op` | Typical `node.target` | What `target` identifies | Typical `module_path` from `nn_module_stack` |
|---|---|---|---|
| `placeholder` | `"x"` | The `forward()` argument named `x` | `None` |
| `get_attr` | `"block.offset"` | The registered buffer at `graph_module.block.offset` | `"block"` or `None` |
| `call_module` | `"block.linear"` | The registered layer returned by `graph_module.get_submodule("block.linear")` | `"block.linear"` |
| `call_function` | `operator.add` | The actual function used to compute `y + offset` | `"block"` |
| `call_function` | `torch.relu` | The actual `torch.relu` function object | `"block"` |
| `call_method` | `"reshape"` | The method invoked as `y.reshape(...)` | `"block"` |
| `output` | `"output"` | The FX graph return boundary | `None` |

> This table describes the general FX node categories, not NCET's supported
operator set. In particular, the current normalizer does not canonicalize an
arbitrary standalone `get_attr` node. Supported `call_module` state for
Linear, Conv2d, and BatchNorm are lifted directly into `GraphIR.constants`.

> The raw `nn_module_stack` can contain several nested modules. NCET's `FXNodeInfo.module_path` keeps only its innermost module path. Provenance is optional, so `module_path` can be `None`.

For the complete list of PyTorch/FX spellings that NCET recognizes, see
[`pytorch_to_ir_operator_mapping.md`](pytorch_to_ir_operator_mapping.md).

#### 2.2.2 `meta`

After:

```python
propagate_shapes(traced, *example_inputs)
```

tensor-producing nodes normally contain:

```python
tensor_meta = node.meta["tensor_meta"]
tensor_meta.shape
tensor_meta.dtype
```

Shape propagation uses real example tensors to execute the graph, but these
values are used to determine metadata rather than input bounds. The high-level
builder constructs these tensors by adding a leading batch dimension of size 1
to each public per-sample input shape. Raw FX metadata therefore includes this
temporary dimension.

### 2.3 `FXNodeInfo`

`describe_graph(traced)` converts raw FX nodes into immutable, readable
`FXNodeInfo` objects. This is for diagnostic purposes only.

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Unique FX node name; also the output-value name for value-producing nodes |
| `op` | `str` | FX operation category |
| `target` | `Any` | Raw FX target |
| `args` | `tuple[Any, ...]` | Raw positional arguments (producer nodes and/or static arguments) |
| `kwargs` | `Mapping[str, Any]` | Raw keyword arguments |
| `users` | `tuple[str, ...]` | Consumer node names |
| `output_shape` | `tuple[int, ...] \| None` | Shape from metadata |
| `output_dtype` | `torch.dtype \| None` | Dtype from metadata |
| `module_path` | `str \| None` | Derived source-module provenance |


## 3. Canonical IR stage

### 3.1 `GraphIR`

`GraphIR` is NCET's framework-independent, static representation of a neural network, organizing canonical operations, runtime tensors, graph boundaries, and fixed parameters in one object. It records the static structure of the neural network and all the parameters that are used to formulate the optimization problem. Unlike raw FX metadata, GraphIR represents one sample, so its tensor shapes and dimension-valued attributes do not contain the temporary batch axis.

```python
@dataclass
class GraphIR:
    nodes: list[IRNode]
    inputs: list[str]
    outputs: list[str]
    tensors: dict[str, TensorSpec]
    constants: dict[str, np.ndarray]
```

| Field | Contents |
|---|---|
| `nodes` | Canonical operations in topological order |
| `inputs` | Tensor names at the graph input boundary |
| `outputs` | Tensor names at the graph output boundary |
| `tensors` | Static metadata for every runtime graph tensor |
| `constants` | Read-only weights, biases, and buffers |

> `graph.inputs` and `graph.outputs` contain tensor names, not node objects.

### 3.2 `IRNode`

```python
@dataclass(frozen=True)
class IRNode:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: Mapping[str, Any]
```

| Field | Meaning |
|---|---|
| `name` | Unique operation/call name |
| `op_type` | NCET canonical operator such as `Linear` or `Add` |
| `inputs` | Names of tensors consumed by the operation |
| `outputs` | Names of tensors produced by the operation |
| `attrs` | attrs stores the operator’s fixed configuration. Small values, such as dimensions and strides, are stored directly, while large parameter arrays, such as weights and biases, are stored once in graph.constants and referenced by name. |

An `Input` node has no inputs and one output. An `Output` node consumes one or
more existing tensors and produces no tensor.

FX `target` and the derived `module_path` are not copied into `IRNode.attrs` in
the current implementation. After normalization, use `op_type` for operation
semantics and operator-specific `attrs` for formulation data.


### 3.3 `TensorSpec`

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    producer: str | None
```

| Field | Meaning |
|---|---|
| `name` | Unique tensor identifier, normally the same as the `IRNode.name` that produces the tensor |
| `shape` | Static per-sample tensor shape; no batch dimension |
| `dtype` | Dtype string such as `"float32"` |
| `producer` | Name of the producing IR node, or `None` for external input |

`TensorSpec` does not contain tensor values, bounds, or CVXPY variables.
The current implementation records `dtype` as metadata but does not yet use it
to select a CVXPY variable type or reject unsupported dtype combinations.

For an input named `x`, GraphIR contains both an `Input` operation and a
`TensorSpec("x", ..., producer=None)`. The operation marks the graph boundary;
`producer=None` states that `x` is supplied externally rather than computed by
another neural-network operation.

### 3.4 Tensor naming

NCET currently reuses `fx.Node.name` for both:

- the normalized `IRNode.name`;
- the tensor produced by that node.

For example, operation node `linear` produces tensor `linear`. They share a
string but represent different concepts. Repeated FX names receive suffixes,
such as `linear`, `linear_1`, and `linear_2`.

### 3.5 Canonical operator attributes

| `op_type` | `attrs` entries |
|---|---|
| `Input` | `{}` |
| `Linear` | `weight`, `bias`: names in `graph.constants` |
| `Conv2d` | `weight`, `bias`, `stride`, `padding`, `dilation`, `groups` |
| `BatchNorm` | `scale`, `shift`: names in `graph.constants` |
| `AdaptiveAvgPool2d` | resolved per-sample `output_size` |
| `AvgPool2d` | `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad`, `divisor_override` |
| `MaxPool2d` | `kernel_size`, `stride`, `padding`, `dilation`, `ceil_mode`, `return_indices` |
| `Identity` | `{}`; also represents evaluation-mode Dropout |
| `ReLU` | `{}` |
| `Add`, `Sub` | `alpha=1` |
| `Concat` | `dim` |
| `Flatten` | `start_dim`, `end_dim` |
| `Reshape` | resolved output `shape`; also represents View and batch-preserving Squeeze/Unsqueeze |
| `Permute` | complete `dims` permutation |
| `Transpose` | `dim0`, `dim1` |
| `GetItem`, `Slice` | canonical `index` |
| `Output` | `{}` |

Dimension-valued attributes use GraphIR's per-sample coordinates. For example,
PyTorch `torch.cat((x, y), dim=1)` becomes `Concat(dim=0)` after the leading
batch axis is removed. Normalization rejects operations that modify batch axis
0 rather than interpreting them heuristically.

Constant indirection example:

```python
weight_name = linear_node.attrs["weight"]
weight_value = graph.constants[weight_name]
```

Canonical index actions are:

```text
("index", value)
("slice", start, stop, step)
("newaxis",)
```

### 3.6 Constants and shared module state

Linear, Conv2d, and BatchNorm nodes do not copy fixed arrays into
`IRNode.attrs`. Instead, attrs contain names that refer into the graph-level
constant table:

```text
IRNode.attrs["weight"]
          │
          ▼
"linear.weight"
          │
          ▼
GraphIR.constants["linear.weight"]
          │
          ▼
read-only NumPy array
```

Normalization copies direct parameters and buffers from the targeted PyTorch
module into read-only NumPy arrays. BatchNorm running statistics and affine
parameters are first combined into per-channel `scale` and `shift` arrays.
When a shared module is called more than once, several IR nodes refer to the
same constant names, so its parameters are stored once. If a supported Linear
or Conv2d module has no bias, normalization creates one fixed zero-bias array
so later passes can use a uniform affine formulation.

Constants are not included in `graph.tensors`, do not receive propagated
`Bounds`, and do not create CVXPY decision variables.

### 3.7 How FX edges become IR tensor inputs

An FX node can reference producer nodes anywhere inside nested positional or keyword arguments. `fx.map_arg()` recursively finds all producer Node references within the current FX node’s nested args and kwargs. NCET converts these references into tensor names and stores them in `IRNode.inputs`, thereby preserving the graph’s actual connectivity.

```python
names = []
fx.map_arg(
    (node.args, node.kwargs),  # Obtain producer nodes in nested args and kwargs
    lambda input_node: names.append(input_node.name),  # Convert producer Node references into tensor names
)
input_names = tuple(names)
```

The resulting names become `IRNode.inputs`. For example:

```text
operator.add(relu_node, x_node)       → ("relu", "x")
torch.cat((left_node, right_node), 1) → ("left", "right")
```

This recursive conversion preserves branches, fan-out, residual shortcuts, nested input containers, and repeated uses of the same tensor. Non-node arguments such as dimensions and flags are not tensor inputs and will not be returned by `fx.map_arg()`; normalization stores their canonical static meanings in `IRNode.attrs` when needed.

### 3.8 Graph validation and capability checks

`normalize_graph()` calls `validate_ir()` before returning. The validator checks
high-risk structural invariants:

- node names are unique;
- every consumed tensor has already been produced;
- a tensor has at most one producer;
- every produced tensor has a `TensorSpec` with a static non-negative shape;
- graph input/output boundaries reference available tensors;
- Linear and Conv2d weight/bias references and BatchNorm scale/shift references
  exist in `graph.constants`.

`analyze_capabilities(graph, supported_ops)` is a separate consumer-specific
check. It counts canonical operators and returns the nodes that a particular
pass or backend does not support. Bound propagation and CVXPY encoding each
apply this check using their own supported-operator sets before processing the
graph.


## 4. Bounds stage

### 4.1 `Bounds`

```python
@dataclass(frozen=True)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray
```

Both arrays must match the corresponding `TensorSpec.shape`, contain finite
values, and satisfy `lower <= upper` elementwise. They describe one sample and
do not contain a batch dimension.

### 4.2 Bounds dictionary

The low-level API returns:

```python
bounds: dict[str, Bounds]
```

Each key is a tensor name. Every graph tensor receives one entry:

```python
bounds["x"]
bounds["linear"]
bounds["relu"]
bounds["add"]
```

When processing an IR node, NCET reads existing input entries and creates an
entry for its output tensor:

```python
input_bounds = tuple(bounds[name] for name in node.inputs)
bounds[node.outputs[0]] = output_bounds
```

An input tensor reused by a residual connection refers to the same dictionary
entry; it is not duplicated.

### 4.3 Topological propagation and validation

`propagate_bounds()` first requires exactly one supplied `Bounds` entry for
every name in `graph.inputs`, with no unknown names. Each lower/upper pair must:

- have the corresponding `TensorSpec.shape`;
- contain only finite values;
- satisfy `lower <= upper` elementwise.

The pass then visits `graph.nodes` in topological order. `Input` and `Output`
nodes require no bound transformation. Every computational node reads the
already available bounds named by `node.inputs`, applies its *canonical interval
rule*, validates the result against the output `TensorSpec`, and stores it under
`node.outputs[0]` ([0] means that currently all the supported operations have only one output).

> Interval propagation is sound but can be conservative because it does not keep
> correlations between values on different branches. These bounds are later used
> as valid big-M values for exact ReLU and MaxPool formulations.

## 5. CVXPY encoding stage

`encode_cvxpy()` first creates one continuous `cp.Variable` for every entry in `graph.tensors`. It then processes `graph.nodes` in topological order and adds the exact constraints for each canonical operation:

```python
variables = {
    name: cp.Variable(spec.shape)
    for name, spec in graph.tensors.items()
}

for node in graph.nodes:
    # Dispatch on node.op_type and append its exact constraints.
    ...
```

Input nodes add lower/upper constraints to their existing graph variables.
Affine, branching, pooling, shape, and index nodes link their input and output variables with equalities or exact mixed-integer constraints. ReLU and MaxPool may additionally create binary variables. The Output node creates no new value; `graph.outputs` selects existing graph variables for `MILPEncoding.outputs`.

### 5.1 `MILPEncoding`

`MILPEncoding` is the output of the CVXPY encoding stage.

```python
@dataclass
class MILPEncoding:
    constraints: list[cp.Constraint]
    inputs: dict[str, cp.Expression]
    outputs: list[cp.Expression]
    values: dict[str, EncodedTensor]
    binaries: dict[str, ReLUBinaries | MaxPoolBinaries]
    graph: GraphIR
    stats: EncodingStats
```

| Field | Key/index | Contents |
|---|---|---|
| `constraints` | list index | Generated exact constraints |
| `inputs` | input tensor name | CVXPY input variable (input of the NN and constraints) |
| `outputs` | output position | CVXPY output variable (output of the NN and constraints) |
| `values` | any tensor name | `EncodedTensor` for that value |
| `binaries` | ReLU or MaxPool node name | Operator-specific binary information |
| `graph` | — | The canonical `GraphIR` |
| `stats` | — | Formulation counts |

`outputs` is a list because a model may return multiple values. `inputs` is a
dictionary because external optimization expressions are usually linked by
input name.

Multiple model return values are supported through multiple names in
`graph.outputs`. This is different from one canonical operator producing
several tensor values: current computational encoders read `node.outputs[0]`, so general multi-output operators are not yet supported.

NCET creates one `cp.Variable` for each graph tensor. `values` wraps this full variable set with bounds and shapes, while `inputs` and `outputs` reference the same underlying variables at the two graph boundaries; they do not create duplicate decision variables. `binaries` contains the separate ReLU activation and MaxPool selection variables required by the MILP formulation.

### 5.2 Builder input types

```python
form_milp(model, input_bounds, relu_binary_mode="reduced")
```

#### Bound types

The public builder accepts:

```python
BoundPair = tuple[Any, Any]
BoundLike = Bounds | BoundPair
InputBounds = BoundLike | Sequence[Bounds] | Mapping[str, BoundLike]
```

| Model inputs | Accepted form | Example |
|---|---|---|
| One | `Bounds` | `Bounds(lower, upper)` |
| One | `(lower, upper)` | `(lower, upper)` |
| Multiple, positional | `list[Bounds]` | `[x_bounds, y_bounds]` |
| One or multiple, named | Mapping | `{"x": x_bounds, "y": y_bounds}` |


Positional bounds must follow the same order as the model's `forward()`
arguments. Named mappings avoid this ambiguity and are
recommended when a model has several inputs and the keys should be the same as the model's `forward()` arguments.

Every bound uses the per-sample input shape. For example, an MLP input with two
features uses shape `(2,)`, not `(1, 2)`, and a single image uses `(C, H, W)`,
not `(1, C, H, W)`. The builder adds the singleton batch dimension only for FX
shape propagation and removes it after shape propagation.

#### ReLU binary mode

| Mode | Meaning |
|---|---|
| `"reduced"` | Binary variables only for unstable ReLU elements |
| `"full"` | Binary variables for every ReLU element |

Both modes are exact.

The option affects ReLU only. MaxPool2d currently always uses its full exact
one-hot formulation, with one selector for every valid candidate in every
pooling window.

### 5.3 `EncodedTensor`

```python
@dataclass(frozen=True)
class EncodedTensor:
    expression: cp.Expression
    bounds: Bounds
    shape: tuple[int, ...]
```

This joins the optimization representation with the static interval and shape
of one per-sample graph tensor. Its shape does not contain a batch dimension.


### 5.4 `ReLUBinaries`

```python
@dataclass(frozen=True)
class ReLUBinaries:
    variable: cp.Variable  # Flattened binary selector vector.
    flat_indices: np.ndarray
    original_tensor_shape: tuple[int, ...]
```

`variable` is always a one-dimensional binary vector. `flat_indices` is its
address table: `variable[k]` controls the ReLU element at C-order flat position
`flat_indices[k]`. `original_tensor_shape` records the shape before flattening
so that a flat position can be converted back to its tensor coordinate. It is
metadata, not the shape of `variable`.

For example, consider a ReLU tensor with shape `(2, 3)`:

```text
tensor coordinate:  (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
C-order flat index:    0     1     2     3     4     5
```

If only positions `1` and `4` are unstable, reduced mode stores:

```python
ReLUBinaries(
    variable=cp.Variable(2, boolean=True),
    flat_indices=np.array([1, 4]),
    original_tensor_shape=(2, 3),
)
```

The mapping is:

```text
variable[0] -> flat index 1 -> coordinate (0, 1)
variable[1] -> flat index 4 -> coordinate (1, 1)
```

The original coordinates can be recovered with:

```python
np.unravel_index(flat_indices, original_tensor_shape)
```

The ReLU formulation uses the same mapping when selecting the continuous
input and output elements:

```python
x_binary = x[flat_indices]
y_binary = y[flat_indices]
```

Thus `variable[k]` controls `y_binary[k] = ReLU(x_binary[k])`. The invariants
are:

```python
variable.size == len(flat_indices)
```

In reduced mode, this size is the number of unstable elements and can be
smaller than `np.prod(original_tensor_shape)`. In full mode, `flat_indices` is
`[0, 1, ..., tensor_size - 1]`, so the element counts are equal.

### 5.5 `MaxPoolBinaries`

```python
@dataclass(frozen=True)
class MaxPoolBinaries:
    variable: cp.Variable
    output_indices: np.ndarray
    input_indices: np.ndarray
```

- `input_indices[k]` is the coordinate of the $k$-th valid input candidate in
  a pooling window.
- `output_indices[k]` is the coordinate of the output element associated with
  that candidate. All candidates in one pooling window share the same output
  coordinate.
- `variable[k]` indicates whether that input candidate is selected as the
  maximum for the associated output element.

The three arrays are aligned by position:

```text
variable[k]: input_indices[k] -> output_indices[k]
```

For example, consider a per-sample input with shape `(1, 2, 3)`:

```text
x00  x01  x02
x10  x11  x12
```

With `kernel_size=(2, 2)`, `stride=(1, 1)`, and no padding, MaxPool2d
creates two overlapping windows:

```text
window 0: x00, x01, x10, x11 -> y00
window 1: x01, x02, x11, x12 -> y01
```

Each of the eight candidate connections receives one binary selector:

```python
output_indices = np.array([
    [0, 0, 0],  # z[0]: x00 -> y00
    [0, 0, 0],  # z[1]: x01 -> y00
    [0, 0, 0],  # z[2]: x10 -> y00
    [0, 0, 0],  # z[3]: x11 -> y00
    [0, 0, 1],  # z[4]: x01 -> y01
    [0, 0, 1],  # z[5]: x02 -> y01
    [0, 0, 1],  # z[6]: x11 -> y01
    [0, 0, 1],  # z[7]: x12 -> y01
])

input_indices = np.array([
    [0, 0, 0],  # x00
    [0, 0, 1],  # x01
    [0, 1, 0],  # x10
    [0, 1, 1],  # x11
    [0, 0, 1],  # x01
    [0, 0, 2],  # x02
    [0, 1, 1],  # x11
    [0, 1, 2],  # x12
])

variable = cp.Variable(8, boolean=True)
```

The formulation adds one one-hot constraint per output window:

$$
z_0+z_1+z_2+z_3=1,
$$

$$
z_4+z_5+z_6+z_7=1.
$$

The overlapping input positions `x01` and `x11` appear once for each output
window and therefore receive separate selectors. Consequently, the number of
MaxPool binaries equals the number of valid candidate connections, not the
number of unique input elements:

```python
variable.size == len(input_indices) == len(output_indices)
```

### 5.6 `EncodingStats`

| Field | Meaning |
|---|---|
| `continuous_variables` | Total scalar graph-tensor variables |
| `binary_variables` | Total scalar ReLU and MaxPool binaries |
| `constraints` | Number of CVXPY constraint objects |
| `always_active_relu` | ReLU elements with lower bound at least zero |
| `always_inactive_relu` | ReLU elements with upper bound at most zero |
| `unstable_relu` | ReLU elements whose interval crosses zero |


## 6. Residual example across all stages

Consider:

```python
def forward(self, x):
    linear = self.linear(x)
    relu = torch.relu(linear)
    return relu + x
```

### 6.1 FX nodes

| `name` | `op` | `target` | Producer inputs |
|---|---|---|---|
| `x` | `placeholder` | `x` | — |
| `linear` | `call_module` | `linear` | `x` |
| `relu` | `call_function` | `torch.relu` | `linear` |
| `add` | `call_function` | `operator.add` | `relu`, `x` |
| `output` | `output` | `output` | `add` |

### 6.2 IR nodes

| `name` | `op_type` | `inputs` | `outputs` |
|---|---|---|---|
| `x` | `Input` | `()` | `("x",)` |
| `linear` | `Linear` | `("x",)` | `("linear",)` |
| `relu` | `ReLU` | `("linear",)` | `("relu",)` |
| `add` | `Add` | `("relu", "x")` | `("add",)` |
| `output` | `Output` | `("add",)` | `()` |

### 6.3 Graph dictionaries

```python
graph.inputs == ["x"]
graph.outputs == ["add"]

graph.tensors.keys() == {"x", "linear", "relu", "add"}
graph.constants.keys() == {"linear.weight", "linear.bias"}

bounds.keys() == {"x", "linear", "relu", "add"}
encoding.values.keys() == {"x", "linear", "relu", "add"}
encoding.inputs.keys() == {"x"}
```

If the ReLU is unstable, `encoding.binaries` also contains:

```python
encoding.binaries["relu"]
```

## 7. Where to look up a piece of information

| Question | Lookup |
|---|---|
| What did FX capture? | `traced.graph.nodes` |
| What does an FX node call? | `node.op`, `node.target` |
| Who produces/consumes an FX value? | `node.all_input_nodes`, `node.users` |
| What is its FX shape/dtype? | `node.meta["tensor_meta"]` |
| What canonical operation is it? | `IRNode.op_type` |
| What tensors does the operation consume? | `IRNode.inputs` |
| What tensor does it produce? | `IRNode.outputs` |
| What is a tensor's static specification? | `graph.tensors[name]` |
| Where are weight and bias names? | `node.attrs["weight"]`, `node.attrs["bias"]` |
| Where are weight and bias values? | `graph.constants[constant_name]` |
| What are a tensor's bounds? | `bounds[name]` or `encoding.values[name].bounds` |
| What is its CVXPY variable? | `encoding.values[name].expression` |
| What is an external graph input in dictionary format? | `encoding.inputs[name]` |
| What are graph outputs in list format? | `encoding.outputs` |
| Which binaries belong to a ReLU or MaxPool? | `encoding.binaries[node_name]` |
| How large is the formulation? | `encoding.stats` |


## 8. Useful inspection snippets

FX:

```python
for node in traced.graph.nodes:
    print(
        node.name,
        node.op,
        node.target,
        [producer.name for producer in node.all_input_nodes],
        [consumer.name for consumer in node.users],
    )
```

IR:

```python
for node in graph.nodes:
    print(node.name, node.op_type, node.inputs, node.outputs, node.attrs)
```

Tensors and bounds:

```python
for name, spec in graph.tensors.items():
    print(name, spec.shape, spec.dtype, spec.producer, bounds[name])
```

Encoding:

```python
print(encoding.inputs)
print(encoding.outputs)
print(encoding.binaries)
print(encoding.stats)
```

## 9. Where capability is defined

This developer reference does not maintain a second operator list. The
authoritative canonical operator and parameter boundary is
[`supported_operators.md`](supported_operators.md), while accepted PyTorch/FX
spellings are maintained in
[`pytorch_to_ir_operator_mapping.md`](pytorch_to_ir_operator_mapping.md).

Internally, support crosses several stages:

```text
recognized FX spelling
        -> canonical GraphIR operation
        -> sound bound propagation
        -> exact CVXPY formulation
```

Successful FX capture alone therefore does not imply NCET support. The
frontend, bounds pass, and backend each reject operations outside their own
consumer capability before a complete `MILPEncoding` is returned.

## 10. Source-file navigation

| Internal logic | Source file |
|---|---|
| FX capture, shape propagation, inspection | [`src/ncet/frontend/fx.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/frontend/fx.py) |
| FX-to-GraphIR normalization | [`src/ncet/frontend/normalize.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/frontend/normalize.py) |
| `GraphIR`, `IRNode`, `TensorSpec` | [`src/ncet/ir/graph.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/ir/graph.py) |
| IR validation | [`src/ncet/ir/validate.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/ir/validate.py) |
| Consumer capability reports | [`src/ncet/ir/capability.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/ir/capability.py) |
| Canonical static indexing | [`src/ncet/ir/indexing.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/ir/indexing.py) |
| Interval bound propagation | [`src/ncet/passes/bounds_ibp.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/passes/bounds_ibp.py) |
| Exact CVXPY formulations | [`src/ncet/backend/cvxpy.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/backend/cvxpy.py) |
| Public LAPSO-style builder | [`src/ncet/builder.py`](https://github.com/xuwkk/ncet/blob/main/src/ncet/builder.py) |

## 11. Related documentation

- [`exactness_contract.md`](exactness_contract.md): model assumptions and the exactness guarantee;
- [`supported_operators.md`](supported_operators.md): authoritative current operator and parameter boundary;
- [`knowledge/index.md`](knowledge/index.md): mathematical knowledge notes and recommended reading order;
- [`knowledge/conv2d_exact_encoding.md`](knowledge/conv2d_exact_encoding.md): sparse Conv2d affine formulation;
- [`knowledge/batchnorm_exact_encoding.md`](knowledge/batchnorm_exact_encoding.md): inference-mode BatchNorm normalization and encoding;
- [`knowledge/adaptive_avgpool2d_exact_encoding.md`](knowledge/adaptive_avgpool2d_exact_encoding.md): adaptive pooling windows and sparse formulation;
- [`knowledge/avgpool2d_exact_encoding.md`](knowledge/avgpool2d_exact_encoding.md): AvgPool2d matrix formulation;
- [`knowledge/maxpool2d_exact_encoding.md`](knowledge/maxpool2d_exact_encoding.md): exact MaxPool2d selection formulation;
- [Representative operator notebook](https://github.com/xuwkk/ncet/blob/main/examples/artificial_test_on_operators.ipynb): executable graph and exact-encoding walkthrough.
