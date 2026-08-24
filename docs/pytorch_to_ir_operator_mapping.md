# PyTorch-to-GraphIR operator mapping

This reference lists the PyTorch spellings that NCET's current FX frontend
projects onto canonical `IRNode.op_type` values. Canonicalization lets later
passes implement one mathematical operator without depending on whether the
model used a module, function, or tensor-method spelling.

For example:

```text
nn.ReLU / F.relu / torch.relu / Tensor.relu -> IRNode(op_type="ReLU")
```

## 1. Graph boundaries

| PyTorch/FX source | `node.op` | `node.target` | `IRNode.op_type` |
|---|---|---|---|
| `forward()` argument | `placeholder` | Argument name, such as `"x"` | `Input` |
| Returned graph value | `output` | `"output"` | `Output` |

An `Input` node produces a graph-boundary tensor. An `Output` node consumes
one or more existing tensors and produces no new tensor.

## 2. Module calls

For `call_module`, `node.target` is the dotted path of a registered submodule.
NCET retrieves the module with `graph_module.get_submodule(node.target)` and
uses its concrete type to choose the canonical operator.

| PyTorch module type | `node.op` | Example `node.target` | `IRNode.op_type` |
|---|---|---|---|
| `nn.Linear` | `call_module` | `"block.linear"` | `Linear` |
| `nn.Conv2d` | `call_module` | `"features.conv"` | `Conv2d` |
| `nn.AvgPool2d` | `call_module` | `"avg_pool"` | `AvgPool2d` |
| `nn.MaxPool2d` | `call_module` | `"max_pool"` | `MaxPool2d` |
| `nn.ReLU` | `call_module` | `"relu"` | `ReLU` |
| `nn.Flatten` | `call_module` | `"flatten"` | `Flatten` |

Example:

```python
self.linear = nn.Linear(4, 3)
y = self.linear(x)
```

```text
FX: call_module[target="linear"]
IR: IRNode(op_type="Linear")
```

## 3. Function calls

For `call_function`, `node.target` is the actual Python or PyTorch callable
recorded by FX.

| PyTorch spelling | `node.op` | `node.target` | `IRNode.op_type` |
|---|---|---|---|
| `x + y` | `call_function` | `operator.add` | `Add` |
| `torch.add(x, y)` | `call_function` | `torch.add` | `Add` |
| `x - y` | `call_function` | `operator.sub` | `Sub` |
| `torch.sub(x, y)` | `call_function` | `torch.sub` | `Sub` |
| `torch.subtract(x, y)` | `call_function` | `torch.subtract` | `Sub` |
| `F.relu(x)` | `call_function` | `torch.nn.functional.relu` | `ReLU` |
| `torch.relu(x)` | `call_function` | `torch.relu` | `ReLU` |
| `F.avg_pool2d(x, ...)` | `call_function` | `torch.nn.functional.avg_pool2d` | `AvgPool2d` |
| `F.max_pool2d(x, ...)` | `call_function` | `torch.nn.functional.max_pool2d` | `MaxPool2d` |
| `torch.cat((x, y), dim)` | `call_function` | `torch.cat` | `Concat` |
| `torch.concat((x, y), dim)` | `call_function` | `torch.concat` | `Concat` |
| `torch.concatenate((x, y), dim)` | `call_function` | `torch.concatenate` | `Concat` |
| `torch.flatten(x, ...)` | `call_function` | `torch.flatten` | `Flatten` |
| `torch.reshape(x, shape)` | `call_function` | `torch.reshape` | `Reshape` |
| `torch.permute(x, dims)` | `call_function` | `torch.permute` | `Permute` |
| `torch.transpose(x, dim0, dim1)` | `call_function` | `torch.transpose` | `Transpose` |
| `x[index]` | `call_function` | `operator.getitem` | `GetItem` or `Slice` |

## 4. Tensor method calls

For `call_method`, `node.target` is the method-name string invoked on the
first tensor argument.

| PyTorch spelling | `node.op` | `node.target` | `IRNode.op_type` |
|---|---|---|---|
| `x.add(y)` | `call_method` | `"add"` | `Add` |
| `x.sub(y)` | `call_method` | `"sub"` | `Sub` |
| `x.subtract(y)` | `call_method` | `"subtract"` | `Sub` |
| `x.relu()` | `call_method` | `"relu"` | `ReLU` |
| `x.flatten(...)` | `call_method` | `"flatten"` | `Flatten` |
| `x.reshape(...)` | `call_method` | `"reshape"` | `Reshape` |
| `x.view(...)` | `call_method` | `"view"` | `Reshape` |
| `x.permute(...)` | `call_method` | `"permute"` | `Permute` |
| `x.transpose(...)` | `call_method` | `"transpose"` | `Transpose` |

## 5. Static indexing classification

FX represents Python tensor indexing as `call_function[operator.getitem]`.
NCET then examines the static index and selects one of two canonical operators:

| Canonical index semantics | `IRNode.op_type` |
|---|---|
| Only integer selection remains at the per-sample level | `GetItem` |
| Any slice, ellipsis, or inserted axis remains | `Slice` |

The temporary tracing batch dimension must remain unchanged. For example, an
FX-level batched index `x[:, 0]` becomes per-sample `x[0]`, whereas `x[0]` is
rejected because it selects from the batch axis.

## 6. Canonicalization groups

```text
nn.ReLU, F.relu, torch.relu, x.relu() -> ReLU
```

```text
x + y, torch.add(x, y), x.add(y) -> Add
```

```text
torch.reshape(x, shape), x.reshape(shape), x.view(shape) -> Reshape
```

Container modules such as `nn.Sequential` and user-defined residual blocks do
not become canonical operators. FX traces their internal tensor operations,
and NCET normalizes those operations individually while retaining their graph
connections.

## 7. Frontend spelling boundary

This document defines only whether an FX spelling can be projected onto a
canonical operator. Recognition of a spelling does not by itself mean that
every parameterization of that operation is supported.

- In-place forms such as `relu_()`, `add_()`, or `F.relu(..., inplace=True)`
  are rejected before GraphIR normalization.
- `F.linear()` and `F.conv2d()` are not currently mapped; use `nn.Linear` and
  `nn.Conv2d`.
- Direct FX `get_attr` nodes are not canonicalized as standalone operators.
  State belonging to supported Linear and Conv2d modules is instead lifted
  into `GraphIR.constants` during normalization.
- Any FX operation spelling not listed above raises
  `UnsupportedOperatorError`.

After a spelling is recognized, its operator attributes, tensor ranks, and
batch-preservation rules must still satisfy the authoritative
[current operator support boundary](supported_operators.md). A graph inside
that boundary is governed by the [exactness contract](exactness_contract.md).
