# onnx-shape-inference

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)
[![codecov](https://codecov.io/gh/justinchuby/onnx-shape-inference/graph/badge.svg?token=JF1zsZfrLM)](https://codecov.io/gh/justinchuby/onnx-shape-inference)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI Downloads](https://static.pepy.tech/badge/onnx-shape-inference/month)](https://pepy.tech/projects/onnx-shape-inference)

Experimental symbolic shape inference for ONNX models. Built on top of [ONNX IR](https://github.com/onnx/ir-py), this library performs shape inference directly on the IR without serialization overhead, using [SymPy](https://www.sympy.org/) for symbolic dimension arithmetic.

## Features

- **Symbolic shape inference** — propagates shapes through the graph using SymPy expressions for symbolic dimensions
- **Shape data propagation** — tracks known element values of shape tensors (e.g. through `Shape → Slice → Concat → Reshape` chains) to resolve concrete output shapes that standard shape inference cannot
- **Extensible registry** — register custom shape inference functions for custom operators
- **Merge policies** — choose between strict and permissive shape merging strategies

## Installation

```console
pip install onnx-shape-inference
```

Or install from source (main branch):

```console
pip install git+https://github.com/justinchuby/onnx-shape-inference.git
```

## Command line

Run shape inference on a model and see how many new shapes were inferred:

```console
onnx-shape-inference model.onnx
```

Save the inferred model to a file:

```console
onnx-shape-inference model.onnx -o model_inferred.onnx
```

Overwrite the input model in place:

```console
onnx-shape-inference model.onnx --in-place
```

Select a different merge policy:

```console
onnx-shape-inference model.onnx --policy strict
```

## Usage

```python
import onnx_ir as ir
from onnx_shape_inference import infer_symbolic_shapes

# Load a model
model = ir.load("model.onnx")

# Run shape inference
model = infer_symbolic_shapes(model)

# Or with a strict merge policy
model = infer_symbolic_shapes(model, policy="strict")
```

### Use with onnxscript optimizer

You can run symbolic shape inference on the model to help the optimizer discover more optimization opportunities.

```py
import onnx_shape_inference
import onnx_ir as ir
import onnxscript.optimizer

model = ir.load("model.onnx")

# Provide more shape information with infer_symbolic_shapes
model = onnx_shape_inference.infer_symbolic_shapes(model)

# onnxscript optimizer can leverage this information to better optimize the model
onnxscript.optimizer.optimize(model)

ir.save(model, "model_optimized.onnx")
```

### Per-node inference

You can run shape inference on individual nodes by using the
`ShapeInferenceContext` and `registry` directly. This is useful for
debugging, testing, or integrating into custom graph passes.

```python
import onnx_ir as ir
from onnx_shape_inference import ShapeInferenceContext, registry

# Populate the registry with all built-in ops
registry.collect()

# Create a context with the model's opset imports
ctx = ShapeInferenceContext(opset_imports={"": 21})

# Look up the inference function for the op
infer_func = registry.get("", "Relu", version=21)

# Build a node (or get one from an existing graph)
x = ir.Value(name="x", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT))
y = ir.Value(name="y")
node = ir.Node("", "Relu", inputs=[x], outputs=[y])

# Run inference
infer_func(ctx, node)

print(y.shape)  # [2,3]
print(y.dtype)  # FLOAT
```

### Registering custom operators

```python
from onnx_shape_inference import registry

@registry.register("com.custom", "MyOp", since_version=1)
def infer_my_op(ctx, node):
    input_shape = node.inputs[0].shape
    output_shape = ir.Shape([...])
    ctx.set_shape(node.outputs[0], output_shape)
```

### Shape data propagation (`pkg.onnx_shape_inference.sym_data`)

Shape inference alone cannot resolve output shapes when ops like `Reshape` consume
non-constant shape tensors that were computed at runtime (e.g. `Shape → Slice → Concat → Reshape`).
The **sym_data** feature bridges this gap by tracking the known element values of
1-D integer tensors as they flow through the graph.

After inference, each value that carries propagated data has a
`pkg.onnx_shape_inference.sym_data` entry in its `metadata_props`. You can read
it directly or use the `SYM_DATA_KEY` constant:

```python
import json
import numpy as np
import onnx_ir as ir
from onnx_shape_inference import SYM_DATA_KEY, infer_symbolic_shapes

model = infer_symbolic_shapes(model)

for node in model.graph:
    for value in node.inputs:
        if SYM_DATA_KEY in value.metadata_props:
            text = value.metadata_props[SYM_DATA_KEY]  # e.g. '["N",3,768]'
            elements = json.loads(text)                # ["N", 3, 768]

            # You can create an ir.Shape from it
            shape = ir.Shape(elements)

            # Then you can replace this input with a constant value
```

When all elements are concrete integers the value is also stored as a constant
tensor, so downstream consumers that read constants directly can access it
without parsing `metadata_props`.

## Development

```console
pip install pytest parameterized
pip install -e .
pytest
```

## License

`onnx-shape-inference` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
