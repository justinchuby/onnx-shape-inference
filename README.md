# onnx-shape-inference

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)

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

## Development

```console
pip install pytest parameterized
pip install -e .
pytest
```

## License

`onnx-shape-inference` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
