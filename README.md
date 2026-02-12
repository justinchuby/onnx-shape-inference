# onnx-shape-inference

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-shape-inference.svg)](https://pypi.org/project/onnx-shape-inference)

Experimental symbolic shape inference for ONNX models. Built on top of [ONNX IR](https://github.com/onnx/ir-py), this library performs shape inference directly on the IR without serialization overhead, using [SymPy](https://www.sympy.org/) for symbolic dimension arithmetic.

## Features

- **Symbolic shape inference** — propagates shapes through the graph using SymPy expressions for symbolic dimensions
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
pip install pytest
pip install -e .
pytest
```

## License

`onnx-shape-inference` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
