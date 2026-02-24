# Copilot Instructions

## Build, Test, Lint

```bash
# Install for development
pip install -r requirements/ci/requirements.txt && pip install -e .

# Run all tests
pytest

# Run a single test file
pytest src/onnx_shape_inference/_ops/_slice_test.py -x -q

# Run a single test method
pytest src/onnx_shape_inference/_ops/_slice_test.py -x -q -k test_basic

# Lint and auto-fix (requires lintrunner + lintrunner-adapters)
lintrunner -a --output oneline
```

## Architecture

This library performs symbolic shape inference on ONNX models using the [onnx-ir](https://github.com/onnx/ir-py) in-memory representation (no serialization). It uses [SymPy](https://www.sympy.org/) (via `ir.SymbolicDim`) for symbolic dimension arithmetic.

**Core flow:** `infer_symbolic_shapes(model)` → `_engine._process_graph()` iterates nodes in topological order → looks up each op's inference function from the `_registry.registry` → calls it with a `ShapeInferenceContext` and the `ir.Node`.

**Key modules:**
- `_engine.py` — Graph traversal, initializer shape correction, anonymous dim naming
- `_context.py` — `ShapeInferenceContext` (merge policies, symbolic dim creation, sym_data propagation), `check_inputs()`, `require_attr()`
- `_registry.py` — `OpShapeInferenceRegistry` with since_version dispatch
- `_broadcast.py` — Multidirectional broadcast shape computation
- `_ops/` — One file per op group (e.g. `_conv.py`, `_elementwise.py`), tests colocated as `_*_test.py`
- `_ops/_testing.py` — `ts()`, `run_shape_inference()`, `run_shape_inference_with_values()`, `const_value()`

**Data propagation:** The `sym_data` system tracks known element values of 1-D integer tensors (shape tensors) through ops like `Shape → Slice → Concat → Reshape`. This lets Reshape resolve output shapes even when the shape input isn't a constant. Stored in `value.metadata_props[SYM_DATA_KEY]`.

## Conventions

### Imports

In source files (`_ops/*.py`), import **modules only** — not names:

```python
# Good
from onnx_shape_inference import _context, _registry

# Bad
from onnx_shape_inference._context import check_inputs
```

In test files, importing `ts` directly from `_testing` is allowed:

```python
from onnx_shape_inference._ops._testing import ts
```

### Op registration

Use `@_registry.registry.register(domain, op_type, since_version=N)` as a decorator. Stack decorators when multiple ops share the same logic:

```python
_reg = _registry.registry.register

@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
def infer_binary_elementwise(ctx, node):
    ...
```

New op files must be imported in `_ops/__init__.py` to trigger registration.

### Symbolic dimensions

- Use `ctx.new_symbolic_dim()` for unknown dimensions — never `ir.SymbolicDim(None)`
- Prefer arithmetic on symbolic dims (`in_dim + 2`, `H // stride`) when the relationship to input dims is known
- Reserve `ctx.new_symbolic_dim()` for truly data-dependent or unknowable dimensions

### Error handling

- `check_inputs(node, "a", "b")` and `require_attr(node, "axis")` raise `OpUsageError` for malformed models
- `ctx.record_error(node, msg)` for semantic errors (axis out of range, rank mismatch)
- When input shape is unknown, set dtype only and return early

### Testing

Tests use `unittest.TestCase` with `parameterized.parameterized.expand`. Use the `ts()` helper for concise type+shape assertions:

```python
actual = run_shape_inference("", "Relu", [ts(FLOAT, [3, 4])], opset_version=21)
self.assertEqual(actual, [ts(FLOAT, [3, 4])])
```

Use `const_value([...])` for ops that read constant inputs (Reshape, Slice, TopK).

### Style

- Ruff for linting and formatting (line length 95, Google-style docstrings)
- All relative imports are banned; use absolute imports
- `from __future__ import annotations` at the top of every file
- Each op file declares `__all__`
