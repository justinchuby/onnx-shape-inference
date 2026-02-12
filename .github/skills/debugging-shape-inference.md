# Debugging Shape Inference Errors

## Diagnosing "Inferred shape and existing shape differ" from ONNX C++

When `onnx.shape_inference.infer_shapes(model, strict_mode=True)` raises
`ShapeInferenceError: Inferred shape and existing shape differ in dimension N`,
the error does **not** include the node or value name. Use the following
workflow to locate the offending value.

### Step 1: Confirm the error source

```python
import onnx

model = onnx.load("model.onnx")
try:
    onnx.shape_inference.infer_shapes(model, data_prop=True, strict_mode=True)
except Exception as e:
    print(e)  # Note the dimension index and values, e.g. dim 0: (4) vs (5)
```

### Step 2: Strip value_info and re-infer

If the model infers cleanly without `value_info`, the bug is in one of those
annotations (not in the graph structure):

```python
stripped = onnx.ModelProto()
stripped.CopyFrom(model)
del stripped.graph.value_info[:]
onnx.shape_inference.infer_shapes(stripped, data_prop=True, strict_mode=True)
# If this succeeds, the error is in value_info
```

### Step 3: Binary search for the problematic value_info entry

```python
vis = list(model.graph.value_info)

def test_with_vis(model, vis_to_keep):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    del m.graph.value_info[:]
    for vi in vis_to_keep:
        m.graph.value_info.append(vi)
    try:
        onnx.shape_inference.infer_shapes(m, data_prop=True, strict_mode=True)
        return True
    except:
        return False

lo, hi = 0, len(vis)
while lo < hi:
    mid = (lo + hi) // 2
    if test_with_vis(model, vis[:mid + 1]):
        lo = mid + 1
    else:
        hi = mid

bad_vi = vis[lo]
print(f"Problematic value: {bad_vi.name}")
```

### Step 4: Compare against ground truth

For initializers, the actual tensor shape is ground truth:

```python
for init in model.graph.initializer:
    if init.name == bad_vi.name:
        print(f"Initializer shape: {list(init.dims)}")

# Extract the annotated shape from value_info
dims = []
for d in bad_vi.type.tensor_type.shape.dim:
    dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
print(f"Annotated shape: {dims}")
```

For intermediate values, compare against what ONNX C++ infers from scratch:

```python
stripped = onnx.ModelProto()
stripped.CopyFrom(model)
del stripped.graph.value_info[:]
inferred = onnx.shape_inference.infer_shapes(stripped, data_prop=True)

for vi in inferred.graph.value_info:
    if vi.name == bad_vi.name:
        # This is what ONNX C++ thinks the shape should be
        print(vi)
```

## Common root causes

| Symptom | Likely cause |
|---------|-------------|
| Initializer shape != value_info shape | Bad annotation in the original model. Fix: correct shapes from the actual tensor at inference start. |
| Intermediate value has wrong concrete dim | Our op inference computed a wrong dimension. Fix: debug the specific op's `infer_*` function. |
| Symbolic dim where C++ infers concrete | Our inference was less precise but not wrong. Usually not an error in strict mode. |

## Prevention: initializer shape correction

The engine (`_engine.py`) corrects initializer shapes at the start of
`_process_graph` using the actual tensor as ground truth. This handles
malformed models where `value_info` annotations disagree with initializer
tensor shapes.
