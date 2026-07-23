# Fuzzing regressions

The deterministic fast fuzzer runs `tests/shape_inference_fuzz_test.py`; replay a
failure with its printed seed:

```bash
FUZZ_SEED=<seed> python3 -m pytest tests/shape_inference_fuzz_test.py
```

Nightly artifacts include a minimized `.onnx` file and ground-truth-derived
reproducer guidance. Do not commit a standalone repro file:

- For a single operator, add a parameterized case to its colocated
  `src/onnx_shape_inference/_ops/_<op>_test.py`, using `ts()` and the test helpers.
- For a multi-op graph, add a builder/assertion to
  `tests/shape_inference_cases_test.py`.

Use the oracle's ONNX Runtime/reference expected shape, never the buggy inferred
shape, as the test expectation.
