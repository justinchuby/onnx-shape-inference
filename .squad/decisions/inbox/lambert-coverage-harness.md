# Coverage regression harness

**Date:** 2026-07-22T21:36:42Z

Added `src/onnx_shape_inference/_coverage_regression_test.py`, a local harness
for the coverage benchmark gaps. A parameterized table exercises symbolic
reshape simplification, symbolic Slice end, Resize→Tile, dynamic TopK,
NonZero arithmetic, If branch-output names, and tiny-LLM Concat anchor names.
The qwen-style Reshape→Transpose→MatMul→Unsqueeze forward-propagation path is
also covered, along with constant-K and unknown-shape dtype-only boundaries.

Current targeted status: **5 passed, 7 xfailed**. The qwen forwarding and
boundary cases are green. The seven xfails are real expected-shape assertions,
pending symbolic simplification/derived-dimension support and graph-output
anchor/constraint propagation. The tiny-LLM test verifies both the existing
`past_seq + seq` derivation and its required `total_seq` adoption.
