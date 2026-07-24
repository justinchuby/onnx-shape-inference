# Project Context

- **Owner:** justinchu@microsoft.com
- **Project:** onnx-shape-inference — experimental symbolic shape inference for ONNX models, built on ONNX IR + SymPy
- **Stack:** Python, ONNX IR (onnx-ir), SymPy, pytest, unittest+parameterized, Ruff, lintrunner
- **Created:** 2026-07-22T21:28:52Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
- 📌 Team update (2026-07-23T07:41:00+0000): SoundnessOracle now reruns isolated single-node inference with fresh untyped outputs to eliminate dtype-contradiction false positives; the remaining coverage gaps are tracked separately in the harness.
2026-07-23: Completed axis validation and historical opset registration on branch fix/axis-historical. Added shared axis normalization/validation helper, regression tests, and pushed commit 35d7153. Targeted tests: 331 passed; full suite: 3229 passed; Ruff passed.
2026-07-23: Fixed release blocker by reverting RNN/GRU/LSTM registration floor from opset 1 to 7 because opsets 1-6 output_sequence changes Y existence. Added RNN opset-1 rejection/opset-7 output boundary tests. Pushed 0d7bfda; targeted 205 passed, full 3227 passed, Ruff passed.
2026-07-23: Added tests-only coverage for axis validation and graceful skip-policy degradation. Coverage: arg 100%, data_dependent 99%, gather 100%, normalization 97%, onehot 100%, split 100%, squeeze 99%. Full suite 3249 passed. Pushed 17163d9.
2026-07-23: Fixed best-effort axis error outputs: Gather now preserves data dtype; TopK types both values and indices outputs. Added strict and skip-policy tests plus Flatten skip coverage. Targeted 195 passed with 97-100% relevant coverage; full 3252 passed. Pushed f52707a.
