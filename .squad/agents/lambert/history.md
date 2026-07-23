# Project Context

- **Owner:** justinchu@microsoft.com
- **Project:** onnx-shape-inference — experimental symbolic shape inference for ONNX models, built on ONNX IR + SymPy
- **Stack:** Python, ONNX IR (onnx-ir), SymPy, pytest, unittest+parameterized, Ruff, lintrunner
- **Created:** 2026-07-22T21:28:52Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
- 📌 Team update (2026-07-23T07:41:00+0000): SoundnessOracle now reruns isolated single-node inference with fresh untyped outputs to eliminate dtype-contradiction false positives; the remaining coverage gaps are tracked separately in the harness.
