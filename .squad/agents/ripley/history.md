# Project Context

- **Owner:** justinchu@microsoft.com
- **Project:** onnx-shape-inference — experimental symbolic shape inference for ONNX models, built on ONNX IR + SymPy
- **Stack:** Python, ONNX IR (onnx-ir), SymPy, pytest, unittest+parameterized, Ruff, lintrunner
- **Created:** 2026-07-22T21:28:52Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
- 📌 Team update (2026-07-23T07:41:00+0000): Shared symbol identity and reshape provenance are now the general fixes for anchor adoption and exact divisibility; the oracle false-positive fix was kept central in SoundnessOracle, not split into per-op dtype planners.
- 📌 Team update (2026-07-23T08:16:00+0000): SoundnessOracle now feeds ORT runtime ground truth downstream between isolated nodes and checks producer const_value claims against actual arrays, closing the sym_data masking gap generally.
