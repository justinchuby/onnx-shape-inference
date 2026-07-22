# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Emit minimized ONNX artifacts and paste-ready regression-test guidance."""

from __future__ import annotations

from pathlib import Path

import onnx
import onnx_ir as ir

from onnx_shape_inference._fuzz._shrink import FailureSignature
from onnx_shape_inference._fuzz._types import FuzzCase, OracleResult

__all__ = ["emit_onnx", "render_reproducer"]


def emit_onnx(case: FuzzCase, path: str | Path) -> Path:
    """Serialize a minimized case for a nightly artifact; never commit it by default."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(ir.serde.serialize_model(case.model), destination)
    return destination


def render_reproducer(
    case: FuzzCase,
    signature: FailureSignature,
    result: OracleResult,
    *,
    artifact_name: str = "minimized.onnx",
) -> str:
    """Return compact test guidance whose expected fact comes from ground truth."""
    target = (
        "a colocated src/onnx_shape_inference/_ops/_<op>_test.py parameterized case"
        if len(case.model.graph) == 1
        else "tests/shape_inference_cases_test.py"
    )
    return "\n".join(
        [
            "# Ground-truth-derived fuzz reproducer",
            "import onnx",
            "import onnx_ir as ir",
            f"model = ir.serde.deserialize_model(onnx.load({artifact_name!r}))",
            "# Add this to " + target + ".",
            f"# seed={case.seed}; oracle={signature.oracle}; value={signature.value_name!r};",
            f"# details={result.details!r}; kind={signature.kind!r}",
        ]
    )
