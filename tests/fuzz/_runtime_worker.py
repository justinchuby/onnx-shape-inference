# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point that isolates unsafe ONNX Runtime graph execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main(model_path: str, feeds_path: str, result_path: str) -> None:
    """Run one concrete model and write output dtype/shape facts as JSON."""
    with np.load(feeds_path, allow_pickle=False) as feeds_file:
        feeds = {name: feeds_file[name] for name in feeds_file.files}
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    session_inputs = {input_.name for input_ in session.get_inputs()}
    outputs = session.run(
        None, {name: value for name, value in feeds.items() if name in session_inputs}
    )
    facts = {
        output.name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for output, value in zip(session.get_outputs(), outputs)
    }
    Path(result_path).write_text(json.dumps(facts, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:])
