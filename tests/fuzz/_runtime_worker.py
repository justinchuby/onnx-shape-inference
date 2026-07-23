# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point that isolates unsafe ONNX Runtime graph execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main(
    model_path: str,
    feeds_path: str,
    result_path: str,
    arrays_path: str | None = None,
) -> None:
    """Run one concrete model and write output dtype/shape facts and arrays.

    The JSON at *result_path* records each output's dtype and shape. When
    *arrays_path* is provided, the actual runtime arrays are also written as an
    ``.npz`` archive so that upstream values can be fed to downstream isolated
    nodes as ground truth. Non-numeric outputs (for example string tensors) are
    reported in the JSON but omitted from the archive.
    """
    with np.load(feeds_path, allow_pickle=False) as feeds_file:
        feeds = {name: feeds_file[name] for name in feeds_file.files}
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    session_inputs = {input_.name for input_ in session.get_inputs()}
    outputs = session.run(
        None, {name: value for name, value in feeds.items() if name in session_inputs}
    )
    facts: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for output, value in zip(session.get_outputs(), outputs):
        array = np.asarray(value)
        facts[output.name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
        if array.dtype != object:
            arrays[output.name] = array
    Path(result_path).write_text(json.dumps(facts, sort_keys=True))
    if arrays_path is not None:
        np.savez(arrays_path, **arrays)


if __name__ == "__main__":
    main(*sys.argv[1:])
