# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point that isolates unsafe ONNX Runtime graph execution."""

from __future__ import annotations

import base64
import json
import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort


def _read_exact(size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sys.stdin.buffer.read(size - len(chunks))
        if not chunk:
            return None if not chunks else b""
        chunks.extend(chunk)
    return bytes(chunks)


def _read_request() -> dict[str, object] | None:
    header = _read_exact(4)
    if header is None:
        return None
    if len(header) != 4:
        raise ValueError("truncated request header")
    length = struct.unpack("!I", header)[0]
    payload = _read_exact(length)
    if payload is None or len(payload) != length:
        raise ValueError("truncated request payload")
    return json.loads(payload)


def _write_response(response: dict[str, object]) -> None:
    encoded = json.dumps(response, sort_keys=True).encode()
    sys.stdout.buffer.write(struct.pack("!I", len(encoded)) + encoded)
    sys.stdout.buffer.flush()


def _decode_feeds(payload: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.frombuffer(
            base64.b64decode(spec["data"]), dtype=np.dtype(spec["dtype"])
        ).reshape(spec["shape"])
        for name, spec in payload.items()
    }


def _run(request: dict[str, object]) -> dict[str, dict[str, object]]:
    model = onnx.ModelProto()
    model.ParseFromString(base64.b64decode(request["model"]))
    feeds = _decode_feeds(request["feeds"])
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    session_inputs = {input_.name for input_ in session.get_inputs()}
    outputs = session.run(
        None, {name: value for name, value in feeds.items() if name in session_inputs}
    )
    facts = {
        output.name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for output, value in zip(session.get_outputs(), outputs)
    }
    return facts


def main() -> None:
    """Serve stateless runtime requests until the parent closes stdin."""
    while (request := _read_request()) is not None:
        try:
            _write_response({"ok": True, "facts": _run(request)})
        except Exception as error:
            _write_response({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
