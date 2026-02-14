# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ScatterElements and ScatterND operators."""

from __future__ import annotations

__all__ = [
    "infer_scatter",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


@_registry.registry.register("", "Scatter", since_version=9)
@_registry.registry.register("", "ScatterElements", since_version=11)
@_registry.registry.register("", "ScatterND", since_version=11)
@_registry.registry.register("", "TensorScatter", since_version=24)
def infer_scatter(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Scatter, ScatterElements, ScatterND, and TensorScatter.

    Output shape and dtype match the data (first) input.
    """
    (data, _indices, _updates) = _context.check_inputs(node, "data", "indices", "updates")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)
