# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Dropout operator."""

from __future__ import annotations

__all__ = [
    "infer_dropout",
    "infer_dropout_v10",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


def _infer_dropout(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    *,
    mask_uses_input_dtype: bool,
) -> None:
    """Infer Dropout outputs using the mask dtype for the selected opset."""
    (data,) = _context.check_inputs(node, "data")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], data.shape, data.dtype)
    if len(node.outputs) > 1:
        mask_dtype = data.dtype if mask_uses_input_dtype else ir.DataType.BOOL
        ctx.set_shape_and_dtype(node.outputs[1], data.shape, mask_dtype)


@_registry.registry.register("", "Dropout", since_version=7)
def infer_dropout(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Dropout operator versions 7-9.

    Output[0] has the same shape and dtype as input[0].
    Output[1] (optional mask) has the same shape and dtype as input[0].

    Spec: https://onnx.ai/onnx/operators/onnx__Dropout.html
    """
    _infer_dropout(ctx, node, mask_uses_input_dtype=True)


@_registry.registry.register("", "Dropout", since_version=10)
def infer_dropout_v10(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Dropout operator version 10 and later."""
    _infer_dropout(ctx, node, mask_uses_input_dtype=False)
