# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Where operator."""

from __future__ import annotations

__all__ = [
    "infer_where",
]

import onnx_ir as ir

from onnx_shape_inference import _broadcast, _context, _registry


@_registry.registry.register("", "Where", since_version=9)
def infer_where(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Where operator.

    Output shape is the broadcast of condition, X, and Y.
    Output dtype is the same as X.

    Spec: https://onnx.ai/onnx/operators/onnx__Where.html
    """
    (condition, x, y) = _context.check_inputs(node, "condition", "X", "Y")

    output_dtype = x.dtype or y.dtype

    # Broadcast all three shapes
    shape_xy = _broadcast.broadcast_shapes(x.shape, y.shape)
    output_shape = _broadcast.broadcast_shapes(condition.shape, shape_xy)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    _propagate_where_symbolic_value(ctx, node, x, y)


def _propagate_where_symbolic_value(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    x: ir.Value,
    y: ir.Value,
) -> None:
    """Propagate the symbolic value of a 1-D ``Where`` over shape tensors.

    For each element position, when both branches carry the *same* known value
    that value is selected regardless of the (unknown) condition; otherwise the
    result is data-dependent and gets a fresh symbolic dim.  This keeps a
    ``Where`` embedded in a shape-building chain (e.g. choosing an interleave
    factor for an Attention KV expansion) from blocking downstream propagation.
    """
    if not node.outputs:
        return
    val_x = ctx.get_symbolic_value(x)
    val_y = ctx.get_symbolic_value(y)
    if val_x is None or val_y is None or len(val_x) != len(val_y):
        return

    result: list[int | ir.SymbolicDim] = []
    for a, b in zip(val_x, val_y):
        if isinstance(a, int) and isinstance(b, int) and a == b:
            result.append(a)
        elif (
            isinstance(a, ir.SymbolicDim)
            and isinstance(b, ir.SymbolicDim)
            and a.value is not None
            and a.value == b.value
        ):
            result.append(a)
        else:
            result.append(ctx.new_symbolic_dim())
    ctx.set_symbolic_value(node.outputs[0], result)
