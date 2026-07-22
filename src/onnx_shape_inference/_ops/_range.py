# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Range operator."""

from __future__ import annotations

__all__ = [
    "infer_range",
]

import math

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _utils


def _scalar_value(
    ctx: _context.ShapeInferenceContext, value: ir.Value
) -> int | float | ir.SymbolicDim | None:
    """Read a known scalar from a constant or symbolic data."""
    const = ir.convenience.get_const_tensor(value)
    if const is not None:
        array = const.numpy()
        if array.size == 1:
            return array.item()
        return None
    symbolic_value = ctx.get_symbolic_value(value)
    if symbolic_value is not None and len(symbolic_value) == 1:
        return symbolic_value[0]
    return None


@_registry.registry.register("", "Range", since_version=11)
def infer_range(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Range operator."""
    (start, limit, delta) = _context.check_inputs(node, "start", "limit", "delta")

    output_len: int | ir.SymbolicDim | None = None
    start_value = _scalar_value(ctx, start)
    limit_value = _scalar_value(ctx, limit)
    delta_value = _scalar_value(ctx, delta)
    if (
        start_value is not None
        and limit_value is not None
        and isinstance(delta_value, (int, float))
    ):
        if delta_value == 0:
            ctx.record_error(node, "Range: delta must not be zero")
            return
        if isinstance(start_value, (int, float)) and isinstance(limit_value, (int, float)):
            output_len = max(0, math.ceil((limit_value - start_value) / delta_value))
        elif isinstance(delta_value, int) and delta_value > 0:
            output_len = _utils.ceil_div_dim(limit_value - start_value, delta_value)

    if output_len is None:
        output_len = ctx.new_symbolic_dim()

    output_shape = ir.Shape([output_len])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, start.dtype)
