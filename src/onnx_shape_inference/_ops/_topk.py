# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for TopK operator."""

from __future__ import annotations

__all__ = [
    "infer_topk",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _shape_ops, _utils


@_registry.registry.register("", "TopK", since_version=10)
def infer_topk(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TopK operator."""
    (x, k) = _context.check_inputs(node, "X", "K")

    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else -1

    output_shape: ir.Shape | None = None
    if x.shape is not None:
        rank = x.shape.rank()
        axis = _shape_ops.normalize_axis(ctx, node, axis, rank)
        if axis is None:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
            if len(node.outputs) > 1:
                ctx.set_shape_and_dtype(node.outputs[1], None, ir.DataType.INT64)
            return

        new_dims: list[int | ir.SymbolicDim] = []
        k_val = _utils.get_known_scalar(ctx, k)
        if not isinstance(k_val, (int, ir.SymbolicDim)):
            k_val = None
        if k_val is None:
            k_val = ctx.new_symbolic_dim()
        for i in range(rank):
            if i == axis:
                new_dims.append(k_val)
            else:
                new_dims.append(x.shape[i])
        output_shape = ir.Shape(new_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)
    if len(node.outputs) > 1:
        ctx.set_shape_and_dtype(node.outputs[1], output_shape, ir.DataType.INT64)
