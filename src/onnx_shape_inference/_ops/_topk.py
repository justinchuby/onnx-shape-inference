# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for TopK operator."""

from __future__ import annotations

__all__ = [
    "infer_topk",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


@_registry.registry.register("", "TopK", since_version=11)
def infer_topk(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for TopK operator."""
    (x, k) = _context.check_inputs(node, "X", "K")

    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else -1

    output_shape: ir.Shape | None = None
    if x.shape is not None:
        rank = x.shape.rank()
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            ctx.record_error(node, f"axis={axis} is out of range for rank {rank}")
            return

        new_dims: list[int | ir.SymbolicDim] = []
        k_val: int | ir.SymbolicDim | None = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            k_const = ir.convenience.get_const_tensor(node.inputs[1])
            if k_const is not None:
                k_val = int(k_const.numpy().item())
            else:
                k_symbolic = ctx.get_symbolic_value(k)
                if k_symbolic is not None and len(k_symbolic) == 1:
                    k_val = k_symbolic[0]
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
