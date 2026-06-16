# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Concat operator."""

from __future__ import annotations

__all__ = [
    "infer_concat",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


@_registry.registry.register("", "Concat", since_version=4)
def infer_concat(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Concat operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Concat.html
    """
    _context.check_inputs(node, "inputs[0]")
    axis_attr = _context.require_attr(node, "axis")
    axis = axis_attr.as_int()

    # Collect shapes and dtype
    shapes: list[ir.Shape] = []
    output_dtype: ir.DataType | None = None

    for inp in node.inputs:
        if inp is None:
            raise _context.OpUsageError(node, "Required input is None")
        if output_dtype is None:
            output_dtype = inp.dtype
        if inp.shape is None:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
            return
        shapes.append(inp.shape)

    if not shapes:
        return

    rank = shapes[0].rank()
    # Normalize negative axis
    if axis < 0:
        axis += rank

    if not 0 <= axis < rank:
        ctx.record_error(node, f"axis={axis} is out of range for rank {rank}")
        return

    for i, s in enumerate(shapes):
        if s.rank() != rank:
            ctx.record_error(
                node,
                f"Input {i} has rank {s.rank()}, expected {rank}",
            )
            return

    # Build output shape
    output_dims: list[int | ir.SymbolicDim] = []
    for dim_idx in range(rank):
        if dim_idx == axis:
            # Sum along concat axis
            total: int | ir.SymbolicDim = 0
            for s in shapes:
                total = total + s[dim_idx]
            output_dims.append(total)  # type: ignore[arg-type]
        else:
            output_dims.append(shapes[0][dim_idx])

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

        # Propagate symbolic_value for 1-D tensors concatenated on axis 0.
        # When an input lacks a known symbolic value but has a concrete 1-D
        # length, fill that segment with fresh symbolic dims (one per element)
        # rather than abandoning propagation entirely — this keeps the
        # surrounding known elements available downstream (e.g. a shape tensor
        # ``[B, H, <unknown>, S, D]`` built by Concat for a later Reshape /
        # Expand).  Propagation is only skipped when a segment length itself is
        # unknown.
        if rank == 1 and axis == 0:
            # All inputs are guaranteed non-None here (the shape-collection loop
            # above raises for any None input).
            combined: list[int | ir.SymbolicDim] = []
            propagate = True
            for inp in node.inputs:
                sv = ctx.get_symbolic_value(inp)
                if sv is not None:
                    combined.extend(sv)
                    continue
                # No known values: fall back to the input's own length.
                length = inp.shape[0] if inp.shape is not None else None
                if isinstance(length, int):
                    combined.extend(ctx.new_symbolic_dim() for _ in range(length))
                else:
                    propagate = False
                    break
            if propagate:
                ctx.set_symbolic_value(node.outputs[0], combined)
