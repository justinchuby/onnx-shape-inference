# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Slice operator."""

from __future__ import annotations

__all__ = [
    "infer_slice",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _utils


def _read_ints(
    ctx: _context.ShapeInferenceContext, value: ir.Value | None
) -> list[int] | None:
    """Read known integer tensor values from a constant or symbolic data."""
    values = _utils.get_known_dim_values(ctx, value)
    if values is not None and all(isinstance(item, int) for item in values):
        return list(values)
    return None


@_registry.registry.register("", "Slice", since_version=10)
def infer_slice(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Slice operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Slice.html
    """
    (data, _, _) = _context.check_inputs(node, "data", "starts", "ends")

    input_shape = data.shape
    input_dtype = data.dtype

    if input_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, input_dtype)
        return

    rank = input_shape.rank()
    starts = _read_ints(ctx, node.inputs[1])
    ends_values = _utils.get_known_dim_values(ctx, node.inputs[2])

    if starts is None or ends_values is None:
        # Dynamic starts/ends — same rank, sliced dims are symbolic
        if len(node.outputs) > 0:
            symbolic_dims: list[int | ir.SymbolicDim] = [
                ctx.new_symbolic_dim() for _ in range(rank)
            ]
            ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(symbolic_dims), input_dtype)
        return

    axes: list[int] | None = None
    if len(node.inputs) >= 4:
        axes = _read_ints(ctx, node.inputs[3])

    steps: list[int] | None = None
    if len(node.inputs) >= 5:
        steps = _read_ints(ctx, node.inputs[4])

    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)

    output_dims: list[int | ir.SymbolicDim] = list(input_shape.dims)
    for i, (start, axis, step) in enumerate(zip(starts, axes, steps)):
        end = ends_values[i]
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            continue

        dim = input_shape[axis]
        if step == 0:
            ctx.record_error(node, f"Step cannot be 0 for axis {axis}")
            return

        sentinels = {2**63 - 1, -(2**63), 2**31 - 1, -(2**31)}
        if start == 0 and step == 1 and isinstance(end, int) and end in sentinels:
            output_dims[axis] = dim
        elif step == -1 and start in sentinels and end in sentinels:
            output_dims[axis] = dim
        elif isinstance(dim, int) and isinstance(end, int):
            # Clamp start/end to [0, dim] for positive step, [-1, dim-1] for negative
            if step > 0:
                clamped_start = max(0, min(start if start >= 0 else start + dim, dim))
                clamped_end = max(0, min(end if end >= 0 else end + dim, dim))
            else:
                clamped_start = max(-1, min(start if start >= 0 else start + dim, dim - 1))
                clamped_end = max(-1, min(end if end >= 0 else end + dim, dim - 1))

            slice_len = max(
                0, (clamped_end - clamped_start + (step - (1 if step > 0 else -1))) // step
            )
            output_dims[axis] = slice_len
        elif step > 0 and start >= 0:
            if isinstance(end, int) and end >= 0:
                # Assume the common concrete-bounds case is in range so downstream
                # optimization sees a concrete extent instead of a symbolic clamp.
                output_dims[axis] = max(0, (end - start + step - 1) // step)
            else:
                effective_end = dim + end if isinstance(end, int) and end < 0 else end
                clamped_end = _utils.min_dim(dim, effective_end)
                if clamped_end is None:
                    output_dims[axis] = ctx.new_symbolic_dim()
                    continue
                sliced_extent = _utils.ceil_div_dim(clamped_end - start, step)
                nonnegative_extent = _utils.max_dim(0, sliced_extent)
                output_dims[axis] = (
                    nonnegative_extent
                    if nonnegative_extent is not None
                    else ctx.new_symbolic_dim()
                )
        else:
            output_dims[axis] = ctx.new_symbolic_dim()

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)

        # Propagate symbolic_value for 1-D tensors sliced on axis 0
        sym_val = ctx.get_symbolic_value(data)
        ends = _read_ints(ctx, node.inputs[2])
        if (
            sym_val is not None
            and ends is not None
            and rank == 1
            and len(axes) == 1
            and axes[0] == 0
        ):
            n = len(sym_val)
            s, e, st = starts[0], ends[0], steps[0]
            # Clamp start/end like Python slicing
            if s < 0:
                s = max(0, s + n)
            else:
                s = min(s, n)
            sentinels = {2**63 - 1, -(2**63), 2**31 - 1, -(2**31)}
            if e in sentinels:
                e = n if st > 0 else -n - 1
            elif e < 0:
                e = max(0, e + n)
            else:
                e = min(e, n)
            ctx.set_symbolic_value(node.outputs[0], sym_val[s:e:st])
