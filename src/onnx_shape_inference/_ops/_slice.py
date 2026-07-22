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
    if value is None:
        return None
    const = ir.convenience.get_const_tensor(value)
    if const is not None:
        return [int(x) for x in const.numpy().flatten()]
    symbolic_value = ctx.get_symbolic_value(value)
    if symbolic_value is not None and all(isinstance(x, int) for x in symbolic_value):
        return list(symbolic_value)
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
    ends = _read_ints(ctx, node.inputs[2])

    # Try symbolic value for ends when not constant
    ends_sym: list[int | ir.SymbolicDim] | None = None
    if ends is None:
        ends_sym = ctx.get_symbolic_value(node.inputs[2])

    if starts is None or (ends is None and ends_sym is None):
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

    # Use concrete ends or symbolic ends
    ends_values: list[int] | list[int | ir.SymbolicDim]
    if ends is not None:
        ends_values = ends
    else:
        assert ends_sym is not None
        ends_values = ends_sym

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

        if not isinstance(end, int):
            if step > 0 and start >= 0:
                sliced_extent = end - start
                output_dims[axis] = _utils.ceil_div_dim(sliced_extent, step)
            else:
                output_dims[axis] = ctx.new_symbolic_dim()
        elif isinstance(dim, int):
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
        else:
            # Symbolic dim with concrete start/end/step.
            #
            # Sentinel values (INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN) are
            # used by ONNX Slice to mean "up to the end" or "from the beginning"
            # depending on the sign of *step*.  When such sentinels appear with
            # a zero start (forward) or as both start and end (reverse), the
            # slice covers the entire axis and the output dim equals the input
            # dim — even when that dim is symbolic.
            #
            # For non-sentinel, non-negative start/end we can compute the slice
            # length directly as ``ceil((end - start) / step)`` because the
            # result does not depend on the actual (unknown) dimension size.
            # This assumes the tensor has at least ``end`` elements along the
            # axis, which is a practical assumption for well-formed models.
            sentinels = {2**63 - 1, -(2**63), 2**31 - 1, -(2**31)}

            # Full forward slice: start=0, step=1, end=sentinel → same dim
            if start == 0 and step == 1 and end in sentinels:
                output_dims[axis] = dim
            # Full reverse slice: start=sentinel, step=-1, end=negative sentinel
            elif step == -1 and start in sentinels and end in sentinels:
                output_dims[axis] = dim
            elif start >= 0 and end >= 0 and start not in sentinels and end not in sentinels:
                slice_len = max(0, -(-max(0, end - start) // abs(step)))
                output_dims[axis] = slice_len
            elif step > 0 and start >= 0 and end < 0 and end not in sentinels:
                sliced_extent = dim + end - start
                output_dims[axis] = _utils.ceil_div_dim(sliced_extent, step)
            else:
                output_dims[axis] = ctx.new_symbolic_dim()

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)

        # Propagate symbolic_value for 1-D tensors sliced on axis 0
        sym_val = ctx.get_symbolic_value(data)
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
