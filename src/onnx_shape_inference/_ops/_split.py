# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Split operator."""

from __future__ import annotations

__all__ = [
    "infer_split",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _utils


@_registry.registry.register("", "Split", since_version=2)
def infer_split(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Split operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Split.html
    """
    (data,) = _context.check_inputs(node, "input")

    input_shape = data.shape
    input_dtype = data.dtype
    num_outputs = len(node.outputs)

    if input_shape is None:
        for out in node.outputs:
            ctx.set_shape_and_dtype(out, None, input_dtype)
        return

    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else 0

    rank = input_shape.rank()
    if axis < 0:
        axis += rank

    # Read split sizes from input[1] (opset >= 13) or attribute
    split_sizes: list[int] | None = None
    if len(node.inputs) >= 2 and node.inputs[1] is not None:
        const = ir.convenience.get_const_tensor(node.inputs[1])
        if const is not None:
            split_sizes = [int(x) for x in const.numpy().flatten()]
    if split_sizes is None:
        split_attr = node.attributes.get("split")
        if split_attr is not None:
            split_sizes = list(split_attr.as_ints())

    if split_sizes is None:
        # Equal split. With a concrete axis dim we can compute exact integer
        # chunk sizes; with a symbolic axis dim we still know the relationship
        # (each of the first ``num_outputs - 1`` chunks is ``ceil(dim/n)`` and
        # the last is the remainder), so propagate that symbolically instead of
        # minting opaque fresh dims.
        dim = input_shape[axis]
        if num_outputs > 0:
            if isinstance(dim, int):
                base = dim // num_outputs
                remainder = dim % num_outputs
                split_sizes = [base + (1 if i < remainder else 0) for i in range(num_outputs)]
            else:
                chunk = _utils.ceil_div_dim(dim, num_outputs)
                last = dim - chunk * (num_outputs - 1)
                split_sizes = [chunk] * (num_outputs - 1) + [last]

    if split_sizes is not None:
        # Propagate symbolic values only when every chunk size is a concrete
        # int (slicing the known element list requires integer offsets).
        all_concrete = all(isinstance(s, int) for s in split_sizes)
        sym_val = (
            ctx.get_symbolic_value(data) if axis == 0 and rank == 1 and all_concrete else None
        )

        offset = 0
        for i, out in enumerate(node.outputs):
            if i < len(split_sizes):
                new_dims = list(input_shape.dims)
                new_dims[axis] = split_sizes[i]
                ctx.set_shape_and_dtype(out, ir.Shape(new_dims), input_dtype)

                if sym_val is not None:
                    size = split_sizes[i]
                    assert isinstance(size, int)
                    ctx.set_symbolic_value(out, sym_val[offset : offset + size])
                    offset += size
            else:
                ctx.set_shape_and_dtype(out, None, input_dtype)
    else:
        # Split sizes unknown — outputs have same rank, split axis is symbolic
        for out in node.outputs:
            new_dims = list(input_shape.dims)
            new_dims[axis] = ctx.new_symbolic_dim()
            ctx.set_shape_and_dtype(out, ir.Shape(new_dims), input_dtype)
