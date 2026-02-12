# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Pad operator."""

from __future__ import annotations

__all__ = [
    "infer_pad",
    "infer_pad_v11",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


def _apply_pads(
    ctx: _context.ShapeInferenceContext,
    data_shape: ir.Shape,
    pads: list[int] | None,
    axes: list[int] | None,
) -> ir.Shape:
    """Compute the padded output shape.

    Args:
        ctx: Inference context (for creating symbolic dims).
        data_shape: Shape of the data input.
        pads: Flat pad values ``[begin_0, …, begin_r, end_0, …, end_r]``,
            or ``None`` if unknown.
        axes: Optional list of axes the pads apply to.

    Returns:
        The output shape with padding applied.
    """
    rank = data_shape.rank()
    if pads is None:
        return ir.Shape([ctx.new_symbolic_dim() for _ in range(rank)])

    new_dims: list[int | ir.SymbolicDim] = list(data_shape.dims)
    if axes is not None:
        n_axes = len(axes)
        for i, ax in enumerate(axes):
            if ax < 0:
                ax += rank
            new_dims[ax] = data_shape[ax] + pads[i] + pads[i + n_axes]
    else:
        for i in range(rank):
            new_dims[i] = data_shape[i] + pads[i] + pads[i + rank]
    return ir.Shape(new_dims)


@_registry.registry.register("", "Pad", since_version=1)
def infer_pad(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Pad operator (opset < 11).

    In opset 1-10, ``pads`` is a required attribute.
    """
    (data,) = _context.check_inputs(node, "data")

    output_shape: ir.Shape | None = None
    if data.shape is not None:
        pads_attr = _context.require_attr(node, "pads")
        pads = [int(x) for x in pads_attr.as_ints()]
        output_shape = _apply_pads(ctx, data.shape, pads, axes=None)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, data.dtype)


@_registry.registry.register("", "Pad", since_version=11)
def infer_pad_v11(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Pad operator (opset 11+).

    In opset 11+, ``pads`` is a required input (input 1).
    """
    (data, pads_value) = _context.check_inputs(node, "data", "pads")

    output_shape: ir.Shape | None = None
    if data.shape is not None:
        pads: list[int] | None = None
        pads_const = ir.convenience.get_const_tensor(pads_value)
        if pads_const is not None:
            pads = [int(x) for x in pads_const.numpy().flatten()]

        # Read optional axes input (input[3])
        axes: list[int] | None = None
        if len(node.inputs) > 3 and node.inputs[3] is not None:
            axes_const = ir.convenience.get_const_tensor(node.inputs[3])
            if axes_const is not None:
                axes = [int(a) for a in axes_const.numpy().flatten()]

        output_shape = _apply_pads(ctx, data.shape, pads, axes)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, data.dtype)
