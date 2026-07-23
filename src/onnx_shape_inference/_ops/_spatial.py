# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for spatial operators."""

from __future__ import annotations

__all__ = [
    "infer_affine_grid",
    "infer_center_crop_pad",
    "infer_col2im",
    "infer_deform_conv",
    "infer_grid_sample",
    "infer_max_roi_pool",
    "infer_max_unpool",
    "infer_non_max_suppression",
    "infer_roi_align",
]

import math

import onnx_ir as ir

from onnx_shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "AffineGrid", since_version=20)
def infer_affine_grid(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for AffineGrid operator.

    Spec: https://onnx.ai/onnx/operators/onnx__AffineGrid.html
    """
    (theta, size) = _context.check_inputs(node, "theta", "size")

    output_dtype = theta.dtype

    size_const = ir.convenience.get_const_tensor(size)
    if size_const is not None:
        size_vals = [int(s) for s in size_const.numpy().flatten()]
        n = size_vals[0]
        if len(size_vals) == 4:
            # 2D: output = [N, H, W, 2]
            output_shape = ir.Shape([n, size_vals[2], size_vals[3], 2])
        elif len(size_vals) == 5:
            # 3D: output = [N, D, H, W, 3]
            output_shape = ir.Shape([n, size_vals[2], size_vals[3], size_vals[4], 3])
        else:
            output_shape = None
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "GridSample", since_version=16)
def infer_grid_sample(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for GridSample operator.

    Output shape: [N, C, D1_out, D2_out, ...] where N,C from X, spatial from grid.

    Spec: https://onnx.ai/onnx/operators/onnx__GridSample.html
    """
    (x, grid) = _context.check_inputs(node, "X", "grid")

    output_dtype = x.dtype

    if x.shape is not None and grid.shape is not None:
        n = x.shape[0]
        c = x.shape[1]
        spatial_dims: list[int | ir.SymbolicDim] = list(grid.shape.dims[1:-1])
        output_dims: list[int | ir.SymbolicDim] = [n, c, *spatial_dims]
        output_shape: ir.Shape | None = ir.Shape(output_dims)
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "Col2Im", since_version=18)
def infer_col2im(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Col2Im operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Col2Im.html
    """
    (input_val, image_shape, block_shape) = _context.check_inputs(
        node, "input", "image_shape", "block_shape"
    )

    output_dtype = input_val.dtype

    image_shape_const = ir.convenience.get_const_tensor(image_shape)
    block_shape_const = ir.convenience.get_const_tensor(block_shape)
    if input_val.shape is not None and image_shape_const is not None:
        image_dims = [int(d) for d in image_shape_const.numpy().flatten()]
        n = input_val.shape[0]
        c: int | ir.SymbolicDim | None = None
        if input_val.shape.rank() > 1 and block_shape_const is not None:
            block_dims = [int(d) for d in block_shape_const.numpy().flatten()]
            prod_block = math.prod(block_dims)
            if prod_block > 0:
                c = input_val.shape[1] // prod_block
        if c is None:
            c = ctx.new_symbolic_dim()
        output_dims: list[int | ir.SymbolicDim] = [n, c, *image_dims]
        output_shape: ir.Shape | None = ir.Shape(output_dims)
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "CenterCropPad", since_version=18)
def infer_center_crop_pad(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for CenterCropPad operator.

    Spec: https://onnx.ai/onnx/operators/onnx__CenterCropPad.html
    """
    (input_data, shape) = _context.check_inputs(node, "input_data", "shape")

    output_dtype = input_data.dtype

    shape_const = ir.convenience.get_const_tensor(shape)
    if input_data.shape is not None and shape_const is not None:
        shape_vals = [int(s) for s in shape_const.numpy().flatten()]
        input_dims = list(input_data.shape.dims)
        rank = input_data.shape.rank()
        axes_attr = node.attributes.get("axes")
        axes = list(axes_attr.as_ints()) if axes_attr is not None else list(range(rank))
        if len(axes) != len(shape_vals):
            ctx.record_error(
                node,
                f"CenterCropPad axes length {len(axes)} does not match shape length {len(shape_vals)}",
            )
            return
        normalized_axes: list[int] = []
        for axis in axes:
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank or axis in normalized_axes:
                ctx.record_error(node, f"CenterCropPad axis {axis} is invalid for rank {rank}")
                return
            normalized_axes.append(axis)
        output_dims: list[int | ir.SymbolicDim] = list(input_dims)
        for axis, target_dim in zip(normalized_axes, shape_vals, strict=True):
            output_dims[axis] = target_dim
        output_shape: ir.Shape | None = ir.Shape(output_dims)
    elif input_data.shape is not None:
        rank = input_data.shape.rank()
        axes_attr = node.attributes.get("axes")
        axes = list(axes_attr.as_ints()) if axes_attr is not None else list(range(rank))
        output_dims_sym: list[int | ir.SymbolicDim] = list(input_data.shape.dims)
        for axis in axes:
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank:
                ctx.record_error(node, f"CenterCropPad axis {axis} is invalid for rank {rank}")
                return
            output_dims_sym[axis] = ctx.new_symbolic_dim()
        output_shape = ir.Shape(output_dims_sym)
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "RoiAlign", since_version=10)
def infer_roi_align(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RoiAlign operator.

    Output: [num_rois, C, output_height, output_width]

    Spec: https://onnx.ai/onnx/operators/onnx__RoiAlign.html
    """
    (x, rois, _batch_indices) = _context.check_inputs(node, "X", "rois", "batch_indices")

    output_dtype = x.dtype

    output_height_attr = node.attributes.get("output_height")
    output_height = output_height_attr.as_int() if output_height_attr is not None else 1

    output_width_attr = node.attributes.get("output_width")
    output_width = output_width_attr.as_int() if output_width_attr is not None else 1

    num_rois: int | ir.SymbolicDim
    if rois.shape is not None and isinstance(rois.shape[0], int):
        num_rois = rois.shape[0]
    else:
        num_rois = ctx.new_symbolic_dim()

    c: int | ir.SymbolicDim
    if x.shape is not None and isinstance(x.shape[1], int):
        c = x.shape[1]
    else:
        c = ctx.new_symbolic_dim()

    output_shape = ir.Shape([num_rois, c, output_height, output_width])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "MaxRoiPool", since_version=1)
def infer_max_roi_pool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MaxRoiPool operator.

    Output: [num_rois, C, pooled_shape[0], pooled_shape[1]]

    Spec: https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html
    """
    (x, rois) = _context.check_inputs(node, "X", "rois")

    output_dtype = x.dtype

    pooled_shape_attr = _context.require_attr(node, "pooled_shape")
    pooled_shape = list(pooled_shape_attr.as_ints())

    num_rois: int | ir.SymbolicDim
    if rois.shape is not None and isinstance(rois.shape[0], int):
        num_rois = rois.shape[0]
    else:
        num_rois = ctx.new_symbolic_dim()

    c: int | ir.SymbolicDim
    if x.shape is not None and isinstance(x.shape[1], int):
        c = x.shape[1]
    else:
        c = ctx.new_symbolic_dim()

    output_shape = ir.Shape([num_rois, c, pooled_shape[0], pooled_shape[1]])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg("", "MaxUnpool", since_version=9)
def infer_max_unpool(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MaxUnpool operator.

    Spec: https://onnx.ai/onnx/operators/onnx__MaxUnpool.html
    """
    (x, _i) = _context.check_inputs(node, "X", "I")

    output_dtype = x.dtype

    # Check for output_shape input (index 2)
    if len(node.inputs) > 2 and node.inputs[2] is not None:
        os_const = ir.convenience.get_const_tensor(node.inputs[2])
        if os_const is not None:
            output_dims = [int(d) for d in os_const.numpy().flatten()]
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)
            return

    if x.shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    # Create symbolic spatial dims
    output_dims_list: list[int | ir.SymbolicDim] = [x.shape[0], x.shape[1]]
    for _ in range(x.shape.rank() - 2):
        output_dims_list.append(ctx.new_symbolic_dim())
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims_list), output_dtype)


@_reg("", "DeformConv", since_version=19)
def infer_deform_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DeformConv operator.

    Same output shape logic as standard Conv.

    Spec: https://onnx.ai/onnx/operators/onnx__DeformConv.html
    """
    (x, w, _offset) = _context.check_inputs(node, "X", "W", "offset")

    # Reuse Conv shape computation by delegating to the conv logic
    # We need to compute the shape the same way Conv does
    output_dtype = x.dtype
    x_shape = x.shape
    w_shape = w.shape

    if x_shape is None or w_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    x_rank = x_shape.rank()
    if x_rank < 3:
        ctx.record_error(node, f"DeformConv input must be at least rank 3, got {x_rank}")
        return

    n_spatial = x_rank - 2

    kernel_shape_attr = node.attributes.get("kernel_shape")
    if kernel_shape_attr is not None:
        kernel_shape: list[int | None] = list(kernel_shape_attr.as_ints())
    elif w_shape.rank() > 2:
        kernel_shape = [
            w_shape[i + 2] if isinstance(w_shape[i + 2], int) else None  # type: ignore[misc]
            for i in range(n_spatial)
        ]
    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    pads_attr = node.attributes.get("pads")
    pads = list(pads_attr.as_ints()) if pads_attr is not None else [0] * (2 * n_spatial)

    batch_dim = x_shape[0]
    out_channels = w_shape[0]

    from onnx_shape_inference._ops._conv import _compute_conv_output_dim

    spatial_dims: list[int | ir.SymbolicDim] = []
    for i in range(n_spatial):
        in_dim = x_shape[i + 2]
        k = kernel_shape[i]
        s = strides[i]
        d = dilations[i]

        if k is None:
            spatial_dims.append(ctx.new_symbolic_dim())
            continue

        out_dim = _compute_conv_output_dim(in_dim, k, s, d, pads[i], pads[i + n_spatial])
        spatial_dims.append(out_dim)

    output_dims_final: list[int | ir.SymbolicDim] = [batch_dim, out_channels, *spatial_dims]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims_final), output_dtype)


@_registry.registry.register("", "NonMaxSuppression", since_version=10)
def infer_non_max_suppression(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for NonMaxSuppression operator.

    Output: [selected_indices_count, 3], dtype=INT64.
    """
    _context.check_inputs(node, "boxes", "scores")

    output_shape = ir.Shape([ctx.new_symbolic_dim(), 3])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT64)
