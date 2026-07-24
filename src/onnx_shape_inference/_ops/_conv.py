# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Conv, ConvInteger, and QLinearConv operators."""

from __future__ import annotations

__all__ = [
    "infer_causal_conv_with_state",
    "infer_conv",
    "infer_conv_integer",
    "infer_qlinear_conv",
]

import math

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


def _compute_conv_output_dim(
    in_dim: int | ir.SymbolicDim,
    kernel: int,
    stride: int,
    dilation: int,
    pad_begin: int,
    pad_end: int,
) -> int | ir.SymbolicDim:
    """Compute the output spatial dimension for Conv with explicit padding."""
    effective_kernel = dilation * (kernel - 1) + 1
    return (in_dim + pad_begin + pad_end - effective_kernel) // stride + 1


def _compute_auto_pad_output_dim(
    in_dim: int | ir.SymbolicDim,
    stride: int,
    auto_pad: str,
) -> int | ir.SymbolicDim:
    """Compute the output spatial dimension for Conv with auto_pad.

    SAME_UPPER/SAME_LOWER: output_dim = ceil(in_dim / stride)
    VALID: output with no padding (handled by caller).
    """
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        return math.ceil(in_dim / stride)  # type: ignore[return-value]
    # Should not be called for NOTSET or VALID
    raise ValueError(f"Unexpected auto_pad value: {auto_pad}")


@_registry.registry.register("", "Conv", since_version=1)
def infer_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Conv operator.

    Supports explicit ``pads`` and ``auto_pad`` (VALID, SAME_UPPER, SAME_LOWER).

    Spec: https://onnx.ai/onnx/operators/onnx__Conv.html
    """
    (x, w) = _context.check_inputs(node, "X", "W")

    # Validate rank before computing shape
    if x.shape is not None and x.shape.rank() < 3:
        ctx.record_error(node, f"Conv input must be at least rank 3, got {x.shape.rank()}")
        return

    output_shape = _compute_conv_shape(ctx, node, x, w)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


def _compute_conv_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    x: ir.Value,
    w: ir.Value,
) -> ir.Shape | None:
    """Compute the output shape for a Conv-like operator given X and W inputs."""
    x_shape = x.shape
    w_shape = w.shape

    if x_shape is None or w_shape is None:
        return None

    x_rank = x_shape.rank()
    if x_rank < 3:
        ctx.record_error(node, f"Conv input must be at least rank 3, got {x_rank}")
        return None

    n_spatial = x_rank - 2

    kernel_shape_attr = node.attributes.get("kernel_shape")
    if kernel_shape_attr is not None:
        kernel_shape: list[int | None] = list(kernel_shape_attr.as_ints())
    elif w_shape.rank() == x_rank:
        kernel_shape = [
            w_shape[i + 2] if isinstance(w_shape[i + 2], int) else None  # type: ignore[misc]
            for i in range(n_spatial)
        ]
    else:
        ctx.record_error(
            node, f"Conv weight rank {w_shape.rank()} does not match input rank {x_rank}"
        )
        return None

    strides_attr = node.attributes.get("strides")
    strides = list(strides_attr.as_ints()) if strides_attr is not None else [1] * n_spatial

    dilations_attr = node.attributes.get("dilations")
    dilations = (
        list(dilations_attr.as_ints()) if dilations_attr is not None else [1] * n_spatial
    )

    auto_pad_attr = node.attributes.get("auto_pad")
    auto_pad = auto_pad_attr.as_string() if auto_pad_attr is not None else "NOTSET"

    pads_attr = node.attributes.get("pads")
    if auto_pad == "NOTSET":
        if pads_attr is not None:
            pads = list(pads_attr.as_ints())
        else:
            pads = [0] * (2 * n_spatial)
    else:
        pads = None

    if len(kernel_shape) != n_spatial:
        ctx.record_error(
            node,
            f"Conv kernel_shape length {len(kernel_shape)} does not match spatial rank {n_spatial}",
        )
        return None
    if len(strides) != n_spatial or any(stride <= 0 for stride in strides):
        ctx.record_error(
            node, "Conv strides must contain one positive value per spatial dimension"
        )
        return None
    if len(dilations) != n_spatial or any(dilation <= 0 for dilation in dilations):
        ctx.record_error(
            node, "Conv dilations must contain one positive value per spatial dimension"
        )
        return None
    if pads is not None and len(pads) != 2 * n_spatial:
        ctx.record_error(
            node, f"Conv pads length {len(pads)} does not match spatial rank {n_spatial}"
        )
        return None

    batch_dim = x_shape[0]
    out_channels = w_shape[0]

    spatial_dims: list[int | ir.SymbolicDim] = []
    for i in range(n_spatial):
        in_dim = x_shape[i + 2]
        k = kernel_shape[i]
        s = strides[i]
        d = dilations[i]

        if k is None:
            spatial_dims.append(ctx.new_symbolic_dim())
            continue

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out_dim = _compute_auto_pad_output_dim(in_dim, s, auto_pad)
        elif auto_pad == "VALID":
            out_dim = _compute_conv_output_dim(in_dim, k, s, d, 0, 0)
        else:
            assert pads is not None
            out_dim = _compute_conv_output_dim(in_dim, k, s, d, pads[i], pads[i + n_spatial])
        spatial_dims.append(out_dim)

    output_dims: list[int | ir.SymbolicDim] = [batch_dim, out_channels, *spatial_dims]
    return ir.Shape(output_dims)


@_registry.registry.register("", "ConvInteger", since_version=10)
def infer_conv_integer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ConvInteger operator.

    Same shape logic as Conv, but output dtype is INT32.

    Spec: https://onnx.ai/onnx/operators/onnx__ConvInteger.html
    """
    (x, w) = _context.check_inputs(node, "X", "W")

    output_shape = _compute_conv_shape(ctx, node, x, w)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, ir.DataType.INT32)


@_registry.registry.register("", "QLinearConv", since_version=10)
def infer_qlinear_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for QLinearConv operator.

    Inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp
    Shape = conv(x, w). Output dtype from y_zp (input 7) if available, else UINT8.

    Spec: https://onnx.ai/onnx/operators/onnx__QLinearConv.html
    """
    # x is input[0], w is input[3]
    if len(node.inputs) < 8:
        raise _context.OpUsageError(
            node, f"Expected at least 8 inputs, got {len(node.inputs)}"
        )
    x = node.inputs[0]
    w = node.inputs[3]
    if x is None:
        raise _context.OpUsageError(node, "Required input 'x' (#0) is None")
    if w is None:
        raise _context.OpUsageError(node, "Required input 'w' (#3) is None")

    output_dtype = ir.DataType.UINT8
    if len(node.inputs) > 7 and node.inputs[7] is not None:
        y_zp_dtype = node.inputs[7].dtype
        if y_zp_dtype is not None:
            output_dtype = y_zp_dtype

    output_shape = _compute_conv_shape(ctx, node, x, w)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_registry.registry.register("", "CausalConvWithState", since_version=27)
def infer_causal_conv_with_state(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for CausalConvWithState operator.

    Spec: https://onnx.ai/onnx/operators/onnx__CausalConvWithState.html

    A stateful causal depthwise 1-D convolution.  ``input``, ``output`` and the
    carry ``present_state`` are rank-3 ``[B, C, L]`` / ``[B, C, k-1]`` where
    ``k`` is the kernel size from ``weight`` of shape ``(C, 1, k)``:

    * ``output``        same shape as ``input``.
    * ``present_state`` ``[B, C, k - 1]`` (the last ``k-1`` positions kept for
      incremental decode).
    """
    (inp, weight) = _context.check_inputs(node, "input", "weight")
    output_dtype = inp.dtype

    # Output 0: same shape as input.
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], inp.shape, output_dtype)

    # Output 1: present_state [B, C, k-1].
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        present_shape: ir.Shape | None = None
        if inp.shape is not None and inp.shape.rank() == 3:
            kernel = (
                weight.shape[2]
                if weight.shape is not None and weight.shape.rank() == 3
                else None
            )
            if isinstance(kernel, int):
                state_len: int | ir.SymbolicDim = max(kernel - 1, 0)
            else:
                state_len = ctx.new_symbolic_dim()
            present_shape = ir.Shape([inp.shape[0], inp.shape[1], state_len])
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, output_dtype)
