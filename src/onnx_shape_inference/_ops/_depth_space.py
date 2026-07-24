# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for DepthToSpace and SpaceToDepth operators."""

from __future__ import annotations

__all__ = [
    "infer_depth_to_space",
    "infer_space_to_depth",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry


def _validate_blocksize(
    ctx: _context.ShapeInferenceContext, node: ir.Node, blocksize: int
) -> bool:
    """Validate the positive blocksize required by depth/space transforms."""
    if blocksize <= 0:
        ctx.record_error(node, f"blocksize must be positive, got {blocksize}")
        return False
    return True


@_registry.registry.register("", "DepthToSpace", since_version=1)
def infer_depth_to_space(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DepthToSpace operator."""
    (x,) = _context.check_inputs(node, "input")

    blocksize_attr = _context.require_attr(node, "blocksize")
    b = blocksize_attr.as_int()
    if not _validate_blocksize(ctx, node, b):
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
        return

    output_shape: ir.Shape | None = None
    if x.shape is not None and x.shape.rank() == 4:
        n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if isinstance(c, int) and c % (b * b) != 0:
            ctx.record_error(
                node, f"DepthToSpace channel dimension {c} is not divisible by {b * b}"
            )
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
            return
        output_shape = ir.Shape([n, c // (b * b), h * b, w * b])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


@_registry.registry.register("", "SpaceToDepth", since_version=1)
def infer_space_to_depth(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for SpaceToDepth operator."""
    (x,) = _context.check_inputs(node, "input")

    blocksize_attr = _context.require_attr(node, "blocksize")
    b = blocksize_attr.as_int()
    if not _validate_blocksize(ctx, node, b):
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
        return

    output_shape: ir.Shape | None = None
    if x.shape is not None and x.shape.rank() == 4:
        n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if isinstance(h, int) and h % b != 0:
            ctx.record_error(
                node, f"SpaceToDepth height dimension {h} is not divisible by {b}"
            )
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
            return
        if isinstance(w, int) and w % b != 0:
            ctx.record_error(node, f"SpaceToDepth width dimension {w} is not divisible by {b}")
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, x.dtype)
            return
        output_shape = ir.Shape([n, c * b * b, h // b, w // b])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)
