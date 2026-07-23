# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for MatMul operator."""

from __future__ import annotations

__all__ = [
    "infer_matmul",
    "infer_matmul_integer",
    "infer_qlinear_matmul",
]

import onnx_ir as ir

from onnx_shape_inference import _broadcast, _context, _registry


def _is_generated_dim(ctx: _context.ShapeInferenceContext, dim: int | ir.SymbolicDim) -> bool:
    """Return whether the context minted this symbolic dimension."""
    return (
        isinstance(dim, ir.SymbolicDim)
        and dim.value is not None
        and ctx.is_generated_symbol(dim.value)
    )


def _shape_may_be_incomplete(value: ir.Value | None) -> bool:
    """Return whether control-flow merging may have supplied only a branch shape."""
    if value is None:
        return False
    producer = value.producer()
    return producer is not None and producer.op_type in {"If", "Loop", "Scan"}


def _resolve_symbolic_batch_dims(
    ctx: _context.ShapeInferenceContext,
    batch_shape: ir.Shape,
    batch_a: ir.Shape,
    batch_b: ir.Shape,
) -> ir.Shape:
    """Resolve differing symbolic batch dims without assuming broadcast equality."""
    dims_a = [1] * (batch_shape.rank() - batch_a.rank()) + list(batch_a.dims)
    dims_b = [1] * (batch_shape.rank() - batch_b.rank()) + list(batch_b.dims)
    output_dims = list(batch_shape.dims)
    equalities = {frozenset((left, right)) for left, right in ctx.symbolic_equalities}
    for i, (dim_a, dim_b) in enumerate(zip(dims_a, dims_b)):
        if not isinstance(dim_a, ir.SymbolicDim) or not isinstance(dim_b, ir.SymbolicDim):
            continue
        if dim_a == dim_b:
            continue
        equality_recorded = frozenset((str(dim_a), str(dim_b))) in equalities
        if equality_recorded:
            if _is_generated_dim(ctx, dim_a) and not _is_generated_dim(ctx, dim_b):
                output_dims[i] = dim_b
            elif _is_generated_dim(ctx, dim_b) and not _is_generated_dim(ctx, dim_a):
                output_dims[i] = dim_a
        else:
            output_dims[i] = ctx.new_symbolic_dim()
    return ir.Shape(output_dims)


def _matmul_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    shape_a: ir.Shape | None,
    shape_b: ir.Shape | None,
    output_dtype: ir.DataType | None,
    *,
    validate_contraction_dims: bool = True,
) -> None:
    """Compute matmul output shape and set on the first output."""
    if shape_a is None or shape_b is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    rank_a = shape_a.rank()
    rank_b = shape_b.rank()
    k_dim_a = shape_a.dims[-1]
    k_dim_b = shape_b.dims[0] if rank_b == 1 else shape_b.dims[-2]
    contraction_mismatch = (
        validate_contraction_dims
        and isinstance(k_dim_a, int)
        and isinstance(k_dim_b, int)
        and k_dim_a != k_dim_b
    )

    # Handle 1-D cases
    if rank_a == 1 and rank_b == 1:
        # Dot product -> scalar
        output_shape = ir.Shape([])
    elif rank_a == 1:
        # (K,) x (..., K, N) -> (..., N)
        output_dims = [*shape_b.dims[:-2], shape_b.dims[-1]]
        output_shape = ir.Shape(output_dims)
    elif rank_b == 1:
        # (..., M, K) x (K,) -> (..., M)
        output_dims = list(shape_a.dims[:-1])
        output_shape = ir.Shape(output_dims)
    else:
        # (..., M, K) x (..., K, N) -> (..., M, N)
        # Broadcast batch dimensions
        batch_a = ir.Shape(list(shape_a.dims[:-2])) if rank_a > 2 else ir.Shape([])
        batch_b = ir.Shape(list(shape_b.dims[:-2])) if rank_b > 2 else ir.Shape([])
        batch_shape = _broadcast.broadcast_shapes(batch_a, batch_b)
        if batch_shape is not None:
            batch_shape = _resolve_symbolic_batch_dims(ctx, batch_shape, batch_a, batch_b)

        m_dim = shape_a.dims[-2]
        n_dim = shape_b.dims[-1]

        if batch_shape is not None:
            output_dims = [*batch_shape.dims, m_dim, n_dim]
        elif batch_a.rank() == 0 and batch_b.rank() == 0:
            # Both inputs are 2-D, no batch dims to broadcast
            output_dims = [m_dim, n_dim]
        else:
            # Incompatible batch shapes
            ctx.record_error(
                node,
                f"Incompatible batch dimensions for MatMul: {batch_a} vs {batch_b}",
            )
            return
        output_shape = ir.Shape(output_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
    if contraction_mismatch:
        ctx.record_error(
            node,
            f"Incompatible MatMul contraction dimensions: {k_dim_a} vs {k_dim_b}",
        )


@_registry.registry.register("", "MatMul", since_version=1)
def infer_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MatMul operator.

    Follows NumPy matmul semantics:
    - 1-D x 1-D: dot product -> scalar
    - 2-D x 2-D: matrix multiply -> (M, N)
    - Broadcast batch dims for higher-rank inputs

    Spec: https://onnx.ai/onnx/operators/onnx__MatMul.html
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")
    output_dtype = input_a.dtype or input_b.dtype
    _matmul_shape(
        ctx,
        node,
        input_a.shape,
        input_b.shape,
        output_dtype,
        validate_contraction_dims=not (
            _shape_may_be_incomplete(input_a) or _shape_may_be_incomplete(input_b)
        ),
    )


@_registry.registry.register("", "MatMulInteger", since_version=10)
def infer_matmul_integer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for MatMulInteger operator.

    Same shape logic as MatMul but output dtype is INT32.
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")
    _matmul_shape(
        ctx,
        node,
        input_a.shape,
        input_b.shape,
        ir.DataType.INT32,
        validate_contraction_dims=not (
            _shape_may_be_incomplete(input_a) or _shape_may_be_incomplete(input_b)
        ),
    )


@_registry.registry.register("", "QLinearMatMul", since_version=10)
def infer_qlinear_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for QLinearMatMul operator.

    Inputs: a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point.
    Shape = matmul(a, b), dtype from y_zero_point (input 7).
    """
    (a,) = _context.check_inputs(node, "a")
    # b is input 3
    output_dtype = ir.DataType.UINT8
    if len(node.inputs) > 7 and node.inputs[7] is not None:
        zp_dtype = node.inputs[7].dtype
        if zp_dtype is not None:
            output_dtype = zp_dtype
    shape_b = (
        node.inputs[3].shape if len(node.inputs) > 3 and node.inputs[3] is not None else None
    )
    b = node.inputs[3] if len(node.inputs) > 3 else None
    _matmul_shape(
        ctx,
        node,
        a.shape,
        shape_b,
        output_dtype,
        validate_contraction_dims=not (
            _shape_may_be_incomplete(a) or _shape_may_be_incomplete(b)
        ),
    )
