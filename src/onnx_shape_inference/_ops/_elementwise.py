# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for binary element-wise operators."""

from __future__ import annotations

__all__ = [
    "infer_binary_elementwise",
    "_infer_variadic_elementwise",
]


import operator as _operator
from collections.abc import Callable

import onnx_ir as ir

from onnx_shape_inference import _broadcast, _context, _registry

_reg = _registry.registry.register

_BinOp = Callable[[object, object], object | None]


def _broadcast_symbolic_values(
    op_func: _BinOp,
    val_a: list[int | ir.SymbolicDim] | None,
    val_b: list[int | ir.SymbolicDim] | None,
) -> list[int | ir.SymbolicDim] | None:
    """Apply ``op_func`` element-wise over two 1-D symbolic value lists.

    Mirrors NumPy broadcasting of two 1-D shape tensors: a length-1 operand is
    repeated to match the other.  Returns ``None`` (no propagation) when either
    operand is missing or the lengths are incompatible (both > 1 and unequal).
    """
    if val_a is None or val_b is None:
        return None
    la, lb = len(val_a), len(val_b)
    if la == lb:
        pairs = zip(val_a, val_b)
    elif la == 1:
        pairs = ((val_a[0], b) for b in val_b)
    elif lb == 1:
        pairs = ((a, val_b[0]) for a in val_a)
    else:
        return None
    result: list[int | ir.SymbolicDim] = []
    for a, b in pairs:
        value = op_func(a, b)
        if value is None:
            return None
        result.append(value)  # type: ignore[arg-type]
    return result


def _truncate_integer_division(
    dividend: object, divisor: object
) -> int | ir.SymbolicDim | None:
    """Divide concrete integers with ONNX's truncation-toward-zero semantics."""
    if not isinstance(dividend, int) or not isinstance(divisor, int) or divisor == 0:
        return None
    quotient = abs(dividend) // abs(divisor)
    return -quotient if (dividend < 0) != (divisor < 0) else quotient


def infer_binary_elementwise(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    *,
    output_dtype_override: ir.DataType | None = None,
) -> None:
    """Infer shape and dtype for a binary element-wise operator.

    Output shape is the broadcast of the two input shapes.
    Output dtype is ``output_dtype_override`` when given, otherwise the dtype of
    the first input that has one.
    """
    (input_a, input_b) = _context.check_inputs(node, "A", "B")

    output_shape = _broadcast.broadcast_shapes(input_a.shape, input_b.shape)
    output_dtype = output_dtype_override or input_a.dtype or input_b.dtype

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


_ARITH_OPS: dict[str, _BinOp] = {
    "Add": _operator.add,
    "Sub": _operator.sub,
    "Mul": _operator.mul,
    "Div": _truncate_integer_division,
}


# --- Arithmetic binary ops (output dtype = input dtype) ---


@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
@_reg("", "Mul", since_version=7)
@_reg("", "Div", since_version=7)
@_reg("", "Mod", since_version=10)
@_reg("", "Pow", since_version=7)
@_reg("", "BitShift", since_version=11)
@_reg("", "BitwiseAnd", since_version=18)
@_reg("", "BitwiseOr", since_version=18)
@_reg("", "BitwiseXor", since_version=18)
def _infer_arithmetic(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node)

    # Propagate symbolic_value for simple arithmetic on shape tensors
    op_func = _ARITH_OPS.get(node.op_type)
    if op_func is not None and len(node.outputs) > 0:
        val_a = ctx.get_symbolic_value(node.inputs[0])  # type: ignore[arg-type]
        val_b = ctx.get_symbolic_value(node.inputs[1])  # type: ignore[arg-type]
        result = _broadcast_symbolic_values(op_func, val_a, val_b)
        if result is not None:
            ctx.set_symbolic_value(node.outputs[0], result)


# --- Comparison ops (output dtype = BOOL) ---


@_reg("", "Equal", since_version=7)
@_reg("", "Less", since_version=7)
@_reg("", "Greater", since_version=7)
@_reg("", "LessOrEqual", since_version=12)
@_reg("", "GreaterOrEqual", since_version=12)
def _infer_comparison(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)


# --- Logical binary ops (output dtype = BOOL) ---


@_reg("", "And", since_version=7)
@_reg("", "Or", since_version=7)
@_reg("", "Xor", since_version=7)
def _infer_logical(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)


# --- String ops ---


@_reg("", "StringConcat", since_version=20)
def _infer_string_concat(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.STRING)


# --- Variadic elementwise ops (output dtype = first input dtype) ---


_VARIADIC_OPS: dict[str, Callable[..., object]] = {
    "Max": max,
    "Min": min,
    "Sum": _operator.add,
}
# ``Mean`` is intentionally absent: averaging is not an integer operation, so
# folding it over shape-tensor elements would not produce faithful values.


def _fold_variadic_element(
    ctx: _context.ShapeInferenceContext,
    op_type: str,
    op_func: Callable[..., object],
    a: int | ir.SymbolicDim,
    b: int | ir.SymbolicDim,
) -> int | ir.SymbolicDim:
    """Fold one element pair of a variadic op, degrading to an unknown dim.

    Concrete integer pairs are folded exactly.  Anything that cannot be folded
    soundly yields a fresh symbolic dim from ``ctx``, so a single unfoldable
    element degrades only itself instead of discarding the whole value.  This
    supersedes the previous early return, which abandoned propagation for every
    element as soon as one of them was symbolic.
    """
    if isinstance(a, int) and isinstance(b, int):
        return op_func(a, b)  # type: ignore[return-value]

    if op_type in ("Max", "Min"):
        # The ordering of a symbolic dim relative to anything else is unknown
        # (and builtin max/min would compare with > / <, which SymbolicDim does
        # not implement). The one sound refinement is max(d, d) == min(d, d) ==
        # d. Anonymous dims compare equal without denoting the same symbol, so
        # exclude them.
        if (
            isinstance(a, ir.SymbolicDim)
            and isinstance(b, ir.SymbolicDim)
            and a.value is not None
            and a == b
        ):
            return a
        return ctx.new_symbolic_dim()

    # Sum: SymbolicDim implements arithmetic, but stay defensive so that an
    # unexpected operand type degrades to "unknown" instead of raising.
    try:
        folded = op_func(a, b)
    except Exception:
        return ctx.new_symbolic_dim()
    if isinstance(folded, (int, ir.SymbolicDim)):
        return folded
    return ctx.new_symbolic_dim()


@_reg("", "Max", since_version=8)
@_reg("", "Mean", since_version=8)
@_reg("", "Min", since_version=8)
@_reg("", "Sum", since_version=8)
def _infer_variadic_elementwise(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for variadic elementwise operators.

    Output shape is the broadcast of all inputs. Output dtype is the first input's dtype.
    """
    if len(node.inputs) < 1:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    inputs = [v for v in node.inputs if v is not None]
    if not inputs:
        raise _context.OpUsageError(node, "Expected at least 1 non-None input")

    output_dtype = inputs[0].dtype
    output_shape = inputs[0].shape
    for inp in inputs[1:]:
        output_shape = _broadcast.broadcast_shapes(output_shape, inp.shape)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # Propagate symbolic values for Max/Min/Sum on shape tensors
    op_func = _VARIADIC_OPS.get(node.op_type)
    if op_func is None or not node.outputs or len(inputs) < 2:
        return

    sym_vals = [ctx.get_symbolic_value(v) for v in inputs]
    if any(sv is None for sv in sym_vals):
        return
    if len({len(sv) for sv in sym_vals}) != 1:  # type: ignore[arg-type]
        return

    result: list[int | ir.SymbolicDim] = list(sym_vals[0])  # type: ignore[arg-type]
    for sv in sym_vals[1:]:
        result = [
            _fold_variadic_element(ctx, node.op_type, op_func, a, b)
            for a, b in zip(result, sv)  # type: ignore[arg-type]
        ]
    ctx.set_symbolic_value(node.outputs[0], result)
