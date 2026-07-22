# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for shape inference operators."""

from __future__ import annotations

__all__ = [
    "ceil_div_dim",
    "dim_product",
    "floor_div_dim",
    "get_known_dim_values",
    "get_known_scalar",
    "is_generated_dim",
    "max_dim",
    "min_dim",
    "normalize_axis",
    "scale_dim",
]

from fractions import Fraction

import onnx_ir as ir
import sympy

from onnx_shape_inference import _context, _symbolic_shapes


def get_known_dim_values(
    ctx: _context.ShapeInferenceContext, value: ir.Value | None
) -> list[int | ir.SymbolicDim] | None:
    """Read integer dimension values from a constant tensor or symbolic data."""
    if value is None:
        return None
    const = ir.convenience.get_const_tensor(value)
    if const is not None:
        return [int(item) for item in const.numpy().flatten()]
    symbolic_value = ctx.get_symbolic_value(value)
    return list(symbolic_value) if symbolic_value is not None else None


def get_known_scalar(
    ctx: _context.ShapeInferenceContext, value: ir.Value | None
) -> int | float | ir.SymbolicDim | None:
    """Read a numeric scalar from a constant tensor or symbolic data."""
    if value is None:
        return None
    const = ir.convenience.get_const_tensor(value)
    if const is not None:
        array = const.numpy()
        if array.size == 1:
            return array.item()
        return None
    symbolic_value = ctx.get_symbolic_value(value)
    if symbolic_value is not None and len(symbolic_value) == 1:
        return symbolic_value[0]
    return None


def is_generated_dim(dim: int | ir.SymbolicDim) -> bool:
    """Return whether a dimension is a fresh context-generated symbol."""
    if not isinstance(dim, ir.SymbolicDim) or dim.value is None:
        return False
    name = dim.value
    return name.startswith("_d") and name[2:].isdigit()


def max_dim(*dims: int | ir.SymbolicDim) -> int | ir.SymbolicDim | None:
    """Return the symbolic maximum, or ``None`` if any dimension is unknown."""
    try:
        expressions = [_symbolic_shapes.dim_to_expr(dim) for dim in dims]
        if any(expr is None for expr in expressions):
            return None
        return _symbolic_shapes.expr_to_dim(sympy.Max(*expressions))
    except (TypeError, ValueError):
        return None


def min_dim(*dims: int | ir.SymbolicDim) -> int | ir.SymbolicDim | None:
    """Return the symbolic minimum, or ``None`` if any dimension is unknown."""
    try:
        expressions = [_symbolic_shapes.dim_to_expr(dim) for dim in dims]
        if any(expr is None for expr in expressions):
            return None
        return _symbolic_shapes.expr_to_dim(sympy.Min(*expressions))
    except (TypeError, ValueError):
        return None


def dim_product(dims: list[int | ir.SymbolicDim]) -> int | ir.SymbolicDim:
    """Multiply a list of dimensions, keeping the result symbolic when needed.

    ``ir.SymbolicDim`` overloads arithmetic operators (backed by SymPy), so
    mixing concrete ``int`` dims with symbolic dims yields a simplified
    symbolic expression (e.g. ``[a, 2, b] -> 2*a*b``).  An empty list returns
    ``1``.
    """
    result: int | ir.SymbolicDim = 1
    for d in dims:
        result = result * d
    return result


def floor_div_dim(
    numerator: int | ir.SymbolicDim,
    denominator: int | ir.SymbolicDim,
) -> int | ir.SymbolicDim:
    """Integer-divide two (possibly symbolic) dimensions with cancellation.

    Common symbolic factors cancel via SymPy (e.g. ``(a*b*c) // (a*b*2)`` →
    ``floor(c/2)``).  When both are concrete, returns a plain ``int``.
    """
    if isinstance(numerator, int) and isinstance(denominator, int):
        return numerator // denominator
    if isinstance(numerator, int) and isinstance(denominator, ir.SymbolicDim):
        # ``int.__floordiv__(SymbolicDim)`` is unsupported; compute
        # ``floor(numerator / denominator)`` via the symbolic true-division.
        return (numerator / denominator) // 1
    return numerator // denominator


def ceil_div_dim(
    numerator: int | ir.SymbolicDim,
    denominator: int,
) -> int | ir.SymbolicDim:
    """Ceiling-divide a (possibly symbolic) dimension by a positive integer.

    Uses the identity ``ceil(x / n) == -((-x) // n)`` so the result stays a
    well-formed symbolic expression that SymPy can simplify (e.g. an even
    ``(2*b + 2*c)`` divided by ``2`` collapses to ``b + c``).
    """
    if isinstance(numerator, int):
        return -((-numerator) // denominator)
    return -((-numerator) // denominator)


def scale_dim(dim: int | ir.SymbolicDim, scale: float) -> int | ir.SymbolicDim:
    """Scale a dimension with exact rational arithmetic before flooring."""
    ratio = Fraction(str(scale))
    return floor_div_dim(dim * ratio.numerator, ratio.denominator)


def normalize_axis(axis: int, rank: int) -> int:
    """Normalize a potentially negative axis to a non-negative value.

    Args:
        axis: The axis value, can be negative.
        rank: The tensor rank.

    Returns:
        The normalized axis in range [0, rank).

    Raises:
        ValueError: If axis is out of range [-rank, rank).
    """
    if axis < -rank or axis >= rank:
        raise ValueError(f"axis {axis} is out of range for rank {rank}")
    if axis < 0:
        axis += rank
    return axis
