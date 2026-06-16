# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for shape inference operators."""

from __future__ import annotations

__all__ = [
    "ceil_div_dim",
    "dim_product",
    "floor_div_dim",
    "normalize_axis",
]

import onnx_ir as ir


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
