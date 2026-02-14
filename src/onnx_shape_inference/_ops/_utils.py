# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for shape inference operators."""

from __future__ import annotations

__all__ = [
    "normalize_axis",
]


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
