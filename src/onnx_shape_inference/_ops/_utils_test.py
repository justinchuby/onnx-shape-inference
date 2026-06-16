# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared symbolic-arithmetic helpers in ``_utils``."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference._ops import _utils


class DimProductTest(unittest.TestCase):
    def test_empty_is_one(self):
        self.assertEqual(_utils.dim_product([]), 1)

    def test_all_concrete(self):
        self.assertEqual(_utils.dim_product([2, 3, 4]), 24)

    def test_mixed_symbolic(self):
        a = ir.SymbolicDim("a")
        b = ir.SymbolicDim("b")
        result = _utils.dim_product([a, 2, b])
        self.assertIsInstance(result, ir.SymbolicDim)
        self.assertEqual(str(result), "2*a*b")


class FloorDivDimTest(unittest.TestCase):
    def test_both_concrete(self):
        self.assertEqual(_utils.floor_div_dim(7, 2), 3)

    def test_symbolic_numerator(self):
        a = ir.SymbolicDim("a")
        result = _utils.floor_div_dim(a * 6, 2)
        self.assertEqual(str(result), "3*a")

    def test_int_numerator_symbolic_denominator(self):
        """``int // SymbolicDim`` routes through symbolic true-division + floor.

        Native ``int.__floordiv__`` does not accept a SymbolicDim, so the helper
        computes ``floor(numerator / denominator)`` symbolically.
        """
        d = ir.SymbolicDim("d")
        result = _utils.floor_div_dim(12, d)
        self.assertIsInstance(result, ir.SymbolicDim)
        self.assertEqual(str(result), "floor(12/d)")

    def test_symbolic_cancellation(self):
        a = ir.SymbolicDim("a")
        b = ir.SymbolicDim("b")
        c = ir.SymbolicDim("c")
        # (a*b*c) // (a*b) cancels to c
        result = _utils.floor_div_dim(a * b * c, a * b)
        self.assertEqual(str(result), "c")


class CeilDivDimTest(unittest.TestCase):
    def test_both_concrete_exact(self):
        self.assertEqual(_utils.ceil_div_dim(8, 2), 4)

    def test_both_concrete_rounds_up(self):
        self.assertEqual(_utils.ceil_div_dim(7, 2), 4)

    def test_symbolic_even_collapses(self):
        b = ir.SymbolicDim("b")
        c = ir.SymbolicDim("c")
        # ceil((2*b + 2*c) / 2) == b + c exactly
        result = _utils.ceil_div_dim(2 * b + 2 * c, 2)
        self.assertEqual(str(result), "b + c")


class NormalizeAxisTest(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(_utils.normalize_axis(1, 3), 1)

    def test_negative(self):
        self.assertEqual(_utils.normalize_axis(-1, 3), 2)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _utils.normalize_axis(3, 3)


if __name__ == "__main__":
    unittest.main()
