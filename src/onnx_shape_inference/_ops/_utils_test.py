# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared symbolic-arithmetic helpers in ``_utils``."""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

from onnx_shape_inference import _context
from onnx_shape_inference._ops import _utils


class KnownValueTest(unittest.TestCase):
    def setUp(self):
        self.ctx = _context.ShapeInferenceContext({"": 21})

    def test_dim_values_from_constant(self):
        value = ir.Value(
            name="shape",
            const_value=ir.Tensor(np.array([2, 3], dtype=np.int64)),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([2]),
        )
        self.assertEqual(_utils.get_known_dim_values(self.ctx, value), [2, 3])

    def test_dim_values_from_symbolic_data(self):
        value = ir.Value(
            name="shape",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([2]),
        )
        self.ctx.set_symbolic_value(value, [ir.SymbolicDim("N"), 3])
        self.assertEqual(
            _utils.get_known_dim_values(self.ctx, value),
            [ir.SymbolicDim("N"), 3],
        )

    def test_dim_values_unknown(self):
        value = ir.Value(name="shape", type=ir.TensorType(ir.DataType.INT64))
        self.assertIsNone(_utils.get_known_dim_values(self.ctx, value))
        self.assertIsNone(_utils.get_known_dim_values(self.ctx, None))

    @parameterized.parameterized.expand(
        [
            ("int", np.array(3, dtype=np.int64), 3),
            ("float", np.array(0.5, dtype=np.float32), 0.5),
        ]
    )
    def test_scalar_from_constant(self, _name, array, expected):
        value = ir.Value(name="value", const_value=ir.Tensor(array), shape=ir.Shape([]))
        self.assertEqual(_utils.get_known_scalar(self.ctx, value), expected)

    def test_scalar_rejects_non_scalar_constant(self):
        value = ir.Value(
            name="value",
            const_value=ir.Tensor(np.array([1, 2], dtype=np.int64)),
            shape=ir.Shape([2]),
        )
        self.assertIsNone(_utils.get_known_scalar(self.ctx, value))


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


class ScaleDimTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("concrete_down", 5, 0.5, 2),
            ("concrete_up", 5, 2.0, 10),
            ("symbolic_down", ir.SymbolicDim("H"), 0.5, "floor(H/2)"),
            ("symbolic_simplifies", ir.SymbolicDim("2*h"), 0.5, "h"),
            ("identity", ir.SymbolicDim("N"), 1.0, "N"),
        ]
    )
    def test_scale(self, _name, dim, scale, expected):
        result = _utils.scale_dim(dim, scale)
        self.assertEqual(str(result), str(expected))


class GeneratedDimTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("generated", ir.SymbolicDim("_d12"), True),
            ("user_named", ir.SymbolicDim("batch"), False),
            ("generated_expression", ir.SymbolicDim("2*_d0"), False),
            ("concrete", 3, False),
        ]
    )
    def test_is_generated_dim(self, _name, dim, expected):
        self.assertEqual(_utils.is_generated_dim(dim), expected)


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
