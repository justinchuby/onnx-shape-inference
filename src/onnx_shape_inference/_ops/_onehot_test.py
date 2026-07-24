# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneHot shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import OpUsageError, ShapeInferenceError
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class OneHotTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "OneHot",
            [ts(INT64, [3, 4]), ts(INT64, []), ts(FLOAT, [2])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, "_d0"])])

    def test_symbolic_batch(self):
        """OneHot with symbolic batch: ["N", 4] → rank 3, batch is SymbolicDim."""
        actual = run_shape_inference(
            "",
            "OneHot",
            [ts(INT64, ["N", 4]), ts(INT64, []), ts(FLOAT, [2])],
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", 4, "_d0"])])

    def test_valid_leading_axis_with_constant_depth(self):
        indices = ir.Value(name="indices", type=ir.TensorType(INT64), shape=ir.Shape([3, 4]))
        values = ir.Value(name="values", type=ir.TensorType(FLOAT), shape=ir.Shape([2]))
        actual = run_shape_inference_with_values(
            "",
            "OneHot",
            [indices, const_value(5, "depth"), values],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=21,
        )
        self.assertEqual(actual, [ts(FLOAT, [5, 3, 4])])

    def test_axis_out_of_range_records_error(self):
        for axis in (5, -5):
            with self.subTest(axis=axis), self.assertRaises(ShapeInferenceError):
                run_shape_inference(
                    "",
                    "OneHot",
                    [ts(INT64, [3, 4]), ts(INT64, []), ts(FLOAT, [2])],
                    {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)},
                    opset_version=21,
                )

    def test_axis_out_of_range_gracefully_degrades(self):
        actual = run_shape_inference(
            "",
            "OneHot",
            [ts(INT64, [3, 4]), ts(INT64, []), ts(FLOAT, [2])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
            opset_version=21,
            policy="skip",
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_opset_9(self):
        actual = run_shape_inference(
            "",
            "OneHot",
            [ts(INT64, [3, 4]), ts(INT64, []), ts(FLOAT, [2])],
            opset_version=9,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, "_d0"])])

    def test_none_input_raises(self):
        v_depth = ir.Value(name="depth", type=ir.TensorType(INT64), shape=ir.Shape([]))
        v_values = ir.Value(name="values", type=ir.TensorType(FLOAT), shape=ir.Shape([2]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "OneHot", [None, v_depth, v_values], opset_version=21
            )


if __name__ == "__main__":
    unittest.main()
