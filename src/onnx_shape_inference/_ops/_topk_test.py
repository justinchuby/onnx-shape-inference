# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TopK shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class TopKTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("concrete", [3, 4, 5], [3, 4, "_d0"]),
            ("symbolic", ["N", "M", 5], ["N", "M", "_d0"]),
        ]
    )
    def test_basic(self, _name, input_shape, expected_shape):
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT, input_shape), ts(INT64, [1])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, -1)},
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, expected_shape))
        self.assertEqual(actual[1], ts(INT64, expected_shape))

    def test_const_k(self):
        """When K is a constant, the axis dim should be the concrete K value."""
        x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4, 5]))
        k = const_value([3])
        actual = run_shape_inference_with_values(
            "", "TopK", [x, k], opset_version=21, num_outputs=2
        )
        self.assertEqual(actual[0], ts(FLOAT, [3, 4, 3]))
        self.assertEqual(actual[1], ts(INT64, [3, 4, 3]))

    def test_symbolic_k_value(self):
        x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape(["N", "D"]))
        k = ir.Value(name="k", type=ir.TensorType(INT64), shape=ir.Shape([1]))
        actual = run_shape_inference_with_values(
            "",
            "TopK",
            [x, k],
            opset_version=21,
            num_outputs=2,
            symbolic_values={1: [ir.SymbolicDim("TopK_k")]},
        )
        self.assertEqual(
            actual,
            [ts(FLOAT, ["N", "TopK_k"]), ts(INT64, ["N", "TopK_k"])],
        )

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT), ts(INT64, [1])],
            opset_version=21,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT))
        self.assertEqual(actual[1], ts(INT64))

    def test_axis_out_of_range_records_error(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "TopK",
                [ts(FLOAT, [3, 4]), ts(INT64, [1])],
                {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
                opset_version=21,
                num_outputs=2,
            )

    def test_axis_out_of_range_sets_output_dtypes(self):
        actual = run_shape_inference(
            "",
            "TopK",
            [ts(FLOAT, [3, 4]), ts(INT64, [1])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
            opset_version=21,
            num_outputs=2,
            policy="skip",
        )
        self.assertEqual(actual, [ts(FLOAT), ts(INT64)])

    def test_none_input_raises(self):
        v = ir.Value(name="k", type=ir.TensorType(INT64), shape=ir.Shape([1]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "TopK", [None, v], opset_version=21, num_outputs=2
            )

    def test_opset_10_input_k(self):
        x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        k = const_value([2])
        actual = run_shape_inference_with_values(
            "",
            "TopK",
            [x, k],
            opset_version=10,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 2]), ts(INT64, [3, 2])])

    def test_attribute_k_schema_is_not_registered(self):
        with self.assertRaises(ValueError):
            run_shape_inference(
                "",
                "TopK",
                [ts(FLOAT, [3, 4])],
                {"k": ir.Attr("k", ir.AttributeType.INT, 2)},
                opset_version=9,
                num_outputs=2,
            )


if __name__ == "__main__":
    unittest.main()
