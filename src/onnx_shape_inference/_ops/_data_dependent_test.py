# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for data-dependent output shape operators (NonZero, Compress, Unique)."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import OpUsageError, ShapeInferenceError, infer_symbolic_shapes
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class NonZeroTest(unittest.TestCase):
    def test_opset_9(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, [3, 4])], opset_version=9)
        self.assertEqual(actual, [ts(INT64, [2, "_d0"])])

    def test_known_rank(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, [3, 4])], opset_version=17)
        self.assertEqual(actual, [ts(INT64, [2, "_d0"])])

    def test_constant_input(self):
        actual = run_shape_inference_with_values(
            "",
            "NonZero",
            [const_value([0, 1, 2])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [1, 2])])

    def test_symbolic_input(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT, ["N", 3])], opset_version=17)
        self.assertEqual(actual, [ts(INT64, [2, "_d0"])])

    def test_unknown_rank(self):
        actual = run_shape_inference("", "NonZero", [ts(FLOAT)], opset_version=17)
        self.assertEqual(actual, [ts(INT64, ["_d0", "_d1"])])

    def test_num_nonzero_symbol_flows_through_reshape(self):
        x = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape(["batch", "seq"]))
        nonzero = ir.Node("", "NonZero", [x], num_outputs=1)
        flattened = ir.Node(
            "",
            "Reshape",
            [nonzero.outputs[0], const_value([-1], "shape")],
            num_outputs=1,
        )
        model = ir.Model(
            ir.Graph(
                [x],
                list(flattened.outputs),
                nodes=[nonzero, flattened],
                opset_imports={"": 21},
            ),
            ir_version=10,
        )

        infer_symbolic_shapes(model)

        self.assertEqual(
            ir.TypeAndShape(nonzero.outputs[0].type, nonzero.outputs[0].shape),
            ts(INT64, [2, "_d0"]),
        )
        self.assertEqual(
            ir.TypeAndShape(flattened.outputs[0].type, flattened.outputs[0].shape),
            ts(INT64, ["2*_d0"]),
        )


class CompressTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "Compress", [ts(FLOAT, ["N", 3])], opset_version=17)
        self.assertEqual(actual, [ts(FLOAT, ["_d0"])])

    def test_axis_out_of_range_records_error(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "Compress",
                [ts(FLOAT, ["N", 3])],
                {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
                opset_version=17,
            )

    def test_axis_out_of_range_gracefully_degrades(self):
        actual = run_shape_inference(
            "",
            "Compress",
            [ts(FLOAT, ["N", 3])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 5)},
            opset_version=17,
            policy="skip",
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_constant_condition_with_valid_axis(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        actual = run_shape_inference_with_values(
            "",
            "Compress",
            [data, const_value([True, False, True])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 4])])

    def test_opset_9(self):
        actual = run_shape_inference("", "Compress", [ts(FLOAT, ["N", 3])], opset_version=9)
        self.assertEqual(actual, [ts(FLOAT, ["_d0"])])


class UniqueTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "", "Unique", [ts(FLOAT, [5])], opset_version=17, num_outputs=4
        )
        self.assertEqual(actual[0], ts(FLOAT, ["_d0"]))
        self.assertEqual(actual[1].type.dtype, INT64)
        self.assertEqual(actual[2].type.dtype, INT64)
        self.assertEqual(actual[3].type.dtype, INT64)

    def test_constant_input_without_axis(self):
        actual = run_shape_inference_with_values(
            "",
            "Unique",
            [const_value([1, 1, 2, 2])],
            opset_version=17,
            num_outputs=4,
        )
        self.assertEqual(
            actual,
            [
                ts(INT64, [2]),
                ts(INT64, [2]),
                ts(INT64, [4]),
                ts(INT64, [2]),
            ],
        )

    def test_constant_input_with_axis(self):
        actual = run_shape_inference_with_values(
            "",
            "Unique",
            [const_value([[1, 1], [1, 2]])],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=17,
            num_outputs=4,
        )
        self.assertEqual(
            actual,
            [
                ts(INT64, [2, 2]),
                ts(INT64, [2]),
                ts(INT64, [2]),
                ts(INT64, [2]),
            ],
        )

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "", "Unique", [None], opset_version=17, num_outputs=4
            )


if __name__ == "__main__":
    unittest.main()
