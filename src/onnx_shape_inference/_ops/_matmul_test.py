# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MatMul shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, _context, _registry
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class MatMulTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("2d", [3, 4], [4, 5], [3, 5]),
            ("batch", [2, 3, 4], [2, 4, 5], [2, 3, 5]),
            ("broadcast_batch", [1, 3, 4], [2, 4, 5], [2, 3, 5]),
            ("1d_1d", [4], [4], []),
            ("2d_1d", [3, 4], [4], [3]),
            ("1d_2d", [4], [4, 5], [5]),
            ("symbolic", ["batch", "M", 64], ["batch", 64, "N"], ["batch", "M", "N"]),
            ("high_rank", [2, 3, 4, 5], [5, 6], [2, 3, 4, 6]),
        ]
    )
    def test_matmul(self, _name, shape_a, shape_b, expected_shape):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, shape_a), ts(FLOAT, shape_b)],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT), ts(FLOAT, [4, 5])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_unproven_named_batch_dim_is_not_substituted(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [
                ts(FLOAT, [None, 16, "sequence_length", "total_sequence_length"]),
                ts(FLOAT, ["batch_size", 16, "total_sequence_length", 128]),
            ],
            opset_version=17,
        )
        self.assertEqual(
            actual,
            [ts(FLOAT, ["_d1", 16, "sequence_length", 128])],
        )

    def test_generated_second_batch_dim_is_conservative(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [
                ts(FLOAT, ["batch_size", 2, 3]),
                ts(FLOAT, [None, 3, 4]),
            ],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d1", 2, 4])])

    def test_generated_first_batch_dim_is_conservative(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [
                ts(FLOAT, [None, 2, 3]),
                ts(FLOAT, ["batch_size", 3, 4]),
            ],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d1", 2, 4])])

    def test_differing_named_batch_dims_produce_fresh_dim(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [
                ts(FLOAT, ["batch_a", 2, 3]),
                ts(FLOAT, ["batch_b", 3, 4]),
            ],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d0", 2, 4])])

    def test_author_named_generated_pattern_is_not_treated_as_generated(self):
        a = ir.Value(
            name="a",
            type=ir.TensorType(FLOAT),
            shape=ir.Shape(["_d0", 2, 3]),
        )
        b = ir.Value(
            name="b",
            type=ir.TensorType(FLOAT),
            shape=ir.Shape(["batch_size", 3, 4]),
        )
        output = ir.Value(name="output")
        node = ir.Node("", "MatMul", inputs=[a, b], outputs=[output], attributes={})
        ctx = _context.ShapeInferenceContext({"": 17})
        ctx.reserve_symbol_names(["_d0", "batch_size"])

        func = _registry.registry.get("", "MatMul", version=17)
        func(ctx, node)

        self.assertEqual(
            ir.TypeAndShape(output.type, output.shape),
            ts(FLOAT, ["_d1", 2, 4]),
        )

    def test_equal_named_batch_dim_is_substituted(self):
        a = ir.Value(
            name="a",
            type=ir.TensorType(FLOAT),
            shape=ir.Shape([None, 16, "sequence_length", "total_sequence_length"]),
        )
        b = ir.Value(
            name="b",
            type=ir.TensorType(FLOAT),
            shape=ir.Shape(["batch_size", 16, "total_sequence_length", 128]),
        )
        output = ir.Value(name="output")
        node = ir.Node("", "MatMul", inputs=[a, b], outputs=[output], attributes={})
        ctx = _context.ShapeInferenceContext({"": 17})
        ctx.name_anonymous_dims(a)
        ctx.add_symbolic_equality("_d0", "batch_size")

        func = _registry.registry.get("", "MatMul", version=17)
        func(ctx, node)

        self.assertEqual(
            ir.TypeAndShape(output.type, output.shape),
            ts(FLOAT, ["batch_size", 16, "sequence_length", 128]),
        )

    def test_matmul_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "MatMul", [], opset_version=17)

    def test_matmul_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3, 4]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "MatMul",
                [v, None],
                opset_version=17,
            )

    def test_matmul_1d_times_2d(self):
        """1D @ 2D: [K] @ [K, N] → [N]."""
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, [4]), ts(FLOAT, [4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [5])])

    def test_contraction_dimension_mismatch_records_error(self):
        with self.assertRaisesRegex(ShapeInferenceError, "contraction dimensions: 3 vs 5"):
            run_shape_inference(
                "",
                "MatMul",
                [ts(FLOAT, [2, 3]), ts(FLOAT, [5, 4])],
                opset_version=17,
            )

    def test_contraction_dimension_mismatch_preserves_best_effort_shape(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, [2, 3]), ts(FLOAT, [5, 4])],
            opset_version=17,
            policy="skip",
        )

        self.assertEqual(actual, [ts(FLOAT, [2, 4])])

    def test_symbolic_contraction_dimensions_are_not_rejected(self):
        actual = run_shape_inference(
            "",
            "MatMul",
            [ts(FLOAT, [2, "K1"]), ts(FLOAT, ["K2", 4])],
            opset_version=17,
        )

        self.assertEqual(actual, [ts(FLOAT, [2, 4])])

    def test_control_flow_output_contraction_dimension_is_conservative(self):
        input_a = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([2, 3]))
        ir.Node("", "If", inputs=[], outputs=[input_a], attributes={})
        input_b = ir.Value(name="b", type=ir.TensorType(FLOAT), shape=ir.Shape([5, 4]))

        actual = run_shape_inference_with_values(
            "",
            "MatMul",
            [input_a, input_b],
            opset_version=17,
        )

        self.assertEqual(actual, [ts(FLOAT, [2, 4])])

    def test_matmul_incompatible_batch_raises(self):
        """Incompatible batch dims [2, ...] vs [3, ...] should raise."""
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "MatMul",
                [ts(FLOAT, [2, 3, 4]), ts(FLOAT, [3, 4, 5])],
                opset_version=17,
            )


INT8 = ir.DataType.INT8
INT32 = ir.DataType.INT32
UINT8 = ir.DataType.UINT8


class MatMulIntegerTest(unittest.TestCase):
    def test_matmul_integer_basic(self):
        actual = run_shape_inference(
            "",
            "MatMulInteger",
            [ts(INT8, [3, 4]), ts(INT8, [4, 5])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [3, 5])])

    def test_matmul_integer_batch(self):
        actual = run_shape_inference(
            "",
            "MatMulInteger",
            [ts(INT8, [2, 3, 4]), ts(INT8, [2, 4, 5])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT32, [2, 3, 5])])


class QLinearMatMulTest(unittest.TestCase):
    def test_qlinear_matmul_basic(self):
        actual = run_shape_inference(
            "",
            "QLinearMatMul",
            [
                ts(UINT8, [3, 4]),  # a
                ts(FLOAT, []),  # a_scale
                ts(UINT8, []),  # a_zero_point
                ts(UINT8, [4, 5]),  # b
                ts(FLOAT, []),  # b_scale
                ts(UINT8, []),  # b_zero_point
                ts(FLOAT, []),  # y_scale
                ts(UINT8, []),  # y_zero_point
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(UINT8, [3, 5])])

    def test_qlinear_matmul_int8_output(self):
        actual = run_shape_inference(
            "",
            "QLinearMatMul",
            [
                ts(INT8, [3, 4]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [4, 5]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),  # y_zero_point dtype determines output
            ],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(INT8, [3, 5])])


if __name__ == "__main__":
    unittest.main()
