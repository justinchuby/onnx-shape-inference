# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for binary element-wise shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, _context, _registry
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
BOOL = ir.DataType.BOOL
INT64 = ir.DataType.INT64

# All arithmetic binary ops share the same broadcast logic
_ARITHMETIC_OPS = ["Add", "Sub", "Mul", "Div", "Pow"]
_COMPARISON_OPS = ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"]
_LOGICAL_OPS = ["And", "Or", "Xor"]


def _infer_symbolic_values(
    op_type: str,
    *symbolic_values: list[int | ir.SymbolicDim],
) -> tuple[ir.Value, list[int | ir.SymbolicDim] | None]:
    inputs = [
        ir.Value(
            name=f"input_{index}",
            type=ir.TensorType(INT64),
            shape=ir.Shape([len(values)]),
        )
        for index, values in enumerate(symbolic_values)
    ]
    output = ir.Value(name="output")
    node = ir.Node("", op_type, inputs=inputs, outputs=[output], attributes={})
    ctx = _context.ShapeInferenceContext({"": 17})
    for input_value, values in zip(inputs, symbolic_values):
        ctx.set_symbolic_value(input_value, values)
    func = _registry.registry.get("", op_type, version=17)
    func(ctx, node)
    return output, ctx.get_symbolic_value(output)


class BinaryElementwiseTest(unittest.TestCase):
    """Tests for binary element-wise shape inference."""

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_broadcast(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 1, 5]), ts(FLOAT, [1, 4, 5])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4, 5])])

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_symbolic(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 128]), ts(FLOAT, [1, 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _ARITHMETIC_OPS])
    def test_arithmetic_missing_shape(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT), ts(FLOAT, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    @parameterized.parameterized.expand([(op,) for op in _COMPARISON_OPS])
    def test_comparison_output_bool(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4]), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [3, 4])])

    @parameterized.parameterized.expand([(op,) for op in _COMPARISON_OPS])
    def test_comparison_broadcast(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, ["batch", 1]), ts(FLOAT, [1, 128])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, ["batch", 128])])

    @parameterized.parameterized.expand([(op,) for op in _LOGICAL_OPS])
    def test_logical_output_bool(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(BOOL, [2, 3]), ts(BOOL, [2, 3])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(BOOL, [2, 3])])

    def test_mod(self):
        actual = run_shape_inference(
            "",
            "Mod",
            [ts(INT64, [3, 4]), ts(INT64, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(INT64, [3, 4])])

    def test_symbolic_broadcast_dims(self):
        """Symbolic broadcast: ["N", 1] * [1, "M"] → ["N", "M"]."""
        actual = run_shape_inference(
            "",
            "Mul",
            [ts(FLOAT, ["N", 1]), ts(FLOAT, [1, "M"])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", "M"])])

    def test_add_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Add", [], opset_version=17)

    def test_add_none_input(self):
        v = ir.Value(name="a", type=ir.TensorType(FLOAT), shape=ir.Shape([3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Add",
                [v, None],
                opset_version=17,
            )

    def test_integer_division_truncates_toward_zero(self):
        output, symbolic_value = _infer_symbolic_values("Div", [-7], [2])

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        self.assertEqual(symbolic_value, [-3])

    def test_integer_division_with_symbolic_operand_skips_propagation(self):
        output, symbolic_value = _infer_symbolic_values(
            "Div", [ir.SymbolicDim("dividend")], [2]
        )

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        self.assertIsNone(symbolic_value)


class VariadicElementwiseTest(unittest.TestCase):
    """Tests for variadic element-wise ops (Max, Min, Mean, Sum)."""

    @parameterized.parameterized.expand([(op,) for op in ["Max", "Min", "Mean", "Sum"]])
    def test_variadic_three_inputs(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4]), ts(FLOAT, [3, 4]), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    @parameterized.parameterized.expand([(op,) for op in ["Max", "Min", "Mean", "Sum"]])
    def test_variadic_broadcast(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 1]), ts(FLOAT, [1, 4]), ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    @parameterized.parameterized.expand([(op,) for op in ["Max", "Min", "Mean", "Sum"]])
    def test_variadic_single_input(self, op):
        actual = run_shape_inference(
            "",
            op,
            [ts(FLOAT, [3, 4])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    @parameterized.parameterized.expand(
        [
            ("Max", [3, -2], [1, 4], [3, 4]),
            ("Min", [3, -2], [1, 4], [1, -2]),
        ]
    )
    def test_max_min_propagate_concrete_values(self, op, values_a, values_b, expected):
        output, symbolic_value = _infer_symbolic_values(op, values_a, values_b)

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [2]))
        self.assertEqual(symbolic_value, expected)

    @parameterized.parameterized.expand([("Max",), ("Min",)])
    def test_max_min_symbolic_values_skip_propagation(self, op):
        output, symbolic_value = _infer_symbolic_values(
            op,
            [ir.SymbolicDim("a")],
            [ir.SymbolicDim("b")],
        )

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        self.assertIsNone(symbolic_value)


if __name__ == "__main__":
    unittest.main()
