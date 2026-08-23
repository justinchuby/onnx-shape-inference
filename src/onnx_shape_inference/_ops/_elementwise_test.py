# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for binary element-wise shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

import onnx_shape_inference
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
    def test_max_min_distinct_symbolic_values_yield_unknown(self, op):
        output, symbolic_value = _infer_symbolic_values(
            op,
            [ir.SymbolicDim("a")],
            [ir.SymbolicDim("b")],
        )

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        assert symbolic_value is not None
        self.assertEqual(len(symbolic_value), 1)
        self.assertIsInstance(symbolic_value[0], ir.SymbolicDim)
        self.assertNotIn(symbolic_value[0], (ir.SymbolicDim("a"), ir.SymbolicDim("b")))

    @parameterized.parameterized.expand(
        [
            ("Max_int_symbolic", "Max", [1], [ir.SymbolicDim("a")]),
            ("Max_symbolic_int", "Max", [ir.SymbolicDim("a")], [1]),
            ("Min_int_symbolic", "Min", [1], [ir.SymbolicDim("a")]),
            ("Min_symbolic_int", "Min", [ir.SymbolicDim("a")], [1]),
        ]
    )
    def test_max_min_mixed_operands_yield_unknown_element(self, _name, op, values_a, values_b):
        output, symbolic_value = _infer_symbolic_values(op, values_a, values_b)

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        assert symbolic_value is not None
        self.assertIsInstance(symbolic_value[0], ir.SymbolicDim)
        # max(a, 1) is not a: a symbolic dim may legitimately be 0.
        self.assertNotEqual(symbolic_value[0], ir.SymbolicDim("a"))

    @parameterized.parameterized.expand([("Max",), ("Min",)])
    def test_max_min_same_symbolic_dim_is_preserved(self, op):
        output, symbolic_value = _infer_symbolic_values(
            op,
            [ir.SymbolicDim("a")],
            [ir.SymbolicDim("a")],
        )

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [1]))
        self.assertEqual(symbolic_value, [ir.SymbolicDim("a")])

    @parameterized.parameterized.expand(
        [
            ("Max", "Max", 8),
            ("Min", "Min", 4),
        ]
    )
    def test_max_min_keeps_concrete_lanes_alongside_symbolic_ones(self, _name, op, expected):
        # One symbolic element must not discard the whole list: the concrete
        # lane still folds exactly, only the symbolic lane degrades to unknown.
        output, symbolic_value = _infer_symbolic_values(
            op,
            [ir.SymbolicDim("a"), 4],
            [2, 8],
        )

        self.assertEqual(ir.TypeAndShape(output.type, output.shape), ts(INT64, [2]))
        assert symbolic_value is not None
        self.assertIsInstance(symbolic_value[0], ir.SymbolicDim)
        self.assertEqual(symbolic_value[1], expected)

    def test_sum_with_symbolic_operand_folds_arithmetically(self):
        _, symbolic_value = _infer_symbolic_values("Sum", [ir.SymbolicDim("a")], [1])

        self.assertEqual(symbolic_value, [ir.SymbolicDim("a") + 1])

    def test_mean_does_not_propagate_symbolic_values(self):
        _, symbolic_value = _infer_symbolic_values("Mean", [4], [2])

        self.assertIsNone(symbolic_value)


class VariadicSymbolicRetentionTest(unittest.TestCase):
    """End-to-end evidence that partially symbolic folds retain information."""

    @staticmethod
    def _build_reshape_model(op_type: str) -> ir.Model:
        """Build ``Reshape(data, Max/Min(Shape(x), [2, 8]))``.

        ``Shape(x)`` is ``[dyn, 4]``, so the first lane of the variadic op is
        symbolic and the second is concrete.  A downstream ``Reshape`` can only
        recover the concrete lane if the variadic op keeps propagating it.
        """
        x = ir.Value(
            name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([ir.SymbolicDim("dyn"), 4])
        )
        data = ir.Value(
            name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([ir.SymbolicDim("n")])
        )
        shape_node = ir.Node("", "Shape", [x], num_outputs=1)
        const = ir.Node(
            "",
            "Constant",
            [],
            num_outputs=1,
            attributes=[ir.AttrTensor("value", ir.tensor([2, 8], dtype=INT64))],
        )
        variadic = ir.Node(
            "", op_type, [shape_node.outputs[0], const.outputs[0]], num_outputs=1
        )
        reshape = ir.Node("", "Reshape", [data, variadic.outputs[0]], num_outputs=1)
        graph = ir.Graph(
            [x, data],
            [reshape.outputs[0]],
            nodes=[shape_node, const, variadic, reshape],
            name="retain",
            opset_imports={"": 24},
        )
        return ir.Model(graph, ir_version=11)

    @parameterized.parameterized.expand([("Max", "Max", 8), ("Min", "Min", 4)])
    def test_downstream_reshape_recovers_the_concrete_dim(self, _name, op, expected):
        model = self._build_reshape_model(op)

        onnx_shape_inference.infer_symbolic_shapes(model, policy="refine")

        output = model.graph.outputs[0]
        assert output.shape is not None
        self.assertEqual(len(output.shape), 2)
        # Before element-wise folding, one symbolic lane discarded the entire
        # symbolic value and this dim was unknown.
        self.assertEqual(output.shape[1], expected)

    @parameterized.parameterized.expand([("Max",), ("Min",)])
    def test_partially_symbolic_value_is_still_propagated(self, op):
        model = self._build_reshape_model(op)

        onnx_shape_inference.infer_symbolic_shapes(model, policy="refine")

        variadic_output = model.graph.node(2).outputs[0]
        self.assertIn(_context.SYM_DATA_KEY, variadic_output.metadata_props)


if __name__ == "__main__":
    unittest.main()
