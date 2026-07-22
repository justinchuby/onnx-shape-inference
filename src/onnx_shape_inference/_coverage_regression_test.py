# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression harness for symbolic-shape coverage benchmark gaps."""

from __future__ import annotations

import unittest
from collections.abc import Callable, Sequence

import numpy as np
import onnx_ir as ir
import parameterized
import pytest

from onnx_shape_inference import infer_symbolic_shapes
from onnx_shape_inference._ops._testing import const_value, run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64

PatternBuilder = Callable[[], tuple[list[ir.Value], list[ir.Value]]]


def _value(name: str, dtype: ir.DataType, shape: Sequence[int | str] | None) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape) if shape is not None else None,
    )


def _float_const(data: Sequence[float], name: str) -> ir.Value:
    tensor = ir.Tensor(np.array(data, dtype=np.float32), name=name)
    return ir.Value(
        name=name,
        const_value=tensor,
        type=ir.TensorType(FLOAT),
        shape=ir.Shape([len(data)]),
    )


def _infer(
    inputs: list[ir.Value],
    nodes: list[ir.Node],
    outputs: list[ir.Value],
    *,
    opset_version: int = 21,
) -> None:
    model = ir.Model(
        ir.Graph(inputs, outputs, nodes=nodes, opset_imports={"": opset_version}),
        ir_version=10,
    )
    infer_symbolic_shapes(model)


def _types_and_shapes(values: list[ir.Value]) -> list[ir.TypeAndShape]:
    return [ir.TypeAndShape(value.type, value.shape) for value in values]


def _qwen_symbol_flow() -> tuple[list[ir.Value], list[ir.Value]]:
    hidden = _value("hidden", FLOAT, ["batch_size", "sequence_length", 16])
    shape = const_value([0, 0, 4, 4], "qkv_shape")
    reshaped = ir.Value(name="reshaped")
    transpose = ir.Node(
        "",
        "Transpose",
        [reshaped],
        outputs=[ir.Value(name="transposed")],
        attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 1, 3])},
    )
    reshape = ir.Node("", "Reshape", [hidden, shape], outputs=[reshaped])
    weight = _value("weight", FLOAT, [4, 4, 8])
    projected = ir.Node("", "MatMul", [transpose.outputs[0], weight], num_outputs=1)
    axes = const_value([2], "axes")
    unsqueeze = ir.Node("", "Unsqueeze", [projected.outputs[0], axes], num_outputs=1)
    _infer(
        [hidden, weight],
        [reshape, transpose, projected, unsqueeze],
        list(unsqueeze.outputs),
    )
    return [reshaped, transpose.outputs[0], projected.outputs[0], unsqueeze.outputs[0]], [
        unsqueeze.outputs[0]
    ]


def _reshape_simplification() -> tuple[list[ir.Value], list[ir.Value]]:
    x = _value("X", FLOAT, ["a", "b", "c"])
    first = ir.Node("", "Reshape", [x, const_value([0, 0, 2, -1], "shape1")], num_outputs=1)
    second = ir.Node(
        "", "Reshape", [first.outputs[0], const_value([0, 0, -1], "shape2")], num_outputs=1
    )
    output = ir.Node("", "Abs", list(second.outputs), num_outputs=1)
    _infer([x], [first, second, output], list(output.outputs))
    return [first.outputs[0], second.outputs[0], output.outputs[0]], list(output.outputs)


def _slice_symbolic_end() -> tuple[list[ir.Value], list[ir.Value]]:
    x = _value("X", FLOAT, ["a", "b", "c"])
    shape = ir.Node("", "Shape", [x], num_outputs=1)
    end = ir.Node("", "Gather", [shape.outputs[0], const_value([2], "index")], num_outputs=1)
    adjusted_end = ir.Node("", "Sub", [end.outputs[0], const_value([1], "one")], num_outputs=1)
    slice_node = ir.Node(
        "",
        "Slice",
        [
            x,
            const_value([0], "starts"),
            adjusted_end.outputs[0],
            const_value([2], "axes"),
        ],
        num_outputs=1,
    )
    output = ir.Node("", "Abs", list(slice_node.outputs), num_outputs=1)
    _infer([x], [shape, end, adjusted_end, slice_node, output], list(output.outputs))
    return [slice_node.outputs[0], output.outputs[0]], list(output.outputs)


def _resize_tile_symbolic() -> tuple[list[ir.Value], list[ir.Value]]:
    x = _value("X", FLOAT, ["H", "2*h"])
    resized = ir.Node(
        "",
        "Resize",
        [x, None, _float_const([0.5, 0.5], "scales")],
        num_outputs=1,
    )
    tiled = ir.Node(
        "", "Tile", [resized.outputs[0], const_value([2, 2], "repeats")], num_outputs=1
    )
    output = ir.Node("", "Max", [tiled.outputs[0], _value("zero", FLOAT, [])], num_outputs=1)
    _infer([x], [resized, tiled, output], list(output.outputs))
    return [resized.outputs[0], tiled.outputs[0], output.outputs[0]], list(output.outputs)


def _topk_anchor() -> tuple[list[ir.Value], list[ir.Value]]:
    x = _value("X", FLOAT, ["N", "N"])
    k = _value("K", INT64, [1])
    topk = ir.Node("", "TopK", [x, k], num_outputs=2)
    # The graph-output annotation is the benchmark's authoritative TopK-k name.
    anchored = _value("Y", FLOAT, ["N", "TopK_k"])
    output = ir.Node("", "Identity", [topk.outputs[0]], outputs=[anchored])
    _infer([x, k], [topk, output], [anchored])
    return [topk.outputs[0], topk.outputs[1], anchored], [anchored]


def _nonzero_anchor() -> tuple[list[ir.Value], list[ir.Value]]:
    x = _value("X", FLOAT, ["batch", "seq"])
    nonzero = ir.Node("", "NonZero", [x], num_outputs=1)
    flattened = ir.Node(
        "", "Reshape", [nonzero.outputs[0], const_value([-1], "flatten")], num_outputs=1
    )
    anchored = _value("Y", INT64, ["2*dnz"])
    output = ir.Node("", "Abs", [flattened.outputs[0]], outputs=[anchored])
    _infer([x], [nonzero, flattened, output], [anchored])
    return [nonzero.outputs[0], flattened.outputs[0], anchored], [anchored]


def _if_anchor() -> tuple[list[ir.Value], list[ir.Value]]:
    cond = _value("cond", ir.DataType.BOOL, [])
    then_a = _value("a_then", FLOAT, [3, 4])
    else_a = _value("a_else", FLOAT, [5, 4])
    else_condition = _value("c_else", ir.DataType.BOOL, [5])
    then_b = _value("b_then", INT64, [3])
    else_b = _value("b_else", INT64, [5])

    then_a_node = ir.Node("", "Identity", [then_a], num_outputs=1)
    then_b_node = ir.Node("", "Identity", [then_b], num_outputs=1)
    then_graph = ir.Graph(
        [],
        [then_a_node.outputs[0], then_b_node.outputs[0]],
        nodes=[then_a_node, then_b_node],
        opset_imports={"": 21},
        name="then",
    )
    else_a_node = ir.Node(
        "",
        "Compress",
        [else_a, else_condition],
        num_outputs=1,
        attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
    )
    else_b_node = ir.Node("", "Identity", [else_b], num_outputs=1)
    else_graph = ir.Graph(
        [],
        [else_a_node.outputs[0], else_b_node.outputs[0]],
        nodes=[else_a_node, else_b_node],
        opset_imports={"": 21},
        name="else",
    )
    if_node = ir.Node(
        "",
        "If",
        [cond],
        num_outputs=2,
        attributes={
            "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
            "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
        },
    )
    y1 = _value("Y1", FLOAT, ["B1", 4])
    y2 = _value("Y2", INT64, ["B2"])
    output_a = ir.Node("", "Abs", [if_node.outputs[0]], outputs=[y1])
    output_b = ir.Node("", "Neg", [if_node.outputs[1]], outputs=[y2])
    _infer(
        [cond, then_a, else_a, else_condition, then_b, else_b],
        [if_node, output_a, output_b],
        [y1, y2],
    )
    return [if_node.outputs[0], if_node.outputs[1], y1, y2], [y1, y2]


def _concat_anchor() -> tuple[list[ir.Value], list[ir.Value]]:
    past = _value("past_key", FLOAT, ["batch", 4, "past_seq", 4])
    current = _value("current_key", FLOAT, ["batch", 4, "seq", 4])
    concat = ir.Node(
        "",
        "Concat",
        [past, current],
        num_outputs=1,
        attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 2)},
    )
    anchored = _value("present_key", FLOAT, ["batch", 4, "total_seq", 4])
    output = ir.Node("", "Identity", list(concat.outputs), outputs=[anchored])
    _infer([past, current], [concat, output], [anchored])
    return [concat.outputs[0], anchored], [anchored]


_BUILDERS: dict[str, PatternBuilder] = {
    "qwen": _qwen_symbol_flow,
    "reshape": _reshape_simplification,
    "slice": _slice_symbolic_end,
    "resize-tile": _resize_tile_symbolic,
    "topk": _topk_anchor,
    "nonzero": _nonzero_anchor,
    "if": _if_anchor,
    "tiny-llm": _concat_anchor,
}


class CoverageRegressionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "reshape_double_floor_division",
                "reshape",
                [
                    ts(FLOAT, ["a", "b", 2, "floor(c/2)"]),
                    ts(FLOAT, ["a", "b", "c"]),
                    ts(FLOAT, ["a", "b", "c"]),
                ],
            ),
            (
                "slice_symbolic_end",
                "slice",
                [ts(FLOAT, ["a", "b", "c - 1"]), ts(FLOAT, ["a", "b", "c - 1"])],
            ),
            (
                "resize_tile_symbolic",
                "resize-tile",
                [
                    ts(FLOAT, ["floor(H/2)", "h"]),
                    ts(FLOAT, ["2*floor(H/2)", "2*h"]),
                    ts(FLOAT, ["2*floor(H/2)", "2*h"]),
                ],
            ),
            (
                "topk_declared_k",
                "topk",
                [
                    ts(FLOAT, ["N", "TopK_k"]),
                    ts(INT64, ["N", "TopK_k"]),
                    ts(FLOAT, ["N", "TopK_k"]),
                ],
            ),
            (
                "nonzero_declared_expression",
                "nonzero",
                [ts(INT64, [2, "dnz"]), ts(INT64, ["2*dnz"]), ts(INT64, ["2*dnz"])],
            ),
            (
                "if_declared_branch_dims",
                "if",
                [
                    ts(FLOAT, ["B1", 4]),
                    ts(INT64, ["B2"]),
                    ts(FLOAT, ["B1", 4]),
                    ts(INT64, ["B2"]),
                ],
            ),
            (
                "tiny_llm_declared_total_seq",
                "tiny-llm",
                [
                    ts(FLOAT, ["batch", 4, "total_seq", 4]),
                    ts(FLOAT, ["batch", 4, "total_seq", 4]),
                ],
            ),
        ]
    )
    @pytest.mark.xfail(
        reason="Pending symbolic simplification, derived dimensions, and anchor propagation",
        strict=False,
    )
    def test_coverage_gaps(
        self, _name: str, pattern: str, expected: list[ir.TypeAndShape]
    ) -> None:
        actual, _ = _BUILDERS[pattern]()
        self.assertEqual(_types_and_shapes(actual), expected)

    def test_qwen_named_dim_forward_propagation(self) -> None:
        actual, _ = _qwen_symbol_flow()
        self.assertEqual(
            _types_and_shapes(actual),
            [
                ts(FLOAT, ["batch_size", "sequence_length", 4, 4]),
                ts(FLOAT, ["batch_size", 4, "sequence_length", 4]),
                ts(FLOAT, ["batch_size", 4, "sequence_length", 8]),
                ts(FLOAT, ["batch_size", 4, 1, "sequence_length", 8]),
            ],
        )

    def test_tiny_llm_concat_expression_before_anchor_adoption(self) -> None:
        actual = run_shape_inference(
            "",
            "Concat",
            [
                ts(FLOAT, ["batch", 4, "past_seq", 4]),
                ts(FLOAT, ["batch", 4, "seq", 4]),
            ],
            {"axis": ir.Attr("axis", ir.AttributeType.INT, 2)},
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 4, "past_seq + seq", 4])])

    @parameterized.parameterized.expand(
        [
            ("topk_constant_k", [3, 4, 5], [3], [3, 4, 3]),
            ("topk_unknown_input_shape", None, [3], None),
            ("resize_unknown_input_shape", None, None, None),
        ]
    )
    def test_dtype_only_and_constant_boundaries(
        self,
        _name: str,
        input_shape: list[int] | None,
        k: list[int] | None,
        expected_shape: list[int] | None,
    ) -> None:
        if k is not None:
            actual = run_shape_inference(
                "",
                "TopK",
                [ts(FLOAT, input_shape), ts(INT64, [1])],
                opset_version=21,
                num_outputs=2,
            )
            if expected_shape is None:
                self.assertEqual(actual, [ts(FLOAT), ts(INT64)])
            else:
                x = _value("x", FLOAT, input_shape)
                topk = ir.Node("", "TopK", [x, const_value(k, "k")], num_outputs=2)
                _infer([x], [topk], list(topk.outputs))
                self.assertEqual(
                    _types_and_shapes(list(topk.outputs)),
                    [ts(FLOAT, expected_shape), ts(INT64, expected_shape)],
                )
        else:
            actual = run_shape_inference("", "Resize", [ts(FLOAT, input_shape)], opset_version=19)
            self.assertEqual(actual, [ts(FLOAT)])


if __name__ == "__main__":
    unittest.main()
