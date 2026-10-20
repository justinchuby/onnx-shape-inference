# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the recurrent and Engram schemas released in ORT 1.31."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import (
    OpUsageError,
    ShapeInferenceContext,
    ShapeInferenceError,
    registry,
)
from onnx_shape_inference._ops._testing import run_shape_inference, ts

FLOAT = ir.DataType.FLOAT
HALF = ir.DataType.FLOAT16
INT32 = ir.DataType.INT32
INT64 = ir.DataType.INT64
MS = "com.microsoft"


class HyperConnectionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("grouped", [2, 7, 4, 64], 0, [2, 7, 64]),
            ("grouped_symbolic", ["B", "S", 4, "H"], 0, ["B", "S", "H"]),
            ("grouped_rank2", [4, 64], 0, [64]),
            ("flattened", [2, 7, 256], 4, [2, 7, 64]),
            ("flattened_symbolic", ["B", "S", "W"], 4, ["B", "S", "floor(W/4)"]),
            ("flattened_vector", [256], 4, [64]),
            ("unknown", None, 4, None),
        ]
    )
    def test_pre_mix(self, _name, shape, branches, expected):
        actual = run_shape_inference(
            MS,
            "HyperConnectionPreMix",
            [ts(HALF, shape), ts(FLOAT)],
            {"num_branches": ir.AttrInt64("num_branches", branches)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(HALF, expected)])

    @parameterized.parameterized.expand(
        [("negative", [256], -1), ("rank", [64], 0), ("scalar", [], 4), ("width", [255], 4)]
    )
    def test_invalid_pre_mix(self, _name, shape, branches):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "HyperConnectionPreMix",
                [ts(FLOAT, shape), ts(FLOAT)],
                {"num_branches": ir.AttrInt64("num_branches", branches)},
                opset_version=1,
            )

    @parameterized.parameterized.expand(
        [
            (op, shape)
            for op in ("BranchwiseRMSNorm", "ScaledSiLU", "HyperConnectionPostMix")
            for shape in ([2, 3, 4, 64], ["B", "S", "W"], None)
        ]
    )
    def test_shape_preserving(self, op, shape):
        inputs = [ts(HALF, shape)]
        if op == "HyperConnectionPostMix":
            inputs += [ts(HALF), ts(FLOAT)]
        actual = run_shape_inference(MS, op, inputs, opset_version=1)
        self.assertEqual(actual, [ts(HALF, shape)])

    @parameterized.parameterized.expand([(1,), (2,)])
    def test_engram_gate(self, outputs):
        inputs = [ts(HALF, ["B", "S", 4, 64])] * 2 + [
            ts(HALF, ["B", "S", 64]),
            ts(HALF, [4, 64]),
            ts(HALF, [4, 64]),
        ]
        if outputs == 2:
            inputs.append(ts(HALF, [4, 64]))
        actual = run_shape_inference(
            MS, "EngramGate", inputs, opset_version=1, num_outputs=outputs
        )
        self.assertEqual(actual, [ts(HALF, ["B", "S", 4, 64])] * outputs)

    def test_engram_gate_missing_optional_scale(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS, "EngramGate", [ts(HALF)] * 5, opset_version=1, num_outputs=2
            )


class NGramHashMappingTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (packed, dtype, symbolic)
            for packed in (False, True)
            for dtype in (INT32, INT64)
            for symbolic in (False, True)
        ]
    )
    def test_shapes_and_state(self, packed, dtype, symbolic):
        batch, seq = ("B", "S") if symbolic else (2, 7)
        inputs = [ts(dtype, [seq] if packed else [batch, seq]), ts(dtype, [4]), ts(dtype, [6])]
        if packed:
            inputs.append(ts(INT32, ["Bp1"] if symbolic else [3]))
        actual = run_shape_inference(
            MS,
            "VarlenNGramHashMapping" if packed else "NGramHashMapping",
            inputs,
            {
                "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
                "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
            },
            opset_version=1,
            num_outputs=3 if packed else 2,
        )
        state_batch = "Bp1 - 1" if packed and symbolic else batch
        expected = [
            ts(dtype, [seq, 6] if packed else [batch, seq, 6]),
            ts(dtype, [state_batch, 3]),
        ]
        if packed:
            expected.append(ts(INT32, [state_batch, 3]))
        self.assertEqual(actual, expected)

    def test_packed_state_update(self):
        actual = run_shape_inference(
            MS,
            "VarlenNGramHashMapping",
            [
                ts(INT64, [13]),
                ts(INT64, [4]),
                ts(INT64, [6]),
                ts(INT32, [3]),
                None,
                None,
                None,
                None,
                None,
                ts(INT32, [2]),
            ],
            {
                "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
                "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
                "state_update_capacity": ir.AttrInt64("state_update_capacity", 3),
            },
            opset_version=1,
            num_outputs=4,
        )
        self.assertEqual(
            actual,
            [
                ts(INT64, [13, 6]),
                ts(INT64, [2, 3]),
                ts(INT32, [2, 3]),
                ts(INT64, [2, 3, 3]),
            ],
        )

    @parameterized.parameterized.expand(
        [("order", 1, 2), ("heads", 4, 0), ("overflow", 2**62, 4)]
    )
    def test_invalid_attributes(self, _name, order, heads):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "NGramHashMapping",
                [ts(INT64, [2, 7]), ts(INT64), ts(INT64)],
                {
                    "max_ngram_size": ir.AttrInt64("max_ngram_size", order),
                    "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", heads),
                },
                opset_version=1,
            )

    def test_unknown_input_shape_keeps_state_width(self):
        actual = run_shape_inference(
            MS,
            "NGramHashMapping",
            [ts(INT64), ts(INT64), ts(INT64)],
            {
                "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
                "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
            },
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(INT64), ts(INT64, ["_d0", 3])])


class VarlenCausalConvTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("concrete", [13, 64], [64, 1, 4], [3], 1, [2, 64, 3]),
            ("dilated", [13, 64], [64, 1, 4], [3], 3, [2, 64, 9]),
            ("symbolic", ["T", "C"], ["C", 1, "K"], ["Bp1"], 2, ["Bp1 - 1", "C", "2*K - 2"]),
        ]
    )
    def test_state(self, _name, shape, weight, offsets, dilation, state):
        actual = run_shape_inference(
            MS,
            "VarlenCausalConvWithState",
            [ts(HALF, shape), ts(HALF, weight), ts(INT32, offsets), None, ts(HALF)],
            {"dilation": ir.AttrInt64("dilation", dilation)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(HALF, shape), ts(HALF, state)])

    def test_compact_update(self):
        actual = run_shape_inference(
            MS,
            "VarlenCausalConvWithState",
            [
                ts(HALF, [13, 64]),
                ts(HALF, [64, 1, 4]),
                ts(INT32, [3]),
                None,
                ts(HALF, [2, 64, 3]),
                ts(INT32, [2]),
            ],
            {"state_update_capacity": ir.AttrInt64("state_update_capacity", 4)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(HALF, [13, 64]), ts(HALF, [2, 64, 3]), ts(HALF, [2, 4, 64])]
        )

    def test_unknown_input_shape(self):
        actual = run_shape_inference(
            MS,
            "VarlenCausalConvWithState",
            [ts(HALF), ts(HALF), ts(INT32, [3]), None, ts(HALF, [2, 64, 3])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(HALF), ts(HALF, [2, 64, 3])])

    @parameterized.parameterized.expand([(0,), (-1,)])
    def test_invalid_dilation(self, dilation):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "VarlenCausalConvWithState",
                [ts(HALF), ts(HALF), ts(INT32), None, ts(HALF)],
                {"dilation": ir.AttrInt64("dilation", dilation)},
                opset_version=1,
                num_outputs=2,
            )

    def test_missing_initial_state(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS,
                "VarlenCausalConvWithState",
                [ts(HALF), ts(HALF), ts(INT32)],
                opset_version=1,
                num_outputs=2,
            )


class GatedDeltaNetTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("packed_tokens", [13, 4, 16], [13, 8, 32], [3], [2, 8, 32, 16], [13, 8, 32]),
            ("padded", [2, 7, 4, 16], [2, 7, 8, 32], None, [2, 8, 32, 16], [2, 7, 8, 32]),
            ("inverse_heads", [13, 8, 16], [13, 4, 32], [3], [2, 4, 32, 16], [13, 8, 32]),
            (
                "symbolic",
                ["T", 4, "K"],
                ["T", 8, "V"],
                ["Bp1"],
                ["Bp1 - 1", 8, "V", "K"],
                ["T", 8, "V"],
            ),
        ]
    )
    def test_separate_inputs(self, _name, query, value, offsets, state, output):
        actual = run_shape_inference(
            MS,
            "GatedDeltaNet",
            [
                ts(HALF, query),
                ts(HALF, query),
                ts(HALF, value),
                ts(INT32, offsets) if offsets is not None else None,
            ],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(HALF, output), ts(FLOAT, state)])

    @parameterized.parameterized.expand(
        [("tokens", [14, 384], [14, 8, 32]), ("padded", [2, 7, 384], [2, 7, 8, 32])]
    )
    def test_packed_qkv(self, _name, query, output):
        actual = run_shape_inference(
            MS,
            "GatedDeltaNet",
            [ts(HALF, query), None, None, None, None, None, ts(FLOAT, [2, 8, 32, 16])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(HALF, output), ts(FLOAT, [2, 8, 32, 16])])

    def test_compact_update(self):
        actual = run_shape_inference(
            MS,
            "GatedDeltaNet",
            [
                ts(HALF, [13, 4, 16]),
                ts(HALF, [13, 4, 16]),
                ts(HALF, [13, 8, 32]),
                ts(INT32, [3]),
                None,
                None,
                None,
                None,
                None,
                ts(INT32, [2]),
            ],
            {"state_update_capacity": ir.AttrInt64("state_update_capacity", 3)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(HALF, [13, 8, 32]), ts(FLOAT, [2, 8, 32, 16]), ts(FLOAT, [2, 984])]
        )

    def test_unknown_shapes(self):
        self.assertEqual(
            run_shape_inference(
                MS, "GatedDeltaNet", [ts(HALF)] * 3, opset_version=1, num_outputs=3
            ),
            [ts(HALF), ts(FLOAT), ts(FLOAT)],
        )

    @parameterized.parameterized.expand(
        [
            ("zero_key_size", [2, 8, 32, 0], 384),
            ("fractional", [2, 8, 32, 16], 385),
            ("no_query", [2, 8, 32, 16], 256),
        ]
    )
    def test_invalid_packed_geometry(self, _name, state, width):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "GatedDeltaNet",
                [ts(HALF, [14, width]), None, None, None, None, None, ts(FLOAT, state)],
                opset_version=1,
                num_outputs=2,
            )

    def test_missing_packed_state(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MS, "GatedDeltaNet", [ts(HALF, [14, 384])], opset_version=1)


class StatefulBoundaryTest(unittest.TestCase):
    def _invalid(self, op, inputs, attrs=None, outputs=2):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(MS, op, inputs, attrs, opset_version=1, num_outputs=outputs)
        with self.assertLogs("onnx_shape_inference._context", level="WARNING"):
            run_shape_inference(
                MS, op, inputs, attrs, opset_version=1, num_outputs=outputs, policy="skip"
            )

    @parameterized.parameterized.expand(
        [
            ("negative", [256], -1),
            ("grouped_rank", [64], 0),
            ("flattened_rank", [], 4),
            ("flattened_width", [255], 4),
        ]
    )
    def test_pre_mix_logged_errors(self, _name, shape, branches):
        self._invalid(
            "HyperConnectionPreMix",
            [ts(HALF, shape), ts(FLOAT)],
            {"num_branches": ir.AttrInt64("num_branches", branches)},
            1,
        )

    @parameterized.parameterized.expand(
        [
            ("ids_rank", 0, [2, 7], 0),
            ("offset_rank", 3, [], 0),
            ("offset_size", 3, [1], 0),
            ("count_rank", 9, [], 1),
            ("count_batch", 9, [3], 1),
            ("capacity", 0, [13], 9),
        ]
    )
    def test_ngram_invalid(self, _name, index, shape, capacity):
        inputs = [ts(INT64, [13]), ts(INT64, [4]), ts(INT64, [6]), ts(INT32, [3])]
        inputs += [None] * 6
        inputs[index] = ts(INT32 if index in (3, 9) else INT64, shape)
        attrs = {
            "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
            "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
            "state_update_capacity": ir.AttrInt64("state_update_capacity", capacity),
        }
        self._invalid("VarlenNGramHashMapping", inputs, attrs, 4)

    def test_ngram_invalid_order_logged(self):
        self._invalid(
            "NGramHashMapping",
            [ts(INT64, [2, 7]), ts(INT64), ts(INT64)],
            {
                "max_ngram_size": ir.AttrInt64("max_ngram_size", 1),
                "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
            },
        )

    @parameterized.parameterized.expand(
        [
            ("input_rank", 0, [13], 0),
            ("weight_rank", 1, [64, 4], 0),
            ("offset_rank", 2, [3, 1], 0),
            ("state_rank", 4, [2, 64], 0),
            ("offset_size", 2, [1], 0),
            ("tokens", 0, [1, 64], 0),
            ("state_geometry", 4, [2, 63, 3], 0),
            ("count_rank", 5, [], 1),
            ("count_batch", 5, [3], 1),
            ("capacity", 0, [13, 64], 9),
        ]
    )
    def test_conv_invalid(self, _name, index, shape, capacity):
        inputs = [
            ts(HALF, [13, 64]),
            ts(HALF, [64, 1, 4]),
            ts(INT32, [3]),
            None,
            ts(HALF, [2, 64, 3]),
            None,
        ]
        inputs[index] = ts(INT32 if index in (2, 5) else HALF, shape)
        self._invalid(
            "VarlenCausalConvWithState",
            inputs,
            {"state_update_capacity": ir.AttrInt64("state_update_capacity", capacity)},
            3,
        )

    def test_conv_invalid_dilation_logged(self):
        self._invalid(
            "VarlenCausalConvWithState",
            [ts(HALF), ts(HALF), ts(INT32), None, ts(HALF)],
            {"dilation": ir.AttrInt64("dilation", 0)},
        )

    @parameterized.parameterized.expand(
        [
            ("query_rank", False, 0, [14, 64]),
            ("key_rank", False, 1, [14, 64]),
            ("value_rank", False, 2, [14, 256]),
            ("state_rank", False, 6, [2, 8, 32]),
            ("zero_head", False, 0, [14, 0, 16]),
            ("packed_zero", True, 6, [2, 8, 32, 0]),
            ("packed_width", True, 0, [14, 385]),
            ("packed_heads", True, 0, [14, 352]),
            ("uniform_tokens", True, 0, [13, 384]),
            ("uniform_batch", True, 6, [0, 8, 32, 16]),
        ]
    )
    def test_gdn_invalid(self, _name, packed, index, shape):
        inputs = (
            [ts(HALF, [14, 384]), None, None, None, None, None, ts(FLOAT, [2, 8, 32, 16])]
            if packed
            else [
                ts(HALF, [14, 4, 16]),
                ts(HALF, [14, 4, 16]),
                ts(HALF, [14, 8, 32]),
                None,
                None,
                None,
                None,
            ]
        )
        inputs[index] = ts(FLOAT if index == 6 else HALF, shape)
        self._invalid("GatedDeltaNet", inputs)

    def test_gdn_invalid_capacity(self):
        self._invalid(
            "GatedDeltaNet",
            [ts(HALF)] * 3,
            {"state_update_capacity": ir.AttrInt64("state_update_capacity", 9)},
        )

    def test_gdn_update_overflow(self):
        self._invalid(
            "GatedDeltaNet",
            [
                ts(HALF, [14, 4, 16]),
                ts(HALF, [14, 4, 16]),
                ts(HALF, [14, 8, 2**62]),
                None,
                None,
                None,
                None,
                None,
                None,
                ts(INT32, [2]),
            ],
            {"state_update_capacity": ir.AttrInt64("state_update_capacity", 8)},
            3,
        )

    @parameterized.parameterized.expand(
        [("GatedDeltaNet", 9), ("VarlenCausalConvWithState", 5), ("VarlenNGramHashMapping", 9)]
    )
    def test_capture_count_presence_contract(self, op, count_index):
        for capacity, supplied in ((1, False), (0, True)):
            if op == "GatedDeltaNet":
                inputs = [ts(HALF)] * 3
                attrs = {}
            elif op == "VarlenCausalConvWithState":
                inputs = [ts(HALF), ts(HALF), ts(INT32), None, ts(HALF)]
                attrs = {}
            else:
                inputs = [ts(INT64)] * 3 + [ts(INT32)]
                attrs = {
                    "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
                    "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
                }
            inputs += [None] * (count_index + 1 - len(inputs))
            if supplied:
                inputs[count_index] = ts(INT32, [2])
            attrs["state_update_capacity"] = ir.AttrInt64("state_update_capacity", capacity)
            with self.assertRaises(OpUsageError):
                run_shape_inference(MS, op, inputs, attrs, opset_version=1, num_outputs=2)

    @parameterized.parameterized.expand(
        [
            ("missing_key", [ts(HALF), None, ts(HALF)]),
            ("missing_value", [ts(HALF), ts(HALF), None]),
        ]
    )
    def test_gdn_mismatched_optional_inputs(self, _name, inputs):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MS, "GatedDeltaNet", inputs, opset_version=1)

    @parameterized.parameterized.expand(
        [
            ("missing_value_shape", [ts(HALF, [14, 4, 16]), ts(HALF), ts(HALF)]),
            (
                "missing_packed_state_shape",
                [ts(HALF, [14, 384]), None, None, None, None, None, ts(FLOAT)],
            ),
        ]
    )
    def test_gdn_partial_shapes(self, _name, inputs):
        self.assertEqual(
            run_shape_inference(MS, "GatedDeltaNet", inputs, opset_version=1, num_outputs=2),
            [ts(HALF), ts(FLOAT)],
        )

    @parameterized.parameterized.expand([("H", "H", "H", "_d0"), ("H", "G", "_d0", "_d1")])
    def test_gdn_symbolic_head_counts(self, q_heads, v_heads, out_heads, batch):
        self.assertEqual(
            run_shape_inference(
                MS,
                "GatedDeltaNet",
                [
                    ts(HALF, ["T", q_heads, "K"]),
                    ts(HALF, ["T", q_heads, "K"]),
                    ts(HALF, ["T", v_heads, "V"]),
                ],
                opset_version=1,
                num_outputs=2,
            ),
            [ts(HALF, ["T", out_heads, "V"]), ts(FLOAT, [batch, v_heads, "V", "K"])],
        )

    def test_conv_unknown_offsets_uses_initial_batch(self):
        self.assertEqual(
            run_shape_inference(
                MS,
                "VarlenCausalConvWithState",
                [ts(HALF, [13, 64]), ts(HALF), ts(INT32), None, ts(HALF, [2, 64, 3])],
                opset_version=1,
                num_outputs=2,
            ),
            [ts(HALF, [13, 64]), ts(HALF, [2, 64, 3])],
        )

    def test_conv_unknown_batch_and_channels(self):
        self.assertEqual(
            run_shape_inference(
                MS,
                "VarlenCausalConvWithState",
                [ts(HALF), ts(HALF), ts(INT32), None, ts(HALF)],
                opset_version=1,
                num_outputs=3,
            ),
            [ts(HALF), ts(HALF), ts(HALF, ["_d0", 0, "_d1"])],
        )

    def test_gdn_symbolic_packed_width(self):
        self.assertEqual(
            run_shape_inference(
                MS,
                "GatedDeltaNet",
                [
                    ts(HALF, [14, "P"]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    ts(FLOAT, [2, 8, "V", "K"]),
                ],
                opset_version=1,
                num_outputs=2,
            ),
            [ts(HALF, [14, 8, "V"]), ts(FLOAT, [2, 8, "V", "K"])],
        )

    def test_gdn_without_optional_outputs(self):
        self.assertEqual(
            run_shape_inference(
                MS,
                "GatedDeltaNet",
                [ts(HALF, [14, 4, 16]), ts(HALF, [14, 4, 16]), ts(HALF, [14, 8, 32])],
                opset_version=1,
            ),
            [ts(HALF, [14, 8, 32])],
        )

    @parameterized.parameterized.expand(
        [
            ("EngramGate", [HALF] * 5, {}),
            (
                "VarlenNGramHashMapping",
                [INT64] * 3 + [INT32],
                {
                    "max_ngram_size": ir.AttrInt64("max_ngram_size", 4),
                    "n_head_per_ngram": ir.AttrInt64("n_head_per_ngram", 2),
                },
            ),
            ("GatedDeltaNet", [HALF] * 3, {}),
        ]
    )
    def test_empty_optional_output(self, op, dtypes, attrs):
        inputs = [
            ir.Value(name=f"in{i}", type=ir.TensorType(dtype))
            for i, dtype in enumerate(dtypes)
        ]
        outputs = [ir.Value(name="out"), ir.Value(name="")]
        node = ir.Node(MS, op, inputs, attributes=attrs, outputs=outputs)
        registry.get(MS, op, 1)(ShapeInferenceContext({MS: 1}), node)
        self.assertIsNotNone(outputs[0].type)
        self.assertIsNone(outputs[1].type)
