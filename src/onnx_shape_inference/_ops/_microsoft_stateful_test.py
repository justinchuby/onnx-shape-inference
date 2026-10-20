# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the recurrent and Engram schemas released in ORT 1.31."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError
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
