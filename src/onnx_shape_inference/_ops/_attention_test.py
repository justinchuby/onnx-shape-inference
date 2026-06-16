# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Attention and RotaryEmbedding shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import OpUsageError
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class AttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_q_shape_propagated(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32]), ts(FLOAT, [2, 8, 32])],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 32])])

    def test_missing_q_shape_falls_back_to_v(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
        )
        # Q has no shape → output shape is None (can't determine without Q)
        self.assertIsNone(actual[0].shape)

    def test_missing_all_shapes(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT), ts(FLOAT), ts(FLOAT)],
            opset_version=23,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Attention", [], opset_version=23)

    def test_none_input(self):
        v = ir.Value(name="q", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 10, 64]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Attention",
                [v, None, None],
                opset_version=23,
            )

    def test_present_key_value_outputs(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64]), ts(FLOAT, [1, 10, 64])],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[2], ts(FLOAT, [1, 10, 64]))


class RotaryEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT, [1, 10, 64])],
            opset_version=24,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 64])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "RotaryEmbedding",
            [ts(FLOAT)],
            opset_version=24,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "RotaryEmbedding", [], opset_version=24)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "RotaryEmbedding",
                [None],
                opset_version=24,
            )

    def test_rotary_embedding_with_position_ids_two_outputs(self):
        input_val = ir.Value(
            name="input", shape=ir.Shape([1, 10, 64]), type=ir.TensorType(FLOAT)
        )
        position_ids = ir.Value(
            name="position_ids", shape=ir.Shape([1, 10]), type=ir.TensorType(INT64)
        )
        actual = run_shape_inference_with_values(
            "",
            "RotaryEmbedding",
            [input_val, position_ids],
            opset_version=24,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 10, 64]))
        self.assertEqual(actual[1], ts(INT64, [1, 10]))


class AttentionSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
                ts(FLOAT, ["B", "S", "D"]),
            ],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, ["B", "S", "D"])])

    def test_symbolic_with_concrete_embed_dim(self):
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
                ts(FLOAT, ["B", "S", 64]),
            ],
            opset_version=23,
        )
        self.assertEqual(actual, [ts(FLOAT, ["B", "S", 64])])

    def test_4d_concrete(self):
        """4D inputs: [batch, num_heads, seq_len, head_size]."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
                ts(FLOAT, [2, 8, 10, 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 10, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 10, 64]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 8, 10, 64]))

    def test_4d_symbolic(self):
        """4D inputs with symbolic batch and seq dims."""
        actual = run_shape_inference(
            "",
            "Attention",
            [
                ts(FLOAT, ["B", 8, "S", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
                ts(FLOAT, ["B", 8, "Sk", 64]),
            ],
            opset_version=23,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, ["B", 8, "S", 64]))
        self.assertEqual(actual[1], ts(FLOAT, ["B", 8, "Sk", 64]))
        self.assertEqual(actual[2], ts(FLOAT, ["B", 8, "Sk", 64]))


def _attrs(**kw) -> dict:
    out = {}
    for name, val in kw.items():
        if isinstance(val, int):
            out[name] = ir.Attr(name, ir.AttributeType.INT, val)
        elif isinstance(val, str):
            out[name] = ir.Attr(name, ir.AttributeType.STRING, val)
        elif isinstance(val, float):
            out[name] = ir.Attr(name, ir.AttributeType.FLOAT, val)
    return out


class LinearAttentionTest(unittest.TestCase):
    def test_prefill_no_past(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32])],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 32]))
        # present_state [B, kv_num_heads, d_k, d_v] = [2, 4, 8, 8]
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_gqa_output_hidden(self):
        """GQA: output hidden = q_num_heads * (value_hidden // kv_num_heads)."""
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 64]), ts(FLOAT, [2, 4, 32]), ts(FLOAT, [2, 4, 32])],
            attributes=_attrs(q_num_heads=8, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_mqa_present_state(self):
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [ts(FLOAT, [2, 4, 64]), ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8])],
            attributes=_attrs(q_num_heads=8, kv_num_heads=1),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 1, 8, 8]))

    def test_past_state_is_ground_truth(self):
        """present_state copies past_state when provided."""
        actual = run_shape_inference_with_values(
            "",
            "LinearAttention",
            [
                ir.Value(name="q", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="k", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="v", shape=ir.Shape([2, 4, 32]), type=ir.TensorType(FLOAT)),
                ir.Value(name="ps", shape=ir.Shape([2, 4, 8, 8]), type=ir.TensorType(FLOAT)),
            ],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, 4, 8, 8]))

    def test_symbolic_hidden_symbolic_heads(self):
        """Symbolic packed hidden -> output hidden and head dims symbolic."""
        actual = run_shape_inference(
            "",
            "LinearAttention",
            [
                ts(FLOAT, ["B", "T", "Dq"]),
                ts(FLOAT, ["B", "T", "Dk"]),
                ts(FLOAT, ["B", "T", "Dv"]),
            ],
            attributes=_attrs(q_num_heads=4, kv_num_heads=4),
            opset_version=27,
            num_outputs=2,
        )
        out0 = actual[0].shape
        self.assertEqual(str(out0[0]), "B")
        self.assertEqual(str(out0[1]), "T")
        self.assertIsInstance(out0[2], ir.SymbolicDim)
        present = actual[1].shape
        self.assertEqual(present[1], 4)
        self.assertIsInstance(present[2], ir.SymbolicDim)
        self.assertIsInstance(present[3], ir.SymbolicDim)


class FlexAttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [1, 2, 4, 8]), ts(FLOAT, [1, 2, 4, 8]), ts(FLOAT, [1, 2, 4, 8])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 2, 4, 8]))

    def test_diff_head_sizes(self):
        """Y v_head_size comes from V's last dim; q_seq from Q."""
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 4, 8, 16]), ts(FLOAT, [2, 4, 6, 16]), ts(FLOAT, [2, 4, 6, 32])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 8, 32]))

    def test_gqa(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 8, 4, 16]), ts(FLOAT, [2, 2, 6, 16]), ts(FLOAT, [2, 2, 6, 16])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 4, 16]))

    def test_symbolic(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [
                ts(FLOAT, ["B", "Hq", "Sq", "D"]),
                ts(FLOAT, ["B", "Hkv", "Sk", "D"]),
                ts(FLOAT, ["B", "Hkv", "Sk", "Dv"]),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, ["B", "Hq", "Sq", "Dv"]))

    def test_non_rank4_falls_back_to_q(self):
        actual = run_shape_inference(
            "ai.onnx.preview",
            "FlexAttention",
            [ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8]), ts(FLOAT, [2, 4, 8])],
            opset_version=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 4, 8]))


if __name__ == "__main__":
    unittest.main()
