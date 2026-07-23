# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for com.microsoft custom operator shape inference."""

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
FLOAT16 = ir.DataType.FLOAT16
INT32 = ir.DataType.INT32
INT64 = ir.DataType.INT64

MSFT = "com.microsoft"


class PassthroughTest(unittest.TestCase):
    """Passthrough ops: output = input[0]."""

    def test_bias_gelu(self):
        actual = run_shape_inference(
            MSFT, "BiasGelu", [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [64])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_fast_gelu(self):
        actual = run_shape_inference(
            MSFT, "FastGelu", [ts(FLOAT, [2, 8, 64])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_gelu(self):
        actual = run_shape_inference(MSFT, "Gelu", [ts(FLOAT, [4, 16, 128])], opset_version=1)
        self.assertEqual(actual, [ts(FLOAT, [4, 16, 128])])

    def test_quick_gelu(self):
        actual = run_shape_inference(
            MSFT, "QuickGelu", [ts(FLOAT, [1, 10, 32])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 32])])

    def test_bias_add(self):
        actual = run_shape_inference(
            MSFT, "BiasAdd", [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [64])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_longformer_attention(self):
        actual = run_shape_inference(
            MSFT, "LongformerAttention", [ts(FLOAT, [1, 512, 64])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 512, 64])])

    def test_group_norm(self):
        actual = run_shape_inference(
            MSFT, "GroupNorm", [ts(FLOAT, [2, 32, 8, 8])], opset_version=1
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 32, 8, 8])])


class SkipGroupNormTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "SkipGroupNorm",
            [ts(FLOAT, [2, 32, 8, 8])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 32, 8, 8]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 32, 8, 8]))


class BiasSplitGeluTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "BiasSplitGelu",
            [ts(FLOAT, [2, 8, 128]), ts(FLOAT, [256])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 128])])

    def test_halves_bias(self):
        actual = run_shape_inference(
            MSFT,
            "BiasSplitGelu",
            [ts(FLOAT, [1, 4, 64]), ts(FLOAT, [128])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 4, 64])])


class GemmLikeTest(unittest.TestCase):
    def test_gemm_fast_gelu(self):
        actual = run_shape_inference(
            MSFT,
            "GemmFastGelu",
            [ts(FLOAT, [2, 64]), ts(FLOAT, [64, 128])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 128])])

    def test_gemm_float8(self):
        actual = run_shape_inference(
            MSFT,
            "GemmFloat8",
            [ts(FLOAT, [4, 32]), ts(FLOAT, [32, 16])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 16])])


class RotaryEmbeddingTest(unittest.TestCase):
    def test_single_output(self):
        actual = run_shape_inference(
            MSFT,
            "RotaryEmbedding",
            [ts(FLOAT, [2, 8, 4, 16])],
            opset_version=1,
            num_outputs=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 4, 16])])

    def test_two_outputs(self):
        actual = run_shape_inference(
            MSFT,
            "RotaryEmbedding",
            [ts(FLOAT, [2, 8, 4, 16]), ts(FLOAT, [1, 8, 32])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 8, 32]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 4, 16]))

    def test_three_outputs(self):
        actual = run_shape_inference(
            MSFT,
            "RotaryEmbedding",
            [ts(FLOAT, [2, 8, 4, 16]), ts(FLOAT, [1, 8, 32])],
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 8, 32]))
        self.assertEqual(actual[1], ts(FLOAT, [1, 8, 32]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 8, 4, 16]))


class MsLayerNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "LayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 1]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 8, 1]))

    def test_axis_1(self):
        actual = run_shape_inference(
            MSFT,
            "LayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, 1, 1]))

    def test_float16_upcasts_mean(self):
        actual = run_shape_inference(
            MSFT,
            "LayerNormalization",
            [ts(FLOAT16, [2, 8, 64])],
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT16, [2, 8, 64]))
        # Mean/inv_std_dev upcast to FLOAT
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 1]))

    def test_simplified(self):
        actual = run_shape_inference(
            MSFT,
            "SimplifiedLayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))

    def test_simplified_inv_std_dev(self):
        # SimplifiedLayerNormalization output[1] is inv_std_var with reduced shape.
        actual = run_shape_inference(
            MSFT,
            "SimplifiedLayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 1]))

    def test_simplified_float16_upcasts_inv_std_dev(self):
        actual = run_shape_inference(
            MSFT,
            "SimplifiedLayerNormalization",
            [ts(FLOAT16, [2, 8, 64])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT16, [2, 8, 64]))
        # inv_std_var upcasts to FLOAT for float16 input
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 1]))

    def test_simplified_default_domain(self):
        # ONNX Runtime may place SimplifiedLayerNormalization under the default
        # opset domain instead of com.microsoft; it should still be inferred.
        actual = run_shape_inference(
            "",
            "SimplifiedLayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        self.assertEqual(actual[1], ts(FLOAT, [2, 8, 1]))


class MsSkipLayerNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "SkipLayerNormalization",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=4,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        self.assertEqual(actual[3], ts(FLOAT, [2, 8, 64]))

    def test_skip_simplified(self):
        actual = run_shape_inference(
            MSFT,
            "SkipSimplifiedLayerNormalization",
            [ts(FLOAT, [1, 4, 32])],
            opset_version=1,
            num_outputs=4,
        )
        self.assertEqual(actual[0], ts(FLOAT, [1, 4, 32]))
        self.assertEqual(actual[3], ts(FLOAT, [1, 4, 32]))


class EmbedLayerNormalizationTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "EmbedLayerNormalization",
            [ts(INT32, [2, 128]), ts(INT32, [2, 128]), ts(FLOAT, [30522, 768])],
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 128, 768]))
        self.assertEqual(actual[1].shape, ir.Shape([2]))
        self.assertEqual(actual[2], ts(FLOAT, [2, 128, 768]))


class MsAttentionTest(unittest.TestCase):
    def test_basic_3d(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [2, 8, 192]), ts(FLOAT, [192, 576]), ts(FLOAT, [576])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 192])])

    def test_missing_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "Attention", [], opset_version=1)

    def test_qkv_hidden_sizes(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [2, 8, 192]), ts(FLOAT, [192, 576]), ts(FLOAT, [576])],
            attributes={
                "qkv_hidden_sizes": ir.Attr(
                    "qkv_hidden_sizes", ir.AttributeType.INTS, [192, 192, 256]
                )
            },
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 256])])

    def test_present_appends_current_sequence(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [
                ts(FLOAT, [2, 8, 192]),
                ts(FLOAT, [192, 576]),
                ts(FLOAT, [576]),
                None,
                ts(FLOAT, [2, 2, 4, 10, 48]),
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, 2, 4, 18, 48]))

    def test_present_appends_symbolic_current_sequence(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [
                ts(FLOAT, ["batch", "sequence", 192]),
                ts(FLOAT, [192, 576]),
                ts(FLOAT, [576]),
                None,
                ts(FLOAT, [2, "batch", 4, "past_sequence", 48]),
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, "batch", 4, "past_sequence + sequence", 48]))

    def test_present_shared_buffer_keeps_buffer_length(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [
                ts(FLOAT, [2, 8, 192]),
                ts(FLOAT, [192, 576]),
                ts(FLOAT, [576]),
                None,
                ts(FLOAT, [2, 2, 4, 128, 48]),
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "past_present_share_buffer": ir.Attr(
                    "past_present_share_buffer", ir.AttributeType.INT, 1
                ),
            },
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[1], ts(FLOAT, [2, 2, 4, 128, 48]))


class MsMultiHeadAttentionTest(unittest.TestCase):
    def test_3d_query(self):
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [2, 10, 64]), ts(FLOAT, [2, 10, 32])],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 32]))


class PackedAttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "PackedAttention",
            [ts(FLOAT, [16, 192]), ts(FLOAT, [192, 576]), ts(FLOAT, [576])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [16, 192])])

    def test_qkv_hidden_sizes(self):
        actual = run_shape_inference(
            MSFT,
            "PackedAttention",
            [ts(FLOAT, [100, 64]), ts(FLOAT, [64, 192]), ts(FLOAT, [192])],
            attributes={
                "qkv_hidden_sizes": ir.Attr(
                    "qkv_hidden_sizes", ir.AttributeType.INTS, [64, 64, 128]
                ),
            },
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([100, 128]))

    def test_non_2d_fallback(self):
        actual = run_shape_inference(
            MSFT,
            "PackedAttention",
            [ts(FLOAT, [2, 8, 192])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 192]))

    def test_hidden_from_weights(self):
        """Hidden inferred from weights when bias is absent."""
        actual = run_shape_inference(
            MSFT,
            "PackedAttention",
            [ts(FLOAT, [100, 192]), ts(FLOAT, [192, 576])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([100, 192]))


class DecoderMaskedMHATest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "DecoderMaskedMultiHeadAttention",
            [ts(FLOAT, [2, 1, 64])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 1, 64])])

    def test_with_past(self):
        actual = run_shape_inference(
            MSFT,
            "DecoderMaskedMultiHeadAttention",
            [
                ts(FLOAT, [2, 1, 64]),  # query
                None,
                None,
                None,
                None,
                ts(FLOAT, [2, 4, 10, 16]),  # past key
            ],
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 1, 64]))
        self.assertEqual(actual[1].shape, ir.Shape([2, 4, 10, 16]))
        self.assertEqual(actual[2].shape, ir.Shape([2, 4, 10, 16]))


class RemovePaddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "RemovePadding",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
            num_outputs=4,
        )
        self.assertEqual(actual[0].shape.rank(), 2)
        self.assertEqual(actual[0].shape[1], 64)
        self.assertEqual(actual[1].shape, ir.Shape([2, 8]))
        self.assertEqual(actual[3].shape, ir.Shape([1]))

    def test_non_3d_returns(self):
        actual = run_shape_inference(
            MSFT,
            "RemovePadding",
            [ts(FLOAT, [8, 64])],
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)


class RestorePaddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "RestorePadding",
            [ts(FLOAT, [16, 64]), ts(INT32, [2, 8])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_incompatible_ranks(self):
        actual = run_shape_inference(
            MSFT,
            "RestorePadding",
            [ts(FLOAT, [100, 64]), ts(INT32, [16])],
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)


class GatedRelativePositionBiasTest(unittest.TestCase):
    def test_from_query(self):
        actual = run_shape_inference(
            MSFT,
            "GatedRelativePositionBias",
            [ts(FLOAT, [4, 8, 32])],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 4, 8, 8])])

    def test_from_token_offset(self):
        actual = run_shape_inference(
            MSFT,
            "GatedRelativePositionBias",
            [
                ts(FLOAT, [2, 8, 64]),  # query_layer
                None,
                None,
                None,
                None,
                None,
                ts(INT32, [2, 8]),  # token_offset
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 4, 8, 8]))

    def test_no_num_heads(self):
        """No num_heads attribute — no output shape set."""
        actual = run_shape_inference(
            MSFT,
            "GatedRelativePositionBias",
            [ts(FLOAT, [2, 8, 64])],
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)


class GroupQueryAttentionTest(unittest.TestCase):
    def test_basic_non_packed(self):
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [
                ts(FLOAT, [2, 8, 64]),  # query
                ts(FLOAT, [2, 8, 16]),  # key
                ts(FLOAT, [2, 8, 16]),  # value
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))

    def test_packed_qkv(self):
        # packed: d = (4 + 2*1) * 16 = 96, output hidden = 4*16 = 64
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [ts(FLOAT, [2, 8, 96])],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        # present_key: [2, 1, ?, 16]
        self.assertEqual(actual[1].shape[0], 2)
        self.assertEqual(actual[1].shape[1], 1)
        self.assertEqual(actual[1].shape[3], 16)

    def test_present_with_past(self):
        # Non-packed decode: query hidden=64 (num_heads=4, head_size=16),
        # key/value hidden=16 (kv_num_heads=1, head_size=16), past seq=10.
        actual = run_shape_inference_with_values(
            MSFT,
            "GroupQueryAttention",
            [
                ir.Value(name="q", shape=ir.Shape([2, 1, 64]), type=ir.TensorType(FLOAT)),
                ir.Value(name="k", shape=ir.Shape([2, 1, 16]), type=ir.TensorType(FLOAT)),
                ir.Value(name="v", shape=ir.Shape([2, 1, 16]), type=ir.TensorType(FLOAT)),
                ir.Value(name="pk", shape=ir.Shape([2, 1, 10, 16]), type=ir.TensorType(FLOAT)),
                ir.Value(name="pv", shape=ir.Shape([2, 1, 10, 16]), type=ir.TensorType(FLOAT)),
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 1, 64]))
        # present_key: [2, 1, 11, 16]  (past_seq=10 + cur_seq=1)
        self.assertEqual(actual[1].shape, ir.Shape([2, 1, 11, 16]))

    def test_non_packed_hidden_divisible_by_packed_total(self):
        """Non-packed query with hidden divisible by the packed total stays non-packed.

        Regression guard: ``num_heads + 2*kv_num_heads`` dividing the hidden size
        must NOT trigger packed detection.  Here hidden=128 == 8 * (8 + 2*4), but
        key/value are present so the layout is non-packed and output hidden=128.
        """
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [
                ts(FLOAT, [2, 8, 128]),  # query (num_heads=8, head_size=16)
                ts(FLOAT, [2, 8, 64]),  # key   (kv_num_heads=4, head_size=16)
                ts(FLOAT, [2, 8, 64]),  # value
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 8),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 4),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 128]))
        # present_key: [2, kv_num_heads=4, seq=8, head_size=16]
        self.assertEqual(actual[1].shape, ir.Shape([2, 4, 8, 16]))

    def test_packed_indivisible_hidden_symbolic(self):
        """Packed QKV with an indivisible hidden keeps output/present symbolic.

        Hidden not divisible by the head total leaves the output hidden and
        present head_size symbolic rather than truncating.
        """
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [ts(FLOAT, [2, 8, 100])],  # 100 % (4 + 2*1) != 0
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        # Output hidden is data-dependent -> symbolic.
        self.assertEqual(actual[0].shape[0], 2)
        self.assertEqual(actual[0].shape[1], 8)
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)
        # present head_size also symbolic.
        self.assertIsInstance(actual[1].shape[3], ir.SymbolicDim)

    def test_packed_symbolic_hidden(self):
        """A symbolic query hidden size leaves packed head_size unknown."""
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [ts(FLOAT, ["B", "S", "D"])],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)
        # present: [B, 1, S, <symbolic head_size>]
        self.assertEqual(str(actual[1].shape[0]), "B")
        self.assertEqual(actual[1].shape[1], 1)
        self.assertIsInstance(actual[1].shape[3], ir.SymbolicDim)

    def test_past_key_symbolic_kv_seq_fresh_dim(self):
        """When the current kv length is unknown, present_seq is a fresh dim.

        Packed QKV with a rank-2 (non-3D) query means the kv length can't be
        read, so the present sequence dim is symbolic while batch / kv_num_heads
        / head_size still come from past_key.
        """
        actual = run_shape_inference_with_values(
            MSFT,
            "GroupQueryAttention",
            [
                # rank-2 query: _kv_sequence_length() returns None in packed mode
                ir.Value(name="q", shape=ir.Shape([2, 64]), type=ir.TensorType(FLOAT)),
                None,
                None,
                ir.Value(name="pk", shape=ir.Shape([2, 1, 10, 16]), type=ir.TensorType(FLOAT)),
                ir.Value(name="pv", shape=ir.Shape([2, 1, 10, 16]), type=ir.TensorType(FLOAT)),
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        present = actual[1].shape
        self.assertEqual(present[0], 2)
        self.assertEqual(present[1], 1)
        self.assertIsInstance(present[2], ir.SymbolicDim)  # unknown present_seq
        self.assertEqual(present[3], 16)

    def test_single_output_skips_present(self):
        """With a single output, present_key/value inference is skipped."""
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [ts(FLOAT, [2, 8, 96])],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=1,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))

    def test_packed_missing_kv_num_heads_symbolic(self):
        """Packed QKV without kv_num_heads cannot resolve head_size -> symbolic."""
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [ts(FLOAT, [2, 8, 96])],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertIsInstance(actual[0].shape[2], ir.SymbolicDim)

    def test_non_packed_indivisible_prefill_present_symbolic(self):
        """Non-packed prefill with an indivisible hidden keeps head_size symbolic.

        When the hidden size is not divisible by num_heads, the present
        head_size stays symbolic.
        """
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [
                ts(FLOAT, [2, 8, 100]),  # 100 % 3 != 0
                ts(FLOAT, [2, 8, 30]),
                ts(FLOAT, [2, 8, 30]),
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 3),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        # output hidden stays as the query hidden (non-packed); present head_size symbolic.
        self.assertEqual(actual[0].shape[2], 100)
        self.assertIsInstance(actual[1].shape[3], ir.SymbolicDim)

    def test_non_packed_rank2_kv_present_seq_symbolic(self):
        """Non-packed rank-2 key/value makes the prefill present_seq symbolic.

        With a rank-2 (not 3-D) key/value the kv length can't be read, so the
        present sequence dim is symbolic.
        """
        q = ir.Value(name="q", shape=ir.Shape([2, 8, 64]), type=ir.TensorType(FLOAT))
        k = ir.Value(name="k", shape=ir.Shape([2, 16]), type=ir.TensorType(FLOAT))
        v = ir.Value(name="v", shape=ir.Shape([2, 16]), type=ir.TensorType(FLOAT))
        actual = run_shape_inference_with_values(
            MSFT,
            "GroupQueryAttention",
            [q, k, v],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        present = actual[1].shape
        self.assertEqual(present[0], 2)
        self.assertEqual(present[1], 1)
        self.assertIsInstance(present[2], ir.SymbolicDim)  # unknown present_seq


class MatMulNBitsTest(unittest.TestCase):
    def test_basic_2d(self):
        # A=[4,128], B=[64,4,16] (packed), scales=[64,4]
        actual = run_shape_inference(
            MSFT,
            "MatMulNBits",
            [ts(FLOAT, [4, 128]), ts(None, [64, 4, 16]), ts(FLOAT, [64, 4])],
            attributes={
                "K": ir.Attr("K", ir.AttributeType.INT, 128),
                "N": ir.Attr("N", ir.AttributeType.INT, 64),
                "block_size": ir.Attr("block_size", ir.AttributeType.INT, 32),
            },
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 64])])

    def test_batched(self):
        actual = run_shape_inference(
            MSFT,
            "MatMulNBits",
            [ts(FLOAT, [2, 8, 128]), ts(None, [64, 4, 16]), ts(FLOAT, [64, 4])],
            attributes={
                "K": ir.Attr("K", ir.AttributeType.INT, 128),
                "N": ir.Attr("N", ir.AttributeType.INT, 64),
                "block_size": ir.Attr("block_size", ir.AttributeType.INT, 32),
            },
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_missing_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "MatMulNBits", [], opset_version=1)

    def test_missing_n_attr_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MSFT,
                "MatMulNBits",
                [ts(FLOAT, [4, 128]), ts(None, [64, 4, 16]), ts(FLOAT, [64, 4])],
                attributes={
                    "K": ir.Attr("K", ir.AttributeType.INT, 128),
                    "block_size": ir.Attr("block_size", ir.AttributeType.INT, 32),
                },
                opset_version=1,
            )


class SparseAttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "SparseAttention",
            [ts(FLOAT, [2, 8, 64])],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        # present_key: [2, 1, ?, 16]
        self.assertEqual(actual[1].shape[0], 2)
        self.assertEqual(actual[1].shape[1], 1)
        self.assertEqual(actual[1].shape[3], 16)


class FusedMatMulTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "FusedMatMul",
            [ts(FLOAT, [4, 32]), ts(FLOAT, [32, 16])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 16])])

    def test_trans_b(self):
        actual = run_shape_inference(
            MSFT,
            "FusedMatMul",
            [ts(FLOAT, [4, 32]), ts(FLOAT, [16, 32])],
            attributes={"transB": ir.Attr("transB", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 16])])

    def test_trans_a(self):
        actual = run_shape_inference(
            MSFT,
            "FusedMatMul",
            [ts(FLOAT, [32, 4]), ts(FLOAT, [32, 16])],
            attributes={"transA": ir.Attr("transA", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 16])])

    def test_transpose_matmul(self):
        actual = run_shape_inference(
            MSFT,
            "TransposeMatMul",
            [ts(FLOAT, [4, 32]), ts(FLOAT, [32, 16])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [4, 16])])


class MoETest(unittest.TestCase):
    def test_basic_2d(self):
        actual = run_shape_inference(
            MSFT,
            "MoE",
            [ts(FLOAT, [16, 64]), ts(FLOAT, [16, 8])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [16, 64])])

    def test_basic_3d(self):
        actual = run_shape_inference(
            MSFT,
            "MoE",
            [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [16, 8])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])

    def test_qmoe(self):
        actual = run_shape_inference(
            MSFT,
            "QMoE",
            [ts(FLOAT, [16, 64]), ts(FLOAT, [16, 8])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [16, 64])])


class QAttentionTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QAttention",
            [
                ts(None, [2, 8, 64]),  # input (int8)
                ts(None, [64, 192]),  # weight
                ts(FLOAT, [192]),  # bias
                ts(FLOAT, []),  # input_scale
                ts(FLOAT, []),  # weight_scale
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0], ts(FLOAT, [2, 8, 64]))
        # present: [2, 2, 4, 8, 16]
        self.assertEqual(actual[1].shape, ir.Shape([2, 2, 4, 8, 16]))


class QGemmTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QGemm",
            [
                ts(None, [4, 32]),  # A
                ts(FLOAT, []),  # a_scale
                ts(None, []),  # a_zp
                ts(None, [32, 16]),  # B
                ts(FLOAT, []),  # b_scale
                ts(None, []),  # b_zp
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))

    def test_trans_b(self):
        actual = run_shape_inference(
            MSFT,
            "QGemm",
            [
                ts(None, [4, 32]),  # A
                ts(FLOAT, []),  # a_scale
                ts(None, []),  # a_zp
                ts(None, [16, 32]),  # B (transposed)
                ts(FLOAT, []),  # b_scale
                ts(None, []),  # b_zp
            ],
            attributes={"transB": ir.Attr("transB", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))


class QLinearBinaryTest(unittest.TestCase):
    def test_add(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearAdd",
            [
                ts(None, [2, 8]),  # A
                ts(FLOAT, []),  # A_scale
                ts(None, []),  # A_zp
                ts(None, [2, 8]),  # B
                ts(FLOAT, []),  # B_scale
                ts(None, []),  # B_zp
                ts(FLOAT, []),  # C_scale
                ts(None, []),  # C_zp
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))

    def test_mul(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearMul",
            [
                ts(None, [4, 16]),  # A
                ts(FLOAT, []),
                ts(None, []),
                ts(None, [4, 16]),  # B
                ts(FLOAT, []),
                ts(None, []),
                ts(FLOAT, []),
                ts(None, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))


class QLinearUnaryTest(unittest.TestCase):
    def test_sigmoid(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearSigmoid",
            [ts(None, [2, 8]), ts(FLOAT, []), ts(None, []), ts(FLOAT, []), ts(None, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))

    def test_leaky_relu(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearLeakyRelu",
            [ts(None, [4, 16]), ts(FLOAT, []), ts(None, []), ts(FLOAT, []), ts(None, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))


class QLinearConcatTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),  # Y_scale
                ts(None, []),  # Y_zp
                ts(None, [2, 4]),  # tensor1
                ts(FLOAT, []),  # scale1
                ts(None, []),  # zp1
                ts(None, [2, 6]),  # tensor2
                ts(FLOAT, []),  # scale2
                ts(None, []),  # zp2
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 10]))


class QOrderedTest(unittest.TestCase):
    def test_gelu(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedGelu",
            [ts(None, [2, 8, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))

    def test_matmul(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedMatMul",
            [ts(None, [4, 32]), ts(FLOAT, []), ts(None, [32, 16])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))


# ---------------------------------------------------------------------------
# Error path tests: OpUsageError on missing inputs
# ---------------------------------------------------------------------------


class ErrorPathTest(unittest.TestCase):
    """Test that missing inputs raise OpUsageError."""

    def test_passthrough_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "BiasGelu", [], opset_version=1)

    def test_skip_group_norm_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "SkipGroupNorm", [], opset_version=1)

    def test_rotary_embedding_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "RotaryEmbedding", [], opset_version=1)

    def test_layer_norm_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "SkipLayerNormalization", [], opset_version=1)

    def test_embed_layer_norm_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "EmbedLayerNormalization", [], opset_version=1)

    def test_attention_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "Attention", [], opset_version=1)

    def test_mha_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "MultiHeadAttention", [], opset_version=1)

    def test_packed_attention_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "PackedAttention", [], opset_version=1)

    def test_packed_mha_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "PackedMultiHeadAttention", [], opset_version=1)

    def test_decoder_masked_mha_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "DecoderMaskedMultiHeadAttention", [], opset_version=1)

    def test_gated_relative_pos_bias_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "GatedRelativePositionBias", [], opset_version=1)

    def test_remove_padding_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "RemovePadding", [], opset_version=1)

    def test_gqa_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "GroupQueryAttention", [], opset_version=1)

    def test_sparse_attention_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "SparseAttention", [], opset_version=1)

    def test_moe_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "MoE", [], opset_version=1)

    def test_qattention_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QAttention", [], opset_version=1)

    def test_qgemm_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QGemm", [], opset_version=1)

    def test_qlinear_unary_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QLinearSigmoid", [], opset_version=1)

    def test_qlinear_binary_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QLinearAdd", [], opset_version=1)

    def test_qlinear_conv_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QLinearConv", [], opset_version=1)

    def test_qlinear_where_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QLinearWhere", [], opset_version=1)

    def test_qordered_passthrough_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QOrderedGelu", [], opset_version=1)

    def test_qordered_matmul_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QOrderedMatMul", [], opset_version=1)

    def test_qembed_layer_norm_no_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(MSFT, "QEmbedLayerNormalization", [], opset_version=1)


# ---------------------------------------------------------------------------
# Attention with present state output
# ---------------------------------------------------------------------------


INT8 = ir.DataType.INT8


class AttentionPresentTest(unittest.TestCase):
    """Test Attention with present state (output[1])."""

    def test_present_no_past(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [2, 8, 192]), ts(FLOAT, [192, 576]), ts(FLOAT, [576])],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        # output: [2, 8, 192]
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 192]))
        # present: [2, 2, 4, 8, 48]  (head_size=192//4=48)
        self.assertEqual(actual[1].shape, ir.Shape([2, 2, 4, 8, 48]))

    def test_non_3d_input(self):
        """Non-3D input falls back to passthrough."""
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [8, 192])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([8, 192]))

    def test_qkv_hidden_sizes(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [64, 192]), ts(FLOAT, [192])],
            attributes={
                "qkv_hidden_sizes": ir.Attr(
                    "qkv_hidden_sizes", ir.AttributeType.INTS, [64, 64, 128]
                ),
            },
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 128]))

    def test_hidden_from_weights(self):
        """Hidden size inferred from weights when bias is missing."""
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [64, 192])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))

    def test_present_with_past(self):
        actual = run_shape_inference(
            MSFT,
            "Attention",
            [
                ts(FLOAT, [2, 8, 192]),  # input
                ts(FLOAT, [192, 576]),  # weights
                ts(FLOAT, [576]),  # bias
                None,  # mask
                ts(FLOAT, [2, 2, 4, 10, 48]),  # past
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[1].shape, ir.Shape([2, 2, 4, 18, 48]))


# ---------------------------------------------------------------------------
# MultiHeadAttention paths
# ---------------------------------------------------------------------------


class MultiHeadAttentionPathsTest(unittest.TestCase):
    def test_3d_with_value(self):
        """3D query with value input determines output hidden dim."""
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [
                ts(FLOAT, [2, 8, 64]),  # query
                ts(FLOAT, [2, 10, 64]),  # key
                ts(FLOAT, [2, 10, 128]),  # value
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 128]))
        # present key: [2, 4, 10, 16]
        self.assertEqual(actual[1].shape, ir.Shape([2, 4, 10, 16]))

    def test_5d_packed(self):
        """5D packed query: [B, Sq, Nh, 3, Hd]."""
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [ts(FLOAT, [2, 8, 4, 3, 16])],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=3,
        )
        # output: [2, 8, 64] = Nh*Hd = 4*16
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))

    def test_non_3d_5d_fallback(self):
        """Non-3D non-5D query falls through."""
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [ts(FLOAT, [8, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([8, 64]))

    def test_null_shape_returns(self):
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [ts(FLOAT, None)],
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)


# ---------------------------------------------------------------------------
# PackedAttention / PackedMultiHeadAttention
# ---------------------------------------------------------------------------


class PackedMHATest(unittest.TestCase):
    def test_2d_value(self):
        """2D value input determines output shape."""
        actual = run_shape_inference(
            MSFT,
            "PackedMultiHeadAttention",
            [ts(FLOAT, [100, 64]), None, ts(FLOAT, [100, 128])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([100, 128]))

    def test_4d_query(self):
        """4D query: [token, Nh, S, Hd] -> [token, Nh*Hd]."""
        actual = run_shape_inference(
            MSFT,
            "PackedMultiHeadAttention",
            [ts(FLOAT, [100, 4, 8, 16])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([100, 64]))

    def test_fallback(self):
        """Non-2D-value and non-4D-query falls through."""
        actual = run_shape_inference(
            MSFT,
            "PackedMultiHeadAttention",
            [ts(FLOAT, [100, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([100, 64]))


# ---------------------------------------------------------------------------
# GroupQueryAttention with present state
# ---------------------------------------------------------------------------


class GroupQueryAttentionPresentTest(unittest.TestCase):
    def test_with_past(self):
        # Non-packed decode with past. query hidden=64 (num_heads=4,
        # head_size=16); key/value hidden=32 (kv_num_heads=2, head_size=16);
        # past_key/value [2, 2, 10, 16].  Output keeps the query hidden size,
        # and present grows the past sequence by the current step (10 + 1).
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [
                ts(FLOAT, [2, 1, 64]),  # query
                ts(FLOAT, [2, 1, 32]),  # key
                ts(FLOAT, [2, 1, 32]),  # value
                ts(FLOAT, [2, 2, 10, 16]),  # past_key
                ts(FLOAT, [2, 2, 10, 16]),  # past_value
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 2),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 1, 64]))
        # present: [2, 2, 11, 16]  (past_seq=10 + cur_seq=1)
        self.assertEqual(actual[1].shape, ir.Shape([2, 2, 11, 16]))

    def test_non_packed(self):
        # Non-packed (key & value present): output hidden equals query hidden.
        actual = run_shape_inference(
            MSFT,
            "GroupQueryAttention",
            [
                ts(FLOAT, [2, 8, 128]),  # query
                ts(FLOAT, [2, 8, 32]),  # key   (kv_num_heads=1, head_size=32)
                ts(FLOAT, [2, 8, 32]),  # value
            ],
            attributes={
                "num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4),
                "kv_num_heads": ir.Attr("kv_num_heads", ir.AttributeType.INT, 1),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 128]))


# ---------------------------------------------------------------------------
# QLinearConv
# ---------------------------------------------------------------------------


class QLinearConvTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConv",
            [
                ts(INT8, [1, 3, 32, 32]),  # x
                ts(FLOAT, []),  # x_scale
                ts(INT8, []),  # x_zp
                ts(INT8, [16, 3, 3, 3]),  # w
                ts(FLOAT, []),  # w_scale
                ts(INT8, []),  # w_zp
                ts(FLOAT, []),  # y_scale
                ts(INT8, []),  # y_zp
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([1, 16, 30, 30]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_dtype_from_y_zero_point(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConv",
            [
                ts(ir.DataType.UINT8, [1, 3, 32, 32]),
                ts(FLOAT, []),
                ts(ir.DataType.UINT8, []),
                ts(INT8, [16, 3, 3, 3]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([1, 16, 30, 30]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_channels_last(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConv",
            [
                ts(ir.DataType.UINT8, [1, 5, 5, 3]),
                ts(FLOAT, []),
                ts(ir.DataType.UINT8, []),
                ts(INT8, [8, 3, 3, 3]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"channels_last": ir.Attr("channels_last", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(INT8, [1, 3, 3, 8])])

    def test_channels_last_symbolic_spatial_dims(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConv",
            [
                ts(ir.DataType.UINT8, ["batch", "height", "width", 3]),
                ts(FLOAT, []),
                ts(ir.DataType.UINT8, []),
                ts(INT8, [8, 3, 3, 3]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"channels_last": ir.Attr("channels_last", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual, [ts(INT8, ["batch", "height - 2", "width - 2", 8])])

    def test_missing_required_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MSFT,
                "QLinearConv",
                [
                    ts(ir.DataType.UINT8, [1, 5, 5, 3]),
                    ts(FLOAT, []),
                    ts(ir.DataType.UINT8, []),
                    ts(INT8, [8, 3, 3, 3]),
                    None,
                    ts(INT8, []),
                    ts(FLOAT, []),
                    ts(INT8, []),
                ],
                attributes={
                    "channels_last": ir.Attr("channels_last", ir.AttributeType.INT, 1)
                },
                opset_version=1,
            )


# ---------------------------------------------------------------------------
# QLinearConcat
# ---------------------------------------------------------------------------


class QLinearConcatFullTest(unittest.TestCase):
    def test_negative_axis(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),  # Y_scale
                ts(INT8, []),  # Y_zp
                ts(INT8, [2, 4]),  # tensor1
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 6]),  # tensor2
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, -1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 10]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_no_data(self):
        """No data tensors — no output."""
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [ts(FLOAT, []), ts(INT8, [])],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)

    def test_null_first_shape(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, None),  # no shape
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            opset_version=1,
        )
        self.assertIsNone(actual[0].shape)

    def test_dtype_from_zp(self):
        """Dtype falls back to zp when data has no dtype."""
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),  # Y_scale
                ts(None, []),  # Y_zp
                ts(None, [2, 4]),  # tensor1 (no dtype)
                ts(FLOAT, []),
                ts(INT8, []),  # zp1 (INT8)
                ts(None, [2, 6]),  # tensor2 (no dtype)
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 10]))
        self.assertEqual(actual[0].type.dtype, INT8)


# ---------------------------------------------------------------------------
# QLinearWhere
# ---------------------------------------------------------------------------


class QLinearWhereTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearWhere",
            [
                ts(ir.DataType.BOOL, [2, 8]),  # condition
                ts(INT8, [2, 8]),  # X
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 8]),  # Y
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_broadcast(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearWhere",
            [
                ts(ir.DataType.BOOL, [2, 1]),
                ts(INT8, [1, 8]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 8]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))

    def test_dtype_from_zp(self):
        """Dtype falls back to x_zp when X has no dtype."""
        actual = run_shape_inference(
            MSFT,
            "QLinearWhere",
            [
                ts(ir.DataType.BOOL, [2, 8]),
                ts(None, [2, 8]),  # X (no dtype)
                ts(FLOAT, []),
                ts(INT8, []),  # x_zp (INT8)
                ts(None, [2, 8]),  # Y
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].type.dtype, INT8)


# ---------------------------------------------------------------------------
# QOrdered extended tests
# ---------------------------------------------------------------------------


class QOrderedExtendedTest(unittest.TestCase):
    def test_layer_norm(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedLayerNormalization",
            [ts(INT8, [2, 8, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))

    def test_attention(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedAttention",
            [ts(INT8, [2, 8, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))

    def test_longformer_attention(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedLongformerAttention",
            [ts(INT8, [2, 8, 64])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))


# ---------------------------------------------------------------------------
# QEmbedLayerNormalization
# ---------------------------------------------------------------------------


class QEmbedLayerNormTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QEmbedLayerNormalization",
            [
                ts(INT32, [2, 8]),  # input_ids
                None,  # segment_ids
                ts(FLOAT, [100, 64]),  # word_embedding
            ],
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))
        self.assertEqual(actual[0].type.dtype, FLOAT)
        self.assertEqual(actual[1].shape, ir.Shape([2]))
        self.assertEqual(actual[1].type.dtype, INT32)


# ---------------------------------------------------------------------------
# Dtype fallback tests for QLinear ops
# ---------------------------------------------------------------------------


class QLinearDtypeFallbackTest(unittest.TestCase):
    """Test that Q* ops fall back to zp dtype when input dtype is None."""

    def test_unary_dtype_from_zp(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearSigmoid",
            [ts(None, [2, 8]), ts(FLOAT, []), ts(INT8, []), ts(FLOAT, []), ts(INT8, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_binary_dtype_from_zp(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearAdd",
            [
                ts(None, [2, 8]),  # A (no dtype)
                ts(FLOAT, []),
                ts(INT8, []),  # A_zp (INT8)
                ts(None, [2, 8]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))
        self.assertEqual(actual[0].type.dtype, INT8)

    def test_qordered_matmul_dtype_from_input(self):
        actual = run_shape_inference(
            MSFT,
            "QOrderedMatMul",
            [ts(INT8, [4, 32]), ts(FLOAT, []), ts(INT8, [32, 16])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))
        self.assertEqual(actual[0].type.dtype, INT8)


# ---------------------------------------------------------------------------
# Additional coverage: QLinearSoftmax, QLinearAveragePool, etc.
# ---------------------------------------------------------------------------


class QLinearSoftmaxTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearSoftmax",
            [ts(INT8, [2, 8]), ts(FLOAT, []), ts(INT8, []), ts(FLOAT, []), ts(INT8, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))


class QLinearPoolTest(unittest.TestCase):
    def test_average_pool(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearAveragePool",
            [ts(INT8, [2, 8]), ts(FLOAT, []), ts(INT8, []), ts(FLOAT, []), ts(INT8, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))

    def test_global_average_pool(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearGlobalAveragePool",
            [ts(INT8, [2, 8]), ts(FLOAT, []), ts(INT8, []), ts(FLOAT, []), ts(INT8, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))

    def test_reduce_mean(self):
        actual = run_shape_inference(
            MSFT,
            "QLinearReduceMean",
            [ts(INT8, [2, 8]), ts(FLOAT, []), ts(INT8, []), ts(FLOAT, []), ts(INT8, [])],
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8]))


# ---------------------------------------------------------------------------
# Additional edge case coverage
# ---------------------------------------------------------------------------


class QAttentionPresentTest(unittest.TestCase):
    def test_non_3d_fallback(self):
        """Non-3D input falls back to passthrough."""
        actual = run_shape_inference(
            MSFT,
            "QAttention",
            [
                ts(None, [8, 64]),
                ts(None, [64, 192]),
                ts(FLOAT, [192]),
                ts(FLOAT, []),
                ts(FLOAT, []),
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([8, 64]))

    def test_with_past(self):
        actual = run_shape_inference(
            MSFT,
            "QAttention",
            [
                ts(None, [2, 8, 64]),
                ts(None, [64, 192]),
                ts(FLOAT, [192]),
                ts(FLOAT, []),
                ts(FLOAT, []),
                None,
                None,
                None,
                ts(FLOAT, [2, 2, 4, 10, 16]),  # past
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual[0].shape, ir.Shape([2, 8, 64]))
        # present: [2, 2, 4, 18, 16]  (past_seq=10 + cur_seq=8)
        self.assertEqual(actual[1].shape, ir.Shape([2, 2, 4, 18, 16]))


class QGemmTransATest(unittest.TestCase):
    def test_trans_a(self):
        actual = run_shape_inference(
            MSFT,
            "QGemm",
            [
                ts(None, [32, 4]),  # A (will be transposed)
                ts(FLOAT, []),
                ts(None, []),
                ts(None, [32, 16]),
                ts(FLOAT, []),
                ts(None, []),
            ],
            attributes={"transA": ir.Attr("transA", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape, ir.Shape([4, 16]))


class MHAPresentWithPastTest(unittest.TestCase):
    def test_3d_with_past(self):
        """MHA with past key input extends total_seq."""
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [
                ts(FLOAT, [2, 8, 64]),  # query
                ts(FLOAT, [2, 10, 64]),  # key
                ts(FLOAT, [2, 10, 64]),  # value
                None,
                None,
                None,
                ts(FLOAT, [2, 4, 5, 16]),  # past_key
            ],
            attributes={"num_heads": ir.Attr("num_heads", ir.AttributeType.INT, 4)},
            opset_version=1,
            num_outputs=3,
        )
        # present: total_seq = 5 + 10 = 15
        self.assertEqual(actual[1].shape, ir.Shape([2, 4, 15, 16]))

    def test_no_num_heads(self):
        """MHA without num_heads — no present state set."""
        actual = run_shape_inference(
            MSFT,
            "MultiHeadAttention",
            [ts(FLOAT, [2, 8, 64]), ts(FLOAT, [2, 10, 64]), ts(FLOAT, [2, 10, 64])],
            opset_version=1,
            num_outputs=3,
        )
        self.assertIsNone(actual[1].shape)


class QLinearConcatSymbolicTest(unittest.TestCase):
    def test_symbolic_dim(self):
        """Symbolic dim in concat axis produces symbolic output."""
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, None]),  # symbolic dim
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 6]),
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape.rank(), 2)

    def test_rank_mismatch(self):
        """Different ranks in concat inputs produces symbolic axis dim."""
        actual = run_shape_inference(
            MSFT,
            "QLinearConcat",
            [
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 4]),
                ts(FLOAT, []),
                ts(INT8, []),
                ts(INT8, [2, 4, 3]),  # different rank
                ts(FLOAT, []),
                ts(INT8, []),
            ],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
            opset_version=1,
        )
        self.assertEqual(actual[0].shape.rank(), 2)


if __name__ == "__main__":
    unittest.main()
