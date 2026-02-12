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


class DecoderMaskedMHATest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "DecoderMaskedMultiHeadAttention",
            [ts(FLOAT, [2, 1, 64])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 1, 64])])


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


class RestorePaddingTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            MSFT,
            "RestorePadding",
            [ts(FLOAT, [16, 64]), ts(INT32, [2, 8])],
            opset_version=1,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 64])])


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
        actual = run_shape_inference_with_values(
            MSFT,
            "GroupQueryAttention",
            [
                ir.Value(name="q", shape=ir.Shape([2, 1, 64]), type=ir.TensorType(FLOAT)),
                None,  # key
                None,  # value
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
        # present_key: [2, 1, 11, 16]
        self.assertEqual(actual[1].shape, ir.Shape([2, 1, 11, 16]))


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


if __name__ == "__main__":
    unittest.main()
