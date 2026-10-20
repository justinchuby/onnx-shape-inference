# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regressions for released ONNX opset 28 and newer contrib semantics."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError, registry
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
HALF = ir.DataType.FLOAT16
MS = "com.microsoft"


class SwiGLUTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("concrete", [2, 7, 64], [2, 7, 64], [2, 7, 64]),
            ("symbolic", ["B", "S", 64], ["B", "S", 64], ["B", "S", 64]),
            ("refine", ["B", 64], [2, 64], [2, 64]),
            ("unknown_left", None, [2, 64], [2, 64]),
            ("unknown_right", [2, 64], None, [2, 64]),
            ("unknown", None, None, None),
            ("scalar", [], [], []),
        ]
    )
    def test_shapes(self, _name, left, right, expected):
        self.assertEqual(
            run_shape_inference(
                "", "SwiGLU", [ts(HALF, left), ts(HALF, right)], opset_version=28
            ),
            [ts(HALF, expected)],
        )

    @parameterized.parameterized.expand(
        [("broadcast_forbidden", [2, 64], [1, 64]), ("rank", [2, 64], [64])]
    )
    def test_no_broadcast(self, _name, left, right):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "", "SwiGLU", [ts(HALF, left), ts(HALF, right)], opset_version=28
            )

    def test_dtype_mismatch(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference("", "SwiGLU", [ts(HALF), ts(FLOAT)], opset_version=28)

    def test_required_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "SwiGLU", [ts(HALF)], opset_version=28)

    def test_version_boundary(self):
        self.assertIsNone(registry.get("", "SwiGLU", 27))
        self.assertIsNotNone(registry.get("", "SwiGLU", 28))

    def test_matches_released_onnx_inference(self):
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("SwiGLU", ["a", "b"], ["y"])],
            "swiglu",
            [
                onnx.helper.make_tensor_value_info(
                    name, onnx.TensorProto.FLOAT16, ["B", 7, 64]
                )
                for name in ("a", "b")
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, None)],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 28)])
        expected = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model))
        actual = run_shape_inference(
            "", "SwiGLU", [ts(HALF, ["B", 7, 64])] * 2, opset_version=28
        )
        out = expected.graph.outputs[0]
        self.assertEqual(actual, [ir.TypeAndShape(out.type, out.shape)])


class ExistingOpset28SemanticsTest(unittest.TestCase):
    def test_signed_bitshift(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "BitShift",
                [ts(ir.DataType.INT32, ["B", 1]), ts(ir.DataType.INT32, [7])],
                {"direction": ir.AttrString("direction", "RIGHT")},
                opset_version=28,
            ),
            [ts(ir.DataType.INT32, ["B", 7])],
        )

    @parameterized.parameterized.expand([("DCR",), ("CRD",)])
    def test_space_to_depth_new_mode_preserves_shape_rule(self, mode):
        self.assertEqual(
            run_shape_inference(
                "",
                "SpaceToDepth",
                [ts(HALF, ["B", 3, 8, 12])],
                {
                    "blocksize": ir.AttrInt64("blocksize", 2),
                    "mode": ir.AttrString("mode", mode),
                },
                opset_version=28,
            ),
            [ts(HALF, ["B", 12, 4, 6])],
        )

    def test_floating_mod(self):
        self.assertEqual(
            run_shape_inference(
                "", "Mod", [ts(FLOAT, ["B", 1]), ts(FLOAT, [7])], opset_version=28
            ),
            [ts(FLOAT, ["B", 7])],
        )


class UpdatedContribSemanticsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("dilated", [2, 64, 7], 0, 3, 0, [2, 64, 9]),
            ("channels_last", [2, 7, 64], 1, 2, 0, [2, 6, 64]),
            ("split_channels", ["B", "S", 4, "H"], 1, 2, 3, [3, "B", 6, 4, "H"]),
            ("legacy", [2, 64, 7], 0, 1, 0, [2, 64, 3]),
        ]
    )
    def test_causal_conv_layout_and_dilation(
        self, _name, shape, channels_last, dilation, window, state
    ):
        actual = run_shape_inference(
            MS,
            "CausalConvWithState",
            [ts(HALF, shape), ts(HALF, [64, 1, 4])],
            {
                "channels_last": ir.AttrInt64("channels_last", channels_last),
                "dilation": ir.AttrInt64("dilation", dilation),
                "state_window": ir.AttrInt64("state_window", window),
            },
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(actual, [ts(HALF, shape), ts(HALF, state)])

    @parameterized.parameterized.expand(
        [
            ("bad_dilation", "dilation", 0),
            ("bad_layout", "channels_last", 2),
            ("bad_ndim", "ndim", 4),
        ]
    )
    def test_invalid_causal_conv_attributes(self, _name, attr, value):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "CausalConvWithState",
                [ts(HALF, [2, 64, 7]), ts(HALF, [64, 1, 4])],
                {attr: ir.AttrInt64(attr, value)},
                opset_version=1,
                num_outputs=2,
            )

    def test_sliding_window_gqa_keeps_cache_capacity(self):
        actual = run_shape_inference(
            MS,
            "GroupQueryAttention",
            [
                ts(HALF, [1, 7, 128]),
                ts(HALF, [1, 7, 64]),
                ts(HALF, [1, 7, 64]),
                ts(HALF, [1, 2, 16, 32]),
                ts(HALF, [1, 2, 16, 32]),
                ts(ir.DataType.INT32, [1]),
                ts(ir.DataType.INT32, []),
            ],
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
                "sliding_window_cache": ir.AttrInt64("sliding_window_cache", 1),
                "local_window_size": ir.AttrInt64("local_window_size", 16),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(HALF, [1, 7, 128]), ts(HALF, [1, 2, 16, 32]), ts(HALF, [1, 2, 16, 32])]
        )

    def test_sliding_window_qk_uses_absolute_total_not_cache_capacity(self):
        specs = [
            ts(HALF, [1, 7, 128]),
            ts(HALF, [1, 7, 64]),
            ts(HALF, [1, 7, 64]),
            ts(HALF, [1, 2, 16, 32]),
            ts(HALF, [1, 2, 16, 32]),
            ts(ir.DataType.INT32, [1]),
            ts(ir.DataType.INT32, [1]),
        ]
        inputs = [
            ir.Value(name=f"in_{i}", shape=s.shape, type=s.type) for i, s in enumerate(specs)
        ]
        actual = run_shape_inference_with_values(
            MS,
            "GroupQueryAttention",
            inputs,
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
                "sliding_window_cache": ir.AttrInt64("sliding_window_cache", 1),
                "local_window_size": ir.AttrInt64("local_window_size", 16),
                "qk_output": ir.AttrInt64("qk_output", 1),
            },
            opset_version=1,
            num_outputs=4,
            symbolic_values={6: [100]},
        )
        self.assertEqual(actual[1:3], [ts(HALF, [1, 2, 16, 32])] * 2)
        self.assertEqual(actual[3], ts(HALF, [1, 4, 7, 100]))

    @parameterized.parameterized.expand([("int4", 64), ("int4", 65), ("float4e2m1", 64)])
    def test_paged_latent_subbyte_cache_geometry(self, logical_type, head_size):
        cache_shape = [16, 8, 1, (head_size + 1) // 2]
        actual = run_shape_inference(
            MS,
            "PagedAttention",
            [
                ts(HALF, [7, 4 * head_size]),
                ts(HALF, [7, head_size]),
                None,
                ts(ir.DataType.UINT8, cache_shape),
                None,
                ts(ir.DataType.INT32),
                ts(ir.DataType.INT32),
                ts(ir.DataType.INT32),
                None,
                None,
                None,
                None,
                None,
                None,
                ts(FLOAT, []),
            ],
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 1),
                "kv_cache_layout": ir.AttrString("kv_cache_layout", "LATENT"),
                "k_cache_dtype": ir.AttrString("k_cache_dtype", logical_type),
                "k_quant_type": ir.AttrString("k_quant_type", "PER_TENSOR"),
            },
            opset_version=1,
            num_outputs=2,
        )
        self.assertEqual(
            actual, [ts(HALF, [7, 4 * head_size]), ts(ir.DataType.UINT8, cache_shape)]
        )

    @parameterized.parameterized.expand([("MoE",), ("QMoE",)])
    def test_moe_packed_tokens(self, op):
        self.assertEqual(
            run_shape_inference(MS, op, [ts(HALF, ["T", 128])], opset_version=1),
            [ts(HALF, ["T", 128])],
        )
