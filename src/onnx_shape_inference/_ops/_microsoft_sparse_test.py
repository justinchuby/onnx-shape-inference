# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for sparse attention schemas released in ORT 1.31."""

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
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

HALF = ir.DataType.FLOAT16
INT8 = ir.DataType.INT8
INT32 = ir.DataType.INT32
MS = "com.microsoft"


def _policy(qsa=True):
    return {
        "policy_mode": ir.AttrString("policy_mode", "qsa" if qsa else "csa"),
        "compress_ratio": ir.AttrInt64("compress_ratio", 4),
        "token_budget" if qsa else "index_topk": ir.AttrInt64(
            "token_budget" if qsa else "index_topk", 8 if qsa else 5
        ),
    }


def _dense_inputs(qsa=True, shape=None, past=None, buffer=None):
    inputs = [
        ts(HALF, shape if shape is not None else [2, 7, 128]),
        ts(HALF, [2, 7, 32 if qsa else 64]),
        ts(HALF, [32]),
        ts(HALF, [32]),
        ts(HALF, [128, 16]),
        ts(HALF, [128, 16]),
        None,
        ts(HALF, past if past is not None else [2, 8, 32]),
    ]
    if not qsa:
        inputs += [
            ts(HALF, [2, 7, 64]),
            ts(HALF, [4, 64]),
            ts(HALF, [2, 7, 4]),
            ts(ir.DataType.INT64, [2, 7]),
            None,
            ts(HALF, buffer if buffer is not None else [2, 2, 6, 64]),
        ]
    return inputs


class SparseAttentionIndexerTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("growing", False, [2, 15, 32]), ("fixed", True, [2, 8, 32])]
    )
    def test_qsa(self, _name, fixed, expected):
        inputs = _dense_inputs()
        if fixed:
            inputs += [None] * 4 + [ts(INT32, [1])]
        self.assertEqual(
            run_shape_inference(
                MS, "SparseAttentionIndexer", inputs, _policy(), opset_version=1, num_outputs=2
            ),
            [ts(INT32, [2, 7, 11]), ts(HALF, expected)],
        )

    def test_packed_qk(self):
        inputs = _dense_inputs(shape=[2, 7, 160])
        inputs[1] = None
        actual = run_shape_inference(
            MS, "SparseAttentionIndexer", inputs, _policy(), opset_version=1, num_outputs=2
        )
        self.assertEqual(actual, [ts(INT32, [2, 7, 11]), ts(HALF, [2, 15, 32])])

    @parameterized.parameterized.expand(
        [
            ("symbolic", ["B", "S", 128], ["B", "P", 32], ["B", "P + S", 32]),
            ("zero_new_tokens", [2, 0, 128], [2, 8, 32], [2, 8, 32]),
        ]
    )
    def test_qsa_symbolic_and_empty(self, _name, query, past, expected):
        self.assertEqual(
            run_shape_inference(
                MS,
                "SparseAttentionIndexer",
                _dense_inputs(shape=query, past=past),
                _policy(),
                opset_version=1,
                num_outputs=2,
            ),
            [ts(INT32, [*query[:-1], 11]), ts(HALF, expected)],
        )

    @parameterized.parameterized.expand(
        [
            ("first_partial", 0, 3, 8, 3),
            ("first_complete", 0, 4, 9, 4),
            ("overlap", 6, 7, 10, 5),
            ("retain_overlap", 4, 1, 8, 5),
            ("empty", 6, 0, 8, 6),
        ]
    )
    def test_csa_window_plan(self, _name, previous, seq, entries, remaining):
        inputs = _dense_inputs(False, shape=[2, seq, 128], buffer=[2, 2, previous, 64])
        self.assertEqual(
            run_shape_inference(
                MS,
                "SparseAttentionIndexer",
                inputs,
                _policy(False),
                opset_version=1,
                num_outputs=3,
            ),
            [
                ts(INT32, [2, seq, 5]),
                ts(HALF, [2, entries, 32]),
                ts(HALF, [2, 2, remaining, 64]),
            ],
        )

    def test_csa_fixed_cache(self):
        inputs = _dense_inputs(False)
        inputs[12] = ts(INT32, [1])
        actual = run_shape_inference(
            MS,
            "SparseAttentionIndexer",
            inputs,
            _policy(False),
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(INT32, [2, 7, 5]), ts(HALF, [2, 8, 32]), ts(HALF, [2, 2, 5, 64])]
        )

    def test_unknown_shapes_keep_dtypes(self):
        inputs = [ts(HALF)] * 6 + [None, ts(HALF)]
        actual = run_shape_inference(
            MS, "SparseAttentionIndexer", inputs, _policy(), opset_version=1, num_outputs=2
        )
        self.assertEqual(actual, [ts(INT32), ts(HALF)])

    @parameterized.parameterized.expand(
        [
            ("bad_policy", "policy_mode", "other"),
            ("zero_ratio", "compress_ratio", 0),
            ("bad_budget", "token_budget", 7),
        ]
    )
    def test_invalid_policy(self, _name, name, value):
        attrs = _policy()
        attrs[name] = (
            ir.AttrString(name, value) if isinstance(value, str) else ir.AttrInt64(name, value)
        )
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "SparseAttentionIndexer",
                _dense_inputs(),
                attrs,
                opset_version=1,
                num_outputs=2,
            )

    def test_invalid_csa_buffer(self):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                MS,
                "SparseAttentionIndexer",
                _dense_inputs(False, buffer=[2, 2, 8, 64]),
                _policy(False),
                opset_version=1,
                num_outputs=3,
            )

    def test_required_past(self):
        inputs = _dense_inputs()
        inputs[7] = None
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS, "SparseAttentionIndexer", inputs, _policy(), opset_version=1, num_outputs=2
            )


class PackedSparseAttentionIndexerTest(unittest.TestCase):
    def _run(self, qsa, *, capture=0, unknown=False):
        specs = [
            ts(HALF, [13, 128]),
            ts(HALF, [13, 32 if qsa else 64]),
            ts(HALF, [32]),
            ts(HALF, [32]),
            ts(HALF, [128, 16]),
            ts(HALF, [128, 16]),
            ts(INT32, [3]),
            ts(INT32, [2]),
            None if qsa else ts(HALF, [13, 64]),
            None if qsa else ts(HALF, [4, 64]),
            None if qsa else ts(HALF, [13, 4]),
            None if qsa else ts(ir.DataType.INT64, [13]),
            ts(HALF, [2, 64, 32]),
            ts(HALF, [2, 7, 32 if qsa else 64]),
            None if qsa else ts(HALF, [2, 7, 64]),
            ts(INT32, [2, 2]),
        ]
        if capture:
            specs.append(ts(INT32, [2]))
        inputs = [
            ir.Value(name=f"in_{i}", type=spec.type, shape=None if unknown else spec.shape)
            if spec is not None
            else None
            for i, spec in enumerate(specs)
        ]
        outputs = [
            ir.Value(name="" if qsa and i == 4 else f"out_{i}")
            for i in range(7 if capture else 6)
        ]
        attrs = _policy(qsa)
        attrs["state_capacity"] = ir.AttrInt64("state_capacity", 64)
        attrs["state_update_capacity"] = ir.AttrInt64("state_update_capacity", capture)
        node = ir.Node(
            MS, "PackedSparseAttentionIndexer", inputs, attributes=attrs, outputs=outputs
        )
        ctx = ShapeInferenceContext({MS: 1}, policy="override")
        registry.get(MS, node.op_type, 1)(ctx, node)
        return [ir.TypeAndShape(v.type, v.shape) for v in outputs]

    @parameterized.parameterized.expand([(True,), (False,)])
    def test_fixed_state(self, qsa):
        self.assertEqual(
            self._run(qsa),
            [
                ts(INT32, [13, 11 if qsa else 5]),
                ts(INT32, [13]),
                ts(HALF, [2, 64, 32]),
                ts(HALF, [2, 7, 32 if qsa else 64]),
                ts() if qsa else ts(HALF, [2, 7, 64]),
                ts(INT32, [2, 2]),
            ],
        )

    def test_transition_capture(self):
        self.assertEqual(self._run(True, capture=3)[-1], ts(HALF, [2, 3, 32]))

    @parameterized.parameterized.expand([(True,), (False,)])
    def test_unknown_shapes(self, qsa):
        self.assertEqual(
            self._run(qsa, unknown=True),
            [
                ts(INT32),
                ts(INT32),
                ts(HALF),
                ts(HALF),
                ts() if qsa else ts(HALF),
                ts(INT32),
            ],
        )

    def test_csa_capture_forbidden(self):
        with self.assertRaises(OpUsageError):
            self._run(False, capture=1)


class DynamicSparseAttentionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("separate", False, [2, 7, 128]),
            ("packed", True, [2, 7, 256]),
            ("symbolic", True, ["B", "S", "W"]),
        ]
    )
    def test_fixed_caches(self, _name, packed, query):
        inputs = [
            ts(HALF, query),
            None if packed else ts(HALF, [2, 7, 64]),
            None if packed else ts(HALF, [2, 7, 64]),
            ts(HALF, ["B", 2, "C", 32]),
            ts(HALF, ["B", 2, "C", 32]),
            None,
            None,
            ts(INT32),
            ts(INT32),
            ts(INT32),
            ts(INT32),
        ]
        actual = run_shape_inference(
            MS,
            "DynamicSparseAttention",
            inputs,
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
            },
            opset_version=1,
            num_outputs=3,
        )
        expected = ["B", "S", "4*floor(W/8)"] if _name == "symbolic" else [2, 7, 128]
        self.assertEqual(
            actual,
            [
                ts(HALF, expected),
                ts(HALF, ["B", 2, "C", 32]),
                ts(HALF, ["B", 2, "C", 32]),
            ],
        )

    def test_prefill_constant_total_length(self):
        inputs = [ir.Value(name="q", type=ir.TensorType(HALF), shape=ir.Shape([2, 7, 256]))]
        inputs += [None] * 6
        inputs += [ir.Value(name=f"i{i}", type=ir.TensorType(INT32)) for i in range(3)]
        inputs += [const_value([7])]
        actual = run_shape_inference_with_values(
            MS,
            "DynamicSparseAttention",
            inputs,
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(HALF, [2, 7, 128]), ts(HALF, [2, 2, 7, 32]), ts(HALF, [2, 2, 7, 32])]
        )


class SparsePagedAttentionTest(unittest.TestCase):
    @parameterized.parameterized.expand([(False,), (True,)])
    def test_packed_and_separate(self, packed):
        actual = run_shape_inference(
            MS,
            "SparsePagedAttention",
            [
                ts(HALF, ["T", 256 if packed else 128]),
                None if packed else ts(HALF, ["T", 64]),
                None if packed else ts(HALF, ["T", 64]),
                ts(INT8, [16, 8, 2, 32]),
                ts(INT8, [16, 8, 2, 32]),
                ts(INT32),
                ts(INT32),
                ts(INT32),
                None,
                ts(INT32, ["T", 11]),
                ts(INT32, ["T"]),
            ],
            {
                "num_heads": ir.AttrInt64("num_heads", 4),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
            },
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(
            actual, [ts(HALF, ["T", 128]), ts(INT8, [16, 8, 2, 32]), ts(INT8, [16, 8, 2, 32])]
        )
