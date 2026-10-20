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
    def _run(
        self,
        qsa,
        *,
        capture=0,
        unknown=False,
        overrides=None,
        attributes=None,
        output_names=None,
        policy="override",
    ):
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
        if overrides:
            for index, spec in overrides.items():
                specs[index] = spec
        inputs = [
            ir.Value(name=f"in_{i}", type=spec.type, shape=None if unknown else spec.shape)
            if spec is not None
            else None
            for i, spec in enumerate(specs)
        ]
        names = (
            output_names
            if output_names is not None
            else ["" if qsa and i == 4 else f"out_{i}" for i in range(7 if capture else 6)]
        )
        outputs = [ir.Value(name=name) for name in names]
        attrs = _policy(qsa)
        attrs["state_capacity"] = ir.AttrInt64("state_capacity", 64)
        attrs["state_update_capacity"] = ir.AttrInt64("state_update_capacity", capture)
        if attributes:
            attrs.update(attributes)
        node = ir.Node(
            MS, "PackedSparseAttentionIndexer", inputs, attributes=attrs, outputs=outputs
        )
        ctx = ShapeInferenceContext({MS: 1}, policy=policy)
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

    @parameterized.parameterized.expand(
        [
            ("key_rank", 12, ts(HALF, [2, 64])),
            ("buffer_rank", 13, ts(HALF, [2, 7])),
            ("length_rank", 15, ts(INT32, [2])),
            ("key_capacity", 12, ts(HALF, [2, 63, 32])),
            ("buffer_capacity", 13, ts(HALF, [2, 6, 32])),
            ("length_columns", 15, ts(INT32, [2, 3])),
        ]
    )
    def test_invalid_state(self, _name, index, spec):
        with self.assertRaises(ShapeInferenceError):
            self._run(True, overrides={index: spec})
        with self.assertLogs("onnx_shape_inference._context", level="WARNING"):
            self._run(True, overrides={index: spec}, policy="skip")

    @parameterized.parameterized.expand([("state_capacity", 0), ("state_update_capacity", 9)])
    def test_invalid_capacity(self, attr, value):
        attributes = {attr: ir.AttrInt64(attr, value)}
        with self.assertRaises(ShapeInferenceError):
            self._run(True, attributes=attributes)
        with self.assertLogs("onnx_shape_inference._context", level="WARNING"):
            self._run(True, attributes=attributes, policy="skip")

    @parameterized.parameterized.expand(
        [
            ("output_count", {"output_names": ["a", "b"]}),
            ("qsa_gate_output", {"output_names": ["a", "b", "c", "d", "e", "f"]}),
            ("capture_input", {"capture": 1, "overrides": {16: None}}),
        ]
    )
    def test_invalid_optional_contract(self, _name, kwargs):
        with self.assertRaises(OpUsageError):
            self._run(True, **kwargs)


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


class SparseBoundaryTest(unittest.TestCase):
    def _invalid(self, op, inputs, attrs, outputs=2):
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(MS, op, inputs, attrs, opset_version=1, num_outputs=outputs)
        with self.assertLogs("onnx_shape_inference._context", level="WARNING"):
            run_shape_inference(
                MS, op, inputs, attrs, opset_version=1, num_outputs=outputs, policy="skip"
            )

    @parameterized.parameterized.expand(
        [
            ("policy", "policy_mode", "invalid"),
            ("ratio", "compress_ratio", 0),
            ("ratio_overflow", "compress_ratio", 2**31),
            ("budget", "token_budget", 7),
            ("capacity_overflow", "token_budget", 2**31),
            ("forbidden", "index_topk", 5),
        ]
    )
    def test_indexer_invalid_attributes(self, _name, name, value):
        attrs = _policy()
        attrs[name] = (
            ir.AttrString(name, value) if isinstance(value, str) else ir.AttrInt64(name, value)
        )
        self._invalid("SparseAttentionIndexer", _dense_inputs(), attrs)

    @parameterized.parameterized.expand(
        [
            ("query_rank", 0, [7, 128]),
            ("norm_rank", 2, []),
            ("cos_rank", 4, [128]),
            ("sin_rank", 5, [128]),
            ("zero_head", 2, [0]),
            ("query_width", 0, [2, 7, 127]),
            ("past_rank", 7, [8, 32]),
        ]
    )
    def test_indexer_invalid_ranks_and_geometry(self, _name, index, shape):
        inputs = _dense_inputs()
        inputs[index] = ts(HALF, shape)
        self._invalid("SparseAttentionIndexer", inputs, _policy())

    @parameterized.parameterized.expand(
        [
            ("rank", [2, 6, 64]),
            ("planes", [3, 2, 6, 64]),
            ("batch", [2, 3, 6, 64]),
            ("width", [2, 2, 6, 63]),
            ("length", [2, 2, 8, 64]),
        ]
    )
    def test_csa_invalid_buffer(self, _name, shape):
        self._invalid(
            "SparseAttentionIndexer", _dense_inputs(False, buffer=shape), _policy(False), 3
        )

    @parameterized.parameterized.expand(
        [("qsa_with_gate", True, 8), ("csa_with_mask", False, 6)]
    )
    def test_policy_input_contract(self, _name, qsa, index):
        inputs = _dense_inputs(qsa)
        inputs += [None] * max(0, index + 1 - len(inputs))
        inputs[index] = ts(HALF)
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS,
                "SparseAttentionIndexer",
                inputs,
                _policy(qsa),
                opset_version=1,
                num_outputs=2 if qsa else 3,
            )

    def test_dense_output_count(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS, "SparseAttentionIndexer", _dense_inputs(), _policy(), opset_version=1
            )

    def test_qsa_unknown_query_keeps_past_batch(self):
        inputs = _dense_inputs()
        inputs[0] = ts(HALF)
        self.assertEqual(
            run_shape_inference(
                MS, "SparseAttentionIndexer", inputs, _policy(), opset_version=1, num_outputs=2
            ),
            [ts(INT32), ts(HALF, [2, "_d0", 32])],
        )

    @parameterized.parameterized.expand([(True,), (False,)])
    def test_csa_unresolved_window_plan(self, unknown_buffer):
        inputs = _dense_inputs(False, shape=[2, "S", 128])
        if unknown_buffer:
            inputs[13] = ts(HALF)
        actual = run_shape_inference(
            MS,
            "SparseAttentionIndexer",
            inputs,
            _policy(False),
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual[0], ts(INT32, [2, "S", 5]))
        self.assertEqual(actual[1], ts(HALF, [2, "_d1", 32]))
        self.assertEqual(
            actual[2], ts(HALF) if unknown_buffer else ts(HALF, [2, 2, "_d0", 64])
        )

    def test_csa_unknown_past_key(self):
        inputs = _dense_inputs(False)
        inputs[7] = ts(HALF)
        actual = run_shape_inference(
            MS,
            "SparseAttentionIndexer",
            inputs,
            _policy(False),
            opset_version=1,
            num_outputs=3,
        )
        self.assertEqual(actual, [ts(INT32, [2, 7, 5]), ts(HALF), ts(HALF, [2, 2, 5, 64])])

    @parameterized.parameterized.expand([(4,), (9,), (10,)])
    def test_sparse_paged_missing_selection_input(self, index):
        inputs = [
            ts(HALF),
            ts(HALF),
            ts(HALF),
            ts(HALF),
            ts(HALF),
            ts(INT32),
            ts(INT32),
            ts(INT32),
            None,
            ts(INT32),
            ts(INT32),
        ]
        inputs[index] = None
        with self.assertRaises(OpUsageError):
            run_shape_inference(MS, "SparsePagedAttention", inputs, opset_version=1)

    @staticmethod
    def _dynamic_inputs():
        return [
            ts(HALF, [2, 7, 256]),
            None,
            None,
            ts(HALF, [2, 2, 16, 32]),
            ts(HALF, [2, 2, 16, 32]),
            None,
            None,
            ts(INT32),
            ts(INT32),
            ts(INT32),
            ts(INT32),
        ]

    @staticmethod
    def _heads():
        return {
            "num_heads": ir.AttrInt64("num_heads", 4),
            "kv_num_heads": ir.AttrInt64("kv_num_heads", 2),
        }

    @parameterized.parameterized.expand(
        [
            ("heads", None, None, "num_heads", 0),
            ("kv_heads", None, None, "kv_num_heads", 3),
            ("query_rank", 0, [7, 256], None, None),
            ("query_width", 0, [2, 7, 255], None, None),
            ("cache_rank", 3, [2, 16, 32], None, None),
        ]
    )
    def test_dynamic_invalid(self, _name, index, shape, attr, value):
        inputs, attrs = self._dynamic_inputs(), self._heads()
        if index is not None:
            inputs[index] = ts(HALF, shape)
        if attr is not None:
            attrs[attr] = ir.AttrInt64(attr, value)
        self._invalid("DynamicSparseAttention", inputs, attrs, 3)

    def test_dynamic_mismatched_optional_inputs(self):
        inputs = self._dynamic_inputs()
        inputs[1] = ts(HALF)
        with self.assertRaises(OpUsageError):
            run_shape_inference(
                MS, "DynamicSparseAttention", inputs, self._heads(), opset_version=1
            )

    def test_dynamic_invalid_constant_length(self):
        specs = self._dynamic_inputs()
        specs[3] = specs[4] = None
        inputs = [
            ir.Value(name=f"i{i}", shape=s.shape, type=s.type) if s is not None else None
            for i, s in enumerate(specs)
        ]
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference_with_values(
                MS,
                "DynamicSparseAttention",
                inputs,
                self._heads(),
                opset_version=1,
                num_outputs=3,
                symbolic_values={10: [7, 8]},
            )
        with self.assertLogs("onnx_shape_inference._context", level="WARNING"):
            run_shape_inference_with_values(
                MS,
                "DynamicSparseAttention",
                inputs,
                self._heads(),
                opset_version=1,
                num_outputs=3,
                symbolic_values={10: [7, 8]},
                policy="skip",
            )

    def test_dynamic_no_cache_outputs(self):
        self.assertEqual(
            run_shape_inference(
                MS,
                "DynamicSparseAttention",
                self._dynamic_inputs(),
                self._heads(),
                opset_version=1,
            ),
            [ts(HALF, [2, 7, 128])],
        )

    @parameterized.parameterized.expand([(True,), (False,)])
    def test_dynamic_unknown_query(self, known_cache):
        inputs = self._dynamic_inputs()
        inputs[0] = ts(HALF)
        if not known_cache:
            inputs[3] = inputs[4] = None
        actual = run_shape_inference(
            MS, "DynamicSparseAttention", inputs, self._heads(), opset_version=1, num_outputs=3
        )
        self.assertEqual(
            actual,
            [
                ts(HALF),
                ts(HALF, [2, 2, 16, 32]) if known_cache else ts(HALF),
                ts(HALF, [2, 2, 16, 32]) if known_cache else ts(HALF),
            ],
        )
