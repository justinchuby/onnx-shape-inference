# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Range shape inference."""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

import onnx_shape_inference
from onnx_shape_inference import OpUsageError
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


class RangeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("float", FLOAT),
            ("int64", INT64),
        ]
    )
    def test_basic(self, _name, dtype):
        actual = run_shape_inference(
            "", "Range", [ts(dtype, []), ts(dtype, []), ts(dtype, [])], opset_version=21
        )
        self.assertEqual(actual, [ts(dtype, ["_d0"])])

    def test_none_input_raises(self):
        v1 = ir.Value(name="limit", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        v2 = ir.Value(name="delta", type=ir.TensorType(FLOAT), shape=ir.Shape([]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "Range", [None, v1, v2], opset_version=21)

    def test_delta_zero_raises(self):
        """Delta == 0 should record an error."""
        start = const_value([0], name="start", dtype=np.float32)
        start.shape = ir.Shape([])
        start.type = ir.TensorType(FLOAT)
        limit = const_value([10], name="limit", dtype=np.float32)
        limit.shape = ir.Shape([])
        limit.type = ir.TensorType(FLOAT)
        delta = const_value([0], name="delta", dtype=np.float32)
        delta.shape = ir.Shape([])
        delta.type = ir.TensorType(FLOAT)
        with self.assertRaises(onnx_shape_inference.ShapeInferenceError):
            run_shape_inference_with_values(
                "", "Range", [start, limit, delta], opset_version=21
            )

    def test_negative_delta(self):
        """Negative delta: Range(10, 0, -3) → len = ceil((0-10)/-3) = 4."""
        start = const_value([10], name="start", dtype=np.int64)
        start.shape = ir.Shape([])
        start.type = ir.TensorType(INT64)
        limit = const_value([0], name="limit", dtype=np.int64)
        limit.shape = ir.Shape([])
        limit.type = ir.TensorType(INT64)
        delta = const_value([-3], name="delta", dtype=np.int64)
        delta.shape = ir.Shape([])
        delta.type = ir.TensorType(INT64)
        actual = run_shape_inference_with_values(
            "", "Range", [start, limit, delta], opset_version=21
        )
        # ceil((0 - 10) / -3) = ceil(10/3) = 4
        self.assertEqual(actual, [ts(INT64, [4])])


if __name__ == "__main__":
    unittest.main()
