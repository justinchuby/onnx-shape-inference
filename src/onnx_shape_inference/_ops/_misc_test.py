# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for miscellaneous operator shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64
UINT8 = ir.DataType.UINT8


class EyeLikeTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "EyeLike", [ts(FLOAT, [3, 4])], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, [3, 4])])

    def test_dtype_attr(self):
        attrs = {
            "dtype": ir.Attr("dtype", ir.AttributeType.INT, ir.DataType.DOUBLE.value),
        }
        actual = run_shape_inference(
            "", "EyeLike", [ts(FLOAT, [3, 4])], attrs, opset_version=21
        )
        self.assertEqual(actual, [ts(ir.DataType.DOUBLE, [3, 4])])

    def test_symbolic_dims(self):
        actual = run_shape_inference("", "EyeLike", [ts(FLOAT, ["N", "M"])], opset_version=21)
        self.assertEqual(actual, [ts(FLOAT, ["N", "M"])])

    def test_none_input_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values("", "EyeLike", [None], opset_version=21)


class ImageDecoderTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference("", "ImageDecoder", [ts(UINT8, [100])], opset_version=20)
        self.assertEqual(actual, [ts(UINT8, ["_d0", "_d1", 3])])

    def test_grayscale(self):
        attrs = {
            "pixel_format": ir.Attr("pixel_format", ir.AttributeType.STRING, "Grayscale"),
        }
        actual = run_shape_inference(
            "", "ImageDecoder", [ts(UINT8, [100])], attrs, opset_version=20
        )
        self.assertEqual(actual, [ts(UINT8, ["_d0", "_d1", 1])])


class MelWeightMatrixTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "MelWeightMatrix",
            [ts(INT64, []), ts(INT64, []), ts(FLOAT, []), ts(FLOAT, []), ts(INT64, [])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d0", "_d1"])])


class TfIdfVectorizerTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("repro", [1], [0], [1]),
            ("larger_feature_dim", [7], [0, 1, 2, 4], [5]),
            ("batched", [3, 7], [0, 1, 2, 4], [3, 5]),
            ("symbolic_batch", ["N", 7], [0, 1, 2, 4], ["N", 5]),
        ]
    )
    def test_feature_dim_from_ngram_indexes(
        self, _name, input_shape, ngram_indexes, expected_shape
    ):
        attrs = {
            "ngram_indexes": ir.Attr("ngram_indexes", ir.AttributeType.INTS, ngram_indexes)
        }
        actual = run_shape_inference(
            "",
            "TfIdfVectorizer",
            [ts(INT64, input_shape)],
            attrs,
            opset_version=9,
        )
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_unknown_input_shape_sets_dtype_only(self):
        attrs = {
            "ngram_indexes": ir.Attr("ngram_indexes", ir.AttributeType.INTS, [0, 1, 2, 4])
        }
        actual = run_shape_inference(
            "", "TfIdfVectorizer", [ts(INT64)], attrs, opset_version=9
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_missing_ngram_indexes_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "TfIdfVectorizer", [ts(INT64, [7])], opset_version=9)

    def test_empty_ngram_indexes_raises(self):
        attrs = {
            "ngram_indexes": ir.Attr("ngram_indexes", ir.AttributeType.INTS, []),
        }
        with self.assertRaisesRegex(OpUsageError, "ngram_indexes must be non-empty"):
            run_shape_inference(
                "", "TfIdfVectorizer", [ts(INT64, [7])], attrs, opset_version=9
            )


if __name__ == "__main__":
    unittest.main()
