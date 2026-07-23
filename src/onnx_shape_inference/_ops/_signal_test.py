# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for signal processing shape inference (DFT, STFT)."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import OpUsageError
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class DFTTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT, [1, 10, 2])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 10, 2])])

    def test_missing_shape(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT)],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT)])

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "DFT", [], opset_version=20)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "DFT",
                [None],
                opset_version=20,
            )

    def test_opset_17_default_axis(self):
        actual = run_shape_inference_with_values(
            "",
            "DFT",
            [
                ir.Value(
                    name="input",
                    type=ir.TensorType(FLOAT),
                    shape=ir.Shape([2, 3, 16, 2]),
                ),
                const_value(8),
            ],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 16, 2])])

    def test_opset_20_default_axis(self):
        actual = run_shape_inference_with_values(
            "",
            "DFT",
            [
                ir.Value(
                    name="input",
                    type=ir.TensorType(FLOAT),
                    shape=ir.Shape([2, 3, 16, 2]),
                ),
                const_value(8),
            ],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 8, 2])])

    def test_opset_20_axis_input(self):
        actual = run_shape_inference_with_values(
            "",
            "DFT",
            [
                ir.Value(
                    name="input",
                    type=ir.TensorType(FLOAT),
                    shape=ir.Shape([2, 3, 16, 2]),
                ),
                const_value(8),
                const_value(1),
            ],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 8, 16, 2])])

    def test_opset_20_dynamic_axis_makes_signal_dims_unknown(self):
        actual = run_shape_inference_with_values(
            "",
            "DFT",
            [
                ir.Value(
                    name="input",
                    type=ir.TensorType(FLOAT),
                    shape=ir.Shape([2, 3, 5, 1]),
                ),
                None,
                ir.Value(
                    name="axis",
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape([]),
                ),
            ],
            attributes={"onesided": ir.Attr("onesided", ir.AttributeType.INT, 1)},
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d0", "_d1", "_d2", 2])])

    def test_opset_20_dynamic_dft_length_makes_axis_dim_unknown(self):
        for onesided in (0, 1):
            with self.subTest(onesided=onesided):
                actual = run_shape_inference_with_values(
                    "",
                    "DFT",
                    [
                        ir.Value(
                            name="input",
                            type=ir.TensorType(FLOAT),
                            shape=ir.Shape([2, 10, 1]),
                        ),
                        ir.Value(
                            name="dft_length",
                            type=ir.TensorType(ir.DataType.INT64),
                            shape=ir.Shape([]),
                        ),
                    ],
                    attributes={
                        "onesided": ir.Attr("onesided", ir.AttributeType.INT, onesided)
                    },
                    opset_version=20,
                )
                self.assertEqual(actual, [ts(FLOAT, [2, "_d0", 2])])

    def test_inverse_onesided_dft_length_is_output_length(self):
        actual = run_shape_inference_with_values(
            "",
            "DFT",
            [
                ir.Value(
                    name="input",
                    type=ir.TensorType(FLOAT),
                    shape=ir.Shape([2, 3, 5, 2]),
                ),
                const_value(8),
            ],
            attributes={
                "inverse": ir.Attr("inverse", ir.AttributeType.INT, 1),
                "onesided": ir.Attr("onesided", ir.AttributeType.INT, 1),
            },
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, [2, 3, 8, 1])])


class STFTTest(unittest.TestCase):
    def test_basic(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT, [1, 128, 1]), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, "_d0", "_d1", 2])])

    def test_missing_signal_shape(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "STFT", [], opset_version=17)

    def test_none_input(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "STFT",
                [None, None],
                opset_version=17,
            )


class DFTSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_dims(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT, ["N", "L", 2])],
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", "L", 2])])

    def test_onesided_symbolic_axis(self):
        actual = run_shape_inference(
            "",
            "DFT",
            [ts(FLOAT, ["batch", "length", 2])],
            {"onesided": ir.Attr("onesided", ir.AttributeType.INT, 1)},
            opset_version=20,
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", "floor(length/2) + 1", 2])])


class STFTSymbolicDimsTest(unittest.TestCase):
    def test_symbolic_batch(self):
        actual = run_shape_inference(
            "",
            "STFT",
            [ts(FLOAT, ["N", 128, 1]), ts(ir.DataType.INT64, [])],
            opset_version=17,
        )
        self.assertEqual(actual[0].shape[0], ir.SymbolicDim("N"))
        self.assertEqual(actual[0].shape[3], 2)


if __name__ == "__main__":
    unittest.main()
