# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ConvTranspose shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError, ShapeInferenceError
from onnx_shape_inference._ops._testing import (
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class ConvTransposeTest(unittest.TestCase):
    def test_opset_10(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            opset_version=10,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 5, 5])])

    def test_invalid_spatial_attributes_raise_shape_error(self):
        attrs = {"strides": ir.Attr("strides", ir.AttributeType.INTS, [0])}
        with self.assertRaises(ShapeInferenceError):
            run_shape_inference(
                "",
                "ConvTranspose",
                [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
                attrs,
                opset_version=11,
            )

    @parameterized.parameterized.expand(
        [
            ("input_rank", [1, 1], [1, 1], {}),
            ("weight_rank", [1, 1, 3, 3], [1, 1, 3], {}),
            ("output_shape", [1, 1, 3, 3], [1, 1, 3, 3], {"output_shape": [5]}),
            ("short_strides", [1, 1, 3, 3], [1, 1, 3, 3], {"strides": [1]}),
            ("zero_stride", [1, 1, 3, 3], [1, 1, 3, 3], {"strides": [0, 1]}),
            ("short_kernel", [1, 1, 3, 3], [1, 1, 3, 3], {"kernel_shape": [3]}),
            ("short_dilations", [1, 1, 3, 3], [1, 1, 3, 3], {"dilations": [1]}),
            ("short_pads", [1, 1, 3, 3], [1, 1, 3, 3], {"pads": [0, 0]}),
            ("short_output_padding", [1, 1, 3, 3], [1, 1, 3, 3], {"output_padding": [0]}),
        ]
    )
    def test_invalid_inputs_degrade_with_skip_policy(
        self, _name, input_shape, weight_shape, attr_values
    ):
        attrs = {
            name: ir.Attr(name, ir.AttributeType.INTS, value)
            for name, value in attr_values.items()
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, input_shape), ts(FLOAT, weight_shape)],
            attrs,
            opset_version=11,
            policy="skip",
        )
        self.assertIsNone(actual[0].shape)

    def test_symbolic_kernel_dim_is_unknown_output_dim(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, "K", 3])],
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, "_d0", 5])])

    def test_basic(self):
        # X=[1,1,3,3], W=[1,1,3,3], stride=1, no pads
        # output = 1*(3-1) + 0 + (3-1)*1+1 - 0 - 0 = 2+3 = 5
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 5, 5])])

    def test_with_strides(self):
        # stride=2: output = 2*(3-1) + 0 + (3-1)*1+1 - 0 - 0 = 4+3 = 7
        attrs = {
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2]),
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            attrs,
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 7, 7])])

    def test_with_output_shape_attr(self):
        attrs = {
            "output_shape": ir.Attr("output_shape", ir.AttributeType.INTS, [10, 10]),
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            attrs,
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 10, 10])])

    def test_auto_pad_same_upper(self):
        attrs = {
            "auto_pad": ir.Attr("auto_pad", ir.AttributeType.STRING, "SAME_UPPER"),
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2]),
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            attrs,
            opset_version=11,
        )
        # SAME: output = input * stride = 3*2 = 6
        self.assertEqual(actual[0].shape, ir.Shape([1, 1, 6, 6]))

    def test_missing_x_shape(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT), ts(FLOAT, [1, 1, 3, 3])],
            opset_version=11,
        )
        self.assertIsNone(actual[0].shape)

    def test_missing_w_shape(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT)],
            opset_version=11,
        )
        self.assertIsNone(actual[0].shape)

    def test_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "ConvTranspose", [], opset_version=11)

    def test_none_input(self):
        v = ir.Value(name="x", type=ir.TensorType(FLOAT), shape=ir.Shape([1, 1, 3, 3]))
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "ConvTranspose",
                [v, None],
                opset_version=11,
            )

    def test_with_pads(self):
        # stride=2, pads=[1,1,1,1]: output = 2*(3-1)+0+(3-1)*1+1-1-1 = 4+3-2 = 5
        attrs = {
            "strides": ir.Attr("strides", ir.AttributeType.INTS, [2, 2]),
            "pads": ir.Attr("pads", ir.AttributeType.INTS, [1, 1, 1, 1]),
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 1, 3, 3]), ts(FLOAT, [1, 1, 3, 3])],
            attrs,
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 1, 5, 5])])

    def test_group_conv_transpose(self):
        # group=2: W=[2, 1, 3, 3] → out_channels = 1*2 = 2
        attrs = {
            "group": ir.Attr("group", ir.AttributeType.INT, 2),
        }
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, [1, 2, 3, 3]), ts(FLOAT, [2, 1, 3, 3])],
            attrs,
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, [1, 2, 5, 5])])

    def test_symbolic_input_dims(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, ["N", "C", "H", "W"]), ts(FLOAT, ["C", 1, 3, 3])],
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", 1, "H + 2", "W + 2"])])

    def test_symbolic_input_concrete_kernel(self):
        actual = run_shape_inference(
            "",
            "ConvTranspose",
            [ts(FLOAT, ["N", 1, "H", "W"]), ts(FLOAT, [1, 1, 3, 3])],
            opset_version=11,
        )
        self.assertEqual(actual, [ts(FLOAT, ["N", 1, "H + 2", "W + 2"])])


if __name__ == "__main__":
    unittest.main()
