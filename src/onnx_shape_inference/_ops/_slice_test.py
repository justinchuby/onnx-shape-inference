# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Slice shape inference."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import OpUsageError
from onnx_shape_inference._ops._testing import (
    const_value,
    run_shape_inference,
    run_shape_inference_with_values,
    ts,
)

FLOAT = ir.DataType.FLOAT


class SliceTest(unittest.TestCase):
    def _run(self, input_ts, starts, ends, axes=None, steps=None):
        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        inputs = [
            data,
            const_value(starts, "starts"),
            const_value(ends, "ends"),
        ]
        if axes is not None:
            inputs.append(const_value(axes, "axes"))
        if steps is not None:
            if axes is None:
                inputs.append(ir.Value(name="axes_empty"))
            inputs.append(const_value(steps, "steps"))
        return run_shape_inference_with_values(
            "",
            "Slice",
            inputs,
            opset_version=17,
        )

    @parameterized.parameterized.expand(
        [
            ("basic", [10, 20], [1], [5], [0], [1], [4, 20]),
            ("negative_end", [10, 20], [0], [-1], [0], [1], [9, 20]),
            ("step_2", [10, 20], [0], [10], [0], [2], [5, 20]),
            ("axis_1", [10, 20], [2], [8], [1], [1], [10, 6]),
            ("multi_axis", [10, 20, 30], [1, 2], [5, 10], [0, 1], [1, 1], [4, 8, 30]),
            # From ONNX test_slice_giant_number: large end value clipped
            ("giant_end", [3, 2], [0, 0], [3, 2147483647], [0, 1], [1, 1], [3, 2]),
            # From ONNX test_slice_giant_step
            ("giant_step", [3, 2], [0, 0], [3, 2], [0, 1], [1, 2147483647], [3, 1]),
            # From ONNX test_slice_negative_start
            ("negative_start", [3, 2], [-2, 0], [3, 2], [0, 1], [1, 1], [2, 2]),
        ]
    )
    def test_slice(self, _name, shape, starts, ends, axes, steps, expected_shape):
        actual = self._run(ts(FLOAT, shape), starts, ends, axes, steps)
        self.assertEqual(actual, [ts(FLOAT, expected_shape)])

    def test_negative_step(self):
        # From ONNX test_slice_negative_step: backward slicing
        actual = self._run(ts(FLOAT, [3, 4]), [2, 3], [0, 1], [0, 1], [-1, -1])
        self.assertEqual(actual, [ts(FLOAT, [2, 2])])

    def test_no_axes_defaults_to_range(self):
        """When axes is not provided, defaults to [0, 1, ..., len(starts)-1]."""
        actual = self._run(ts(FLOAT, [3, 2]), [1, 0], [2, 2])
        self.assertEqual(actual, [ts(FLOAT, [1, 2])])

    def test_symbolic_dim_const_start_end(self):
        """Slicing on a symbolic dim with non-negative const start/end computes size."""
        actual = self._run(ts(FLOAT, ["N", 10]), [1], [5], [0], [1])
        self.assertEqual(actual, [ts(FLOAT, [4, 10])])

    def test_symbolic_dim_const_start_end_step_2(self):
        """Slicing on a symbolic dim with step=2."""
        actual = self._run(ts(FLOAT, ["N", 10]), [0], [6], [0], [2])
        self.assertEqual(actual, [ts(FLOAT, [3, 10])])

    def test_symbolic_dim_negative_start_becomes_unknown(self):
        """Negative start depends on actual dim size, so result is unknown."""
        actual = self._run(ts(FLOAT, ["N", 10]), [-2], [5], [0], [1])
        self.assertEqual(actual, [ts(FLOAT, ["_d0", 10])])

    def test_symbolic_dim_preserved_on_non_sliced_axis(self):
        """Symbolic dim on non-sliced axis is preserved; sliced axis is concrete."""
        actual = self._run(ts(FLOAT, ["a", 2]), [0], [1], [1], [1])
        self.assertEqual(actual, [ts(FLOAT, ["a", 1])])

    def test_symbolic_input_const_slice(self):
        """Slice on ["N", "C"] with const starts/ends on axis 1 → ["N", 6]."""
        actual = self._run(ts(FLOAT, ["N", 10]), [2], [8], [1], [1])
        self.assertEqual(actual, [ts(FLOAT, ["N", 6])])

    def test_symbolic_dim_sentinel_end_forward_preserves_dim(self):
        """start=0, step=1, end=INT_MAX on symbolic dim → preserves dim."""
        actual = self._run(ts(FLOAT, ["N", 10]), [0], [2**63 - 1], [0], [1])
        self.assertEqual(actual, [ts(FLOAT, ["N", 10])])

    def test_symbolic_dim_sentinel_end_int32_max(self):
        """start=0, step=1, end=INT32_MAX on symbolic dim → preserves dim."""
        actual = self._run(ts(FLOAT, ["N", 10]), [0], [2**31 - 1], [0], [1])
        self.assertEqual(actual, [ts(FLOAT, ["N", 10])])

    def test_symbolic_dim_sentinel_reverse_preserves_dim(self):
        """Full reverse: start=INT_MAX, end=INT_MIN, step=-1 → preserves dim."""
        actual = self._run(ts(FLOAT, ["N", 10]), [2**63 - 1], [-(2**63)], [0], [-1])
        self.assertEqual(actual, [ts(FLOAT, ["N", 10])])

    def test_missing_input_shape(self):
        data = ir.Value(name="data", type=ir.TensorType(FLOAT))
        starts = const_value([0], "starts")
        ends = const_value([5], "ends")
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertIsNone(actual[0].shape)

    def test_dynamic_starts_ends(self):
        """When starts/ends are not const, output shape has same rank with symbolic dims."""
        data = ir.Value(name="data", shape=ir.Shape([10, 20]), type=ir.TensorType(FLOAT))
        starts = ir.Value(name="starts", type=ir.TensorType(ir.DataType.INT64))
        ends = ir.Value(name="ends", type=ir.TensorType(ir.DataType.INT64))
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d0", "_d1"])])

    def test_slice_no_inputs(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference("", "Slice", [ts(FLOAT, [5])], opset_version=17)

    def test_slice_none_data(self):
        starts = const_value([0])
        ends = const_value([3])
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Slice",
                [None, starts, ends],
                opset_version=17,
            )

    def test_slice_non_const_starts(self):
        """Non-constant starts/ends → output shape has same rank with symbolic dims."""
        data = ir.Value(name="data", type=ir.TensorType(FLOAT), shape=ir.Shape([10, 20]))
        starts = ir.Value(
            name="starts", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1])
        )
        ends = ir.Value(
            name="ends", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1])
        )
        actual = run_shape_inference_with_values(
            "",
            "Slice",
            [data, starts, ends],
            opset_version=17,
        )
        self.assertEqual(actual, [ts(FLOAT, ["_d0", "_d1"])])

    def _run_with_symbolic_ends(self, input_ts, starts, ends_sym, axes=None, steps=None):
        """Run Slice inference with a symbolic (non-constant) ends input."""
        from onnx_shape_inference import _context, _registry

        data = ir.Value(name="data", shape=input_ts.shape, type=input_ts.type)
        ends_val = ir.Value(
            name="ends", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([len(ends_sym)])
        )
        inputs = [
            data,
            const_value(starts, "starts"),
            ends_val,
        ]
        if axes is not None:
            inputs.append(const_value(axes, "axes"))
        if steps is not None:
            if axes is None:
                inputs.append(ir.Value(name="axes_empty"))
            inputs.append(const_value(steps, "steps"))

        output_values = [ir.Value(name="output_0")]
        node = ir.Node("", "Slice", inputs=inputs, outputs=output_values, attributes={})
        ctx = _context.ShapeInferenceContext({"": 17})
        ctx.name_anonymous_dims(data)
        # Set symbolic value for ends
        ctx.set_symbolic_value(ends_val, ends_sym)

        func = _registry.registry.get("", "Slice", version=17)
        func(ctx, node)
        return [ir.TypeAndShape(v.type, v.shape) for v in output_values]

    def test_symbolic_ends_match_input_shape(self):
        """Slice with symbolic ends matching input dims is a no-op per axis."""
        batch = ir.SymbolicDim("batch")
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, ["batch", 100]),
            starts=[0, 0],
            ends_sym=[batch, 100],
            axes=[0, 1],
            steps=[1, 1],
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 100])])

    def test_symbolic_ends_no_match_gives_new_dim(self):
        """Slice with symbolic end that doesn't match input dim gives a new symbolic dim."""
        other = ir.SymbolicDim("other")
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, ["batch", 100]),
            starts=[0, 0],
            ends_sym=[other, 100],
            axes=[0, 1],
            steps=[1, 1],
        )
        # axis 0: end=other != batch → new symbolic dim; axis 1: end=100 concrete → 100
        self.assertIsNotNone(actual[0].shape)
        self.assertNotEqual(actual[0].shape[0], ir.SymbolicDim("batch"))
        self.assertEqual(actual[0].shape[1], 100)

    def test_symbolic_ends_start_nonzero_gives_new_dim(self):
        """Slice with symbolic end and start!=0 gives a new symbolic dim."""
        batch = ir.SymbolicDim("batch")
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, ["batch", 100]),
            starts=[1, 0],
            ends_sym=[batch, 100],
            axes=[0, 1],
            steps=[1, 1],
        )
        self.assertIsNotNone(actual[0].shape)
        # axis 0: start=1, so even though end=batch=input_dim, it's not identity
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 100)

    def test_symbolic_ends_step_not_one_gives_new_dim(self):
        """Slice with symbolic end and step!=1 gives a new symbolic dim."""
        batch = ir.SymbolicDim("batch")
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, ["batch", 100]),
            starts=[0, 0],
            ends_sym=[batch, 100],
            axes=[0, 1],
            steps=[2, 1],
        )
        self.assertIsNotNone(actual[0].shape)
        # axis 0: step=2, so even though start=0 and end=batch, result is not batch
        self.assertIsInstance(actual[0].shape[0], ir.SymbolicDim)
        self.assertEqual(actual[0].shape[1], 100)

    def test_symbolic_ends_partial_axes(self):
        """Slice with symbolic ends on a subset of axes preserves non-sliced dims."""
        batch = ir.SymbolicDim("batch")
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, ["batch", 100, 200]),
            starts=[0],
            ends_sym=[batch],
            axes=[0],
            steps=[1],
        )
        self.assertEqual(actual, [ts(FLOAT, ["batch", 100, 200])])

    def test_symbolic_ends_concrete_dim_computed(self):
        """Slice with symbolic ends on a concrete dim computes the result."""
        actual = self._run_with_symbolic_ends(
            ts(FLOAT, [10, 20]),
            starts=[0, 0],
            ends_sym=[10, 20],
            axes=[0, 1],
            steps=[1, 1],
        )
        self.assertEqual(actual, [ts(FLOAT, [10, 20])])


if __name__ == "__main__":
    unittest.main()
