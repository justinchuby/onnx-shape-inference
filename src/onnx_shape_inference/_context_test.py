# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShapeMergePolicy in ShapeInferenceContext."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference._context import ShapeInferenceContext


class ShapeMergePolicyTest(unittest.TestCase):
    """Tests for ShapeMergePolicy in context."""

    def test_skip_policy_keeps_existing(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="skip")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_override_policy_replaces(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="override")

        modified = ctx.set_shape(value, ir.Shape([4, 5, 6]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [4, 5, 6])

    def test_refine_policy_updates_unknown_to_known(self):
        value = ir.val("test", shape=ir.Shape([None, 2, 3]))
        ctx = ShapeInferenceContext(policy="refine")

        modified = ctx.set_shape(value, ir.Shape([1, 2, 3]))
        self.assertTrue(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_refine_policy_keeps_concrete(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="refine")

        # Try to refine with symbolic - should keep concrete
        modified = ctx.set_shape(value, ir.Shape(["batch", 2, 3]))
        self.assertFalse(modified)
        self.assertEqual(value.shape, [1, 2, 3])

    def test_strict_policy_raises_on_conflict(self):
        value = ir.val("test", shape=ir.Shape([1, 2, 3]))
        ctx = ShapeInferenceContext(policy="strict")

        with self.assertRaises(ValueError) as cm:
            ctx.set_shape(value, ir.Shape([4, 2, 3]))
        self.assertIn("conflict", str(cm.exception).lower())


class SimplifyDimTest(unittest.TestCase):
    """Tests for ShapeInferenceContext.simplify_dim."""

    def test_int_returned_unchanged(self):
        ctx = ShapeInferenceContext()
        self.assertEqual(ctx.simplify_dim(5), 5)

    def test_unknown_symbolic_dim_unchanged(self):
        ctx = ShapeInferenceContext()
        dim = ir.SymbolicDim(None)
        self.assertIs(ctx.simplify_dim(dim), dim)

    def test_divisible_collapses_to_int_or_symbol(self):
        ctx = ShapeInferenceContext()
        result = ctx.simplify_dim(ir.SymbolicDim("2*floor(c/2)"), assume_divisible=True)
        self.assertEqual(result, ir.SymbolicDim("c"))

    def test_default_keeps_floor(self):
        ctx = ShapeInferenceContext()
        dim = ir.SymbolicDim("2*floor(H/2)")
        result = ctx.simplify_dim(dim)
        self.assertEqual(result, ir.SymbolicDim("2*floor(H/2)"))


class SymbolicEqualityTest(unittest.TestCase):
    """Tests for add_symbolic_equality / add_upper_bound recording."""

    def test_records_equality(self):
        ctx = ShapeInferenceContext()
        ctx.add_symbolic_equality(ir.SymbolicDim("_d0"), ir.SymbolicDim("dnz"))
        self.assertIn(("_d0", "dnz"), ctx.symbolic_equalities)

    def test_ignores_identical_and_concrete(self):
        ctx = ShapeInferenceContext()
        ctx.add_symbolic_equality(ir.SymbolicDim("a"), ir.SymbolicDim("a"))
        ctx.add_symbolic_equality(3, 3)
        self.assertEqual(list(ctx.symbolic_equalities), [])

    def test_deduplicates_reversed_pair(self):
        ctx = ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "dnz")
        ctx.add_symbolic_equality("dnz", "_d0")
        self.assertEqual(len(ctx.symbolic_equalities), 1)

    def test_records_upper_bound(self):
        ctx = ShapeInferenceContext()
        ctx.add_upper_bound(ir.SymbolicDim("_d0"), ir.SymbolicDim("N"))
        self.assertIn(("_d0", "N"), ctx.symbolic_upper_bounds)

    def test_refine_records_anchor_equality(self):
        # Declared anchor (existing) meets a different inferred symbol.
        value = ir.val("Y", shape=ir.Shape(["N", "TopK_k"]))
        ctx = ShapeInferenceContext(policy="refine")
        ctx.set_shape(value, ir.Shape(["N", "_d0"]))
        self.assertIn(("_d0", "TopK_k"), ctx.symbolic_equalities)

    def test_refine_ignores_anonymous_anchor(self):
        # Existing side is itself anonymous: no user name to adopt.
        value = ir.val("Y", shape=ir.Shape(["N", "_d5"]))
        ctx = ShapeInferenceContext(policy="refine")
        ctx.set_shape(value, ir.Shape(["N", "_d0"]))
        self.assertEqual(list(ctx.symbolic_equalities), [])


if __name__ == "__main__":
    unittest.main()
