# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShapeMergePolicy in ShapeInferenceContext."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

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

    @parameterized.parameterized.expand(
        [
            # name, input dim, assume_divisible, expected
            ("concrete_int", 5, False, 5),
            ("concrete_int_divisible", 5, True, 5),
            ("named_symbol_unchanged", ir.SymbolicDim("N"), True, ir.SymbolicDim("N")),
            (
                "divisible_cancels_floor",
                ir.SymbolicDim("2*floor(c/2)"),
                True,
                ir.SymbolicDim("c"),
            ),
            (
                "default_keeps_floor",
                ir.SymbolicDim("2*floor(H/2)"),
                False,
                ir.SymbolicDim("2*floor(H/2)"),
            ),
            # Collapses to a concrete int (returned as int, not SymbolicDim).
            ("collapses_to_int", ir.SymbolicDim("(2*H)//H"), True, 2),
        ]
    )
    def test_simplify_dim(self, _name, dim, assume_divisible, expected):
        ctx = ShapeInferenceContext()
        self.assertEqual(ctx.simplify_dim(dim, assume_divisible=assume_divisible), expected)

    def test_unknown_symbolic_dim_returned_as_is(self):
        # An unknown placeholder has no expression to simplify.
        ctx = ShapeInferenceContext()
        dim = ir.SymbolicDim(None)
        self.assertIs(ctx.simplify_dim(dim), dim)

    def test_collapsed_int_is_python_int(self):
        ctx = ShapeInferenceContext()
        result = ctx.simplify_dim(ir.SymbolicDim("(2*H)//H"), assume_divisible=True)
        self.assertIsInstance(result, int)


class SymbolicEqualityTest(unittest.TestCase):
    """Tests for add_symbolic_equality / add_upper_bound recording."""

    @parameterized.parameterized.expand(
        [
            # name, a, b, expected pair recorded (or None to expect nothing)
            ("leaf", ir.SymbolicDim("_d0"), ir.SymbolicDim("dnz"), ("_d0", "dnz")),
            ("scaled", ir.SymbolicDim("2*_d0"), ir.SymbolicDim("2*dnz"), ("2*_d0", "2*dnz")),
            ("string_inputs", "_d0", "dnz", ("_d0", "dnz")),
            ("compound", "past_seq + seq", "total_seq", ("past_seq + seq", "total_seq")),
            # Concrete and identical operands record nothing.
            ("identical", ir.SymbolicDim("a"), ir.SymbolicDim("a"), None),
            ("both_concrete", 3, 3, None),
            ("distinct_concrete", 3, 4, None),
            # An unknown placeholder has no expression string -> nothing recorded.
            ("unknown_operand", ir.SymbolicDim(None), ir.SymbolicDim("N"), None),
        ]
    )
    def test_add_symbolic_equality(self, _name, a, b, expected):
        ctx = ShapeInferenceContext()
        ctx.add_symbolic_equality(a, b)
        if expected is None:
            self.assertEqual(list(ctx.symbolic_equalities), [])
        else:
            self.assertIn(expected, ctx.symbolic_equalities)

    def test_deduplicates_reversed_pair(self):
        ctx = ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "dnz")
        ctx.add_symbolic_equality("dnz", "_d0")
        self.assertEqual(len(ctx.symbolic_equalities), 1)

    def test_records_upper_bound(self):
        ctx = ShapeInferenceContext()
        ctx.add_upper_bound(ir.SymbolicDim("_d0"), ir.SymbolicDim("N"))
        self.assertIn(("_d0", "N"), ctx.symbolic_upper_bounds)

    def test_upper_bound_ignores_concrete_and_identical(self):
        ctx = ShapeInferenceContext()
        ctx.add_upper_bound(4, 8)
        ctx.add_upper_bound(ir.SymbolicDim("a"), ir.SymbolicDim("a"))
        self.assertEqual(list(ctx.symbolic_upper_bounds), [])

    @parameterized.parameterized.expand(
        [
            # name, anchor (existing) shape, inferred shape, expected pair or None
            ("named_anchor", ["N", "TopK_k"], ["N", "_d0"], ("_d0", "TopK_k")),
            (
                "compound_anchor",
                ["batch", "past_seq + seq"],
                ["batch", "_d0"],
                ("_d0", "past_seq + seq"),
            ),
            ("scaled_anchor", ["2*dnz"], ["2*_d0"], ("2*_d0", "2*dnz")),
            # Anchor itself anonymous: no user name to adopt.
            ("anonymous_anchor", ["N", "_d5"], ["N", "_d0"], None),
            # Anchor concrete: nothing to relate.
            ("concrete_anchor", [4, 5], [4, "_d0"], None),
        ]
    )
    def test_refine_records_anchor_equality(self, _name, anchor, inferred, expected):
        # The refine merge path is where declared anchors meet inferred symbols.
        value = ir.val("Y", shape=ir.Shape(anchor))
        ctx = ShapeInferenceContext(policy="refine")
        ctx.set_shape(value, ir.Shape(inferred))
        if expected is None:
            self.assertEqual(list(ctx.symbolic_equalities), [])
        else:
            self.assertIn(expected, ctx.symbolic_equalities)

    def test_override_policy_records_nothing(self):
        # The override policy bypasses refine, so no anchor equality is recorded.
        value = ir.val("Y", shape=ir.Shape(["N", "TopK_k"]))
        ctx = ShapeInferenceContext(policy="override")
        ctx.set_shape(value, ir.Shape(["N", "_d0"]))
        self.assertEqual(list(ctx.symbolic_equalities), [])


if __name__ == "__main__":
    unittest.main()
