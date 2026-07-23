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
        # Names spelled `_dN` in the anchor stand in for engine-minted symbols;
        # register them so eligibility is decided by minted-identity (not spelling).
        for dim in anchor:
            if isinstance(dim, str) and dim.startswith("_d") and dim[2:].isdigit():
                ctx._allocator.generated.add(dim)
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


class NewSymbolicDimTest(unittest.TestCase):
    """Tests for new_symbolic_dim minting, its registry, and collision avoidance."""

    def test_mints_sequential_unique_names(self):
        ctx = ShapeInferenceContext()
        names = [ctx.new_symbolic_dim().value for _ in range(3)]
        self.assertEqual(names, ["_d0", "_d1", "_d2"])
        self.assertEqual(len(set(names)), 3)

    def test_minted_names_are_reported_generated(self):
        ctx = ShapeInferenceContext()
        dim = ctx.new_symbolic_dim()
        self.assertTrue(ctx.is_generated_symbol(dim.value))
        self.assertIn(dim.value, ctx.generated_dim_names)

    def test_unminted_name_is_not_generated(self):
        # A name that merely LOOKS anonymous but was never minted is not eligible.
        ctx = ShapeInferenceContext()
        self.assertFalse(ctx.is_generated_symbol("_d0"))
        self.assertFalse(ctx.is_generated_symbol("N"))

    def test_reserved_name_is_skipped_when_minting(self):
        # A reserved (author-declared) `_d0` must not be reused by minting.
        ctx = ShapeInferenceContext()
        ctx.reserve_symbol_names({"_d0"})
        minted = ctx.new_symbolic_dim().value
        self.assertNotEqual(minted, "_d0")
        self.assertFalse(ctx.is_generated_symbol("_d0"))
        self.assertTrue(ctx.is_generated_symbol(minted))

    def test_reserved_run_of_names_all_skipped(self):
        ctx = ShapeInferenceContext()
        ctx.reserve_symbol_names({"_d0", "_d1", "_d2"})
        minted = [ctx.new_symbolic_dim().value for _ in range(2)]
        self.assertEqual(minted, ["_d3", "_d4"])

    def test_generated_dim_names_is_immutable_snapshot(self):
        ctx = ShapeInferenceContext()
        ctx.new_symbolic_dim()
        snapshot = ctx.generated_dim_names
        ctx.new_symbolic_dim()
        # The earlier snapshot is not retroactively mutated.
        self.assertEqual(snapshot, frozenset({"_d0"}))


class ChildContextAllocatorTest(unittest.TestCase):
    """A child context must share the parent's symbol allocator state.

    Regression tests for the re-review gap: function-body child contexts used to
    share only the counter, so (a) a child could mint a name the parent reserved
    and (b) names a child minted were not recognised as generated at the parent.
    """

    def test_child_shares_allocator_object(self):
        parent = ShapeInferenceContext()
        child = parent.create_child()
        self.assertIs(child._allocator, parent._allocator)

    def test_child_inherits_parent_reserved_names(self):
        # Parent reserves an author-declared `_d0`; the child must not mint it.
        parent = ShapeInferenceContext()
        parent.reserve_symbol_names({"_d0"})
        child = parent.create_child()
        minted = child.new_symbolic_dim().value
        self.assertNotEqual(minted, "_d0")
        self.assertFalse(child.is_generated_symbol("_d0"))
        self.assertFalse(parent.is_generated_symbol("_d0"))

    def test_child_reservation_visible_to_parent(self):
        parent = ShapeInferenceContext()
        child = parent.create_child()
        child.reserve_symbol_names({"_d0"})
        # Parent honours the child's reservation too (shared allocator).
        self.assertNotEqual(parent.new_symbolic_dim().value, "_d0")

    def test_child_minted_symbol_is_generated_at_parent(self):
        parent = ShapeInferenceContext()
        child = parent.create_child()
        minted = child.new_symbolic_dim().value
        # Recognised as engine-generated upstream, enabling correct renaming.
        self.assertTrue(parent.is_generated_symbol(minted))
        self.assertIn(minted, parent.generated_dim_names)

    def test_parent_and_child_names_stay_globally_unique(self):
        parent = ShapeInferenceContext()
        p0 = parent.new_symbolic_dim().value
        child = parent.create_child()
        c0 = child.new_symbolic_dim().value
        p1 = parent.new_symbolic_dim().value
        self.assertEqual([p0, c0, p1], ["_d0", "_d1", "_d2"])
        self.assertEqual(len({p0, c0, p1}), 3)

    def test_child_inherits_opset_and_policy_by_default(self):
        parent = ShapeInferenceContext({"": 17}, policy="skip")
        child = parent.create_child()
        self.assertEqual(child.opset, 17)
        self.assertEqual(child.policy, "skip")

    def test_child_can_override_opsets_and_attrs(self):
        parent = ShapeInferenceContext({"": 17})
        attrs = {"axis": ir.Attr("axis", ir.AttributeType.INT, 1)}
        child = parent.create_child({"": 21}, resolved_attrs=attrs)
        self.assertEqual(child.opset, 21)
        self.assertIs(child.resolved_attrs, attrs)


if __name__ == "__main__":
    unittest.main()
