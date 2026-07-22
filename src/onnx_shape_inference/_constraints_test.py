# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the anchor/constraint propagation pass (_constraints.py)."""

from __future__ import annotations

import unittest

import onnx_ir as ir
import parameterized

from onnx_shape_inference import _constraints, _context, _symbolic_shapes


def _sym(name: str) -> ir.SymbolicDim:
    return ir.SymbolicDim(name)


def _identity_graph(shapes: dict[str, list | None]) -> ir.Graph:
    """Build a trivial single Identity-node graph whose values carry *shapes*.

    The first entry is the graph input; the remaining entries are outputs.  A
    ``None`` shape produces an unshaped value.
    """
    values = {
        name: ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=None if dims is None else ir.Shape(dims),
        )
        for name, dims in shapes.items()
    }
    names = list(values)
    node = ir.Node(
        "",
        "Identity",
        inputs=[values[names[0]]],
        outputs=[values[names[1]]] if len(names) > 1 else None,
    )
    return ir.Graph(
        [values[names[0]]],
        list(values.values())[1:] or [values[names[0]]],
        nodes=[node],
        opset_imports={"": 21},
    )


class LeafEqualityTest(unittest.TestCase):
    """Direct tests for the reusable _leaf_equality helper."""

    @parameterized.parameterized.expand(
        [
            # name, a, b, expected leaf pair (as a set) or None
            ("bare", "_d0", "dnz", {"_d0", "dnz"}),
            ("scaled_equal", "2*_d0", "2*dnz", {"_d0", "dnz"}),
            ("negative_scaled", "3*a", "3*b", {"a", "b"}),
            # A constant term prevents a clean leaf equality.
            ("with_constant", "_d0", "dnz + 1", None),
            # Three symbol terms -> not a leaf equality (handled as compound).
            ("three_terms", "past + seq", "total", None),
            # Unequal magnitudes -> not a leaf equality.
            ("unequal_coeffs", "2*a", "3*b", None),
            # Identical expressions carry no information.
            ("identical", "a", "a", None),
        ]
    )
    def test_leaf_equality(self, _name, a, b, expected):
        a_expr = _symbolic_shapes.parse_symbolic_expression(a)
        b_expr = _symbolic_shapes.parse_symbolic_expression(b)
        result = _constraints._leaf_equality(a_expr, b_expr)
        if expected is None:
            self.assertIsNone(result)
        else:
            self.assertEqual(set(result), expected)


class CanonicalNameTest(unittest.TestCase):
    """Direct tests for the reusable _canonical_name helper."""

    @parameterized.parameterized.expand(
        [
            # Declared names always win over anonymous ones.
            ("prefers_declared", ["_d0", "N"], "N"),
            ("prefers_declared_multi", ["_d3", "_d0", "total_seq"], "total_seq"),
            # Ties among declared names broken by (length, lexicographic).
            ("shortest_declared", ["batch", "B"], "B"),
            ("lexicographic_tie", ["bb", "aa"], "aa"),
            # No declared name: same ordering over anon names.
            ("all_anonymous", ["_d2", "_d10", "_d1"], "_d1"),
        ]
    )
    def test_canonical_name(self, _name, names, expected):
        self.assertEqual(_constraints._canonical_name(names), expected)


class PropagateSymbolicConstraintsTest(unittest.TestCase):
    def test_no_equalities_is_noop(self):
        graph = _identity_graph({"x": ["_d0"], "y": ["_d0"]})
        ctx = _context.ShapeInferenceContext()
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    def test_equalities_without_renamable_symbols_is_noop(self):
        # Both sides declared: no _dN to rename, so no change is reported.
        graph = _identity_graph({"x": ["A"], "y": ["A"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("A", "B")
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(graph.inputs[0].shape[0], _sym("A"))

    def test_leaf_rename_across_graph(self):
        graph = _identity_graph({"x": ["N", "_d0"], "y": ["N", "_d0"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "TopK_k")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        for value in list(graph.inputs) + list(graph.outputs):
            self.assertEqual(value.shape[-1], _sym("TopK_k"))

    def test_transitive_rename(self):
        # _d0 == _d1 and _d1 == N collapse both anon symbols to N.
        graph = _identity_graph({"x": ["_d0"], "y": ["_d1"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "_d1")
        ctx.add_symbolic_equality("_d1", "N")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], _sym("N"))
        self.assertEqual(list(graph.outputs)[0].shape[0], _sym("N"))

    def test_scaled_leaf_equality_rewrites_compound(self):
        graph = _identity_graph({"x": ["_d0"], "y": ["2*_d0"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("2*_d0", "2*dnz")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], _sym("dnz"))
        self.assertEqual(list(graph.outputs)[0].shape[0], _sym("2*dnz"))

    def test_compound_replacement(self):
        graph = _identity_graph(
            {"x": ["batch", "past_seq + seq"], "y": ["batch", "past_seq + seq"]}
        )
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("past_seq + seq", "total_seq")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        for value in list(graph.inputs) + list(graph.outputs):
            self.assertEqual(value.shape[-1], _sym("total_seq"))

    def test_declared_names_never_renamed(self):
        graph = _identity_graph({"x": ["A"], "y": ["A"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("A", "B")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], _sym("A"))

    def test_concrete_and_unrelated_dims_preserved(self):
        # Concrete dims and symbols not in the map are left untouched.
        graph = _identity_graph({"x": [4, "M", "_d0"], "y": [4, "M", "_d0"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "dnz")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        out = list(graph.outputs)[0].shape
        self.assertEqual(out[0], 4)
        self.assertEqual(out[1], _sym("M"))
        self.assertEqual(out[2], _sym("dnz"))

    def test_unknown_shape_is_skipped(self):
        # A value with no shape must not raise and stays unshaped.
        graph = _identity_graph({"x": ["_d0"], "y": None})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "N")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertIsNone(list(graph.outputs)[0].shape)
        self.assertEqual(graph.inputs[0].shape[0], _sym("N"))

    def test_propagates_into_subgraph(self):
        # A symbol declared on the outer graph is renamed inside an If branch.
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        body_out = ir.Value(
            name="body_out",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["_d0"]),
        )
        body = ir.Graph([], [body_out], nodes=[], opset_imports={"": 21}, name="then_branch")
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            num_outputs=1,
            attributes=[ir.Attr("then_branch", ir.AttributeType.GRAPH, body)],
        )
        y = if_node.outputs[0]
        y.name = "Y"
        y.shape = ir.Shape(["_d0"])
        graph = ir.Graph([cond], [y], nodes=[if_node], opset_imports={"": 21})

        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "B1")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        self.assertEqual(y.shape[0], _sym("B1"))
        self.assertEqual(body_out.shape[0], _sym("B1"))


if __name__ == "__main__":
    unittest.main()
