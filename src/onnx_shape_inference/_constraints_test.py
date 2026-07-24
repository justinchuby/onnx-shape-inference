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


def _anon(name: str) -> bool:
    """Spelling-based generated-name predicate for pure-helper unit tests."""
    return name.startswith("_d") and name[2:].isdigit()


def _generated_ctx(*generated_names: str) -> _context.ShapeInferenceContext:
    """A context that reports *generated_names* as engine-minted.

    Mirrors what :meth:`ShapeInferenceContext.new_symbolic_dim` records, without
    forcing tests to mint dims in counter order.  Only these names are eligible
    for renaming by the propagation pass.
    """
    ctx = _context.ShapeInferenceContext()
    ctx._allocator.generated.update(generated_names)
    return ctx


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


class ReduceCommonFactorTest(unittest.TestCase):
    """Direct tests for the reusable _reduce_common_factor helper (ISSUE B)."""

    @parameterized.parameterized.expand(
        [
            # A reshape numel equality carrying a divisibility fact reduces to
            # the provenance ``c == 2*floor(c/2)`` after cancelling ``a*b``.
            ("reshape_numel", "a*b*c", "2*a*b*floor(c/2)", "c", "2*floor(c/2)"),
            # A trivial numel equality (inherited floor, no new factor) is a
            # tautology and stays unchanged -> constrains nothing.
            (
                "inherited_floor",
                "2*floor(H/2)",
                "2*floor(H/2)",
                "2*floor(H/2)",
                "2*floor(H/2)",
            ),
            # Plain scalar common factor cancels.
            ("scaled", "2*x", "2*y", "x", "y"),
            # Coprime sides are left as-is.
            ("coprime", "x", "y", "x", "y"),
            # Higher-order grouped width: 16*floor(V/8) stays symbolic.
            (
                "grouped_width",
                "a*b*16*floor(V/8)",
                "a*b*w",
                "16*floor(V/8)",
                "w",
            ),
        ]
    )
    def test_reduce_common_factor(self, _name, a, b, exp_a, exp_b):
        a_expr = _symbolic_shapes.parse_symbolic_expression(a)
        b_expr = _symbolic_shapes.parse_symbolic_expression(b)
        red_a, red_b = _constraints._reduce_common_factor(a_expr, b_expr)
        self.assertEqual(red_a, _symbolic_shapes.parse_symbolic_expression(exp_a))
        self.assertEqual(red_b, _symbolic_shapes.parse_symbolic_expression(exp_b))

    def test_reduction_drives_compound_replacement(self):
        # A reshape numel equality should yield the rewrite 2*floor(c/2) -> c.
        symbol_map, compound = _constraints._build_replacements(
            [("a*b*c", "2*a*b*floor(c/2)")], _anon
        )
        self.assertEqual(symbol_map, {})
        self.assertEqual(len(compound), 1)
        source, target = compound[0]
        self.assertEqual(source, _symbolic_shapes.parse_symbolic_expression("2*floor(c/2)"))
        self.assertEqual(target, _symbolic_shapes.parse_symbolic_expression("c"))

    def test_trivial_numel_equality_produces_no_replacement(self):
        # 2*floor(H/2) == 2*floor(H/2): nothing to rewrite -> H stays symbolic.
        symbol_map, compound = _constraints._build_replacements(
            [("2*floor(H/2)", "2*floor(H/2)")], _anon
        )
        self.assertEqual(symbol_map, {})
        self.assertEqual(compound, [])


class RecordReshapeNumelEqualitiesTest(unittest.TestCase):
    """Tests for record_reshape_numel_equalities (ISSUE B provenance)."""

    def _reshape_graph(
        self, in_dims: list | None, out_dims: list | None
    ) -> tuple[ir.Graph, ir.Node]:
        data = ir.Value(
            name="data",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=None if in_dims is None else ir.Shape(in_dims),
        )
        shape = ir.Value(name="shape", type=ir.TensorType(ir.DataType.INT64))
        node = ir.Node("", "Reshape", inputs=[data, shape], num_outputs=1)
        out = node.outputs[0]
        out.name = "out"
        out.type = ir.TensorType(ir.DataType.FLOAT)
        if out_dims is not None:
            out.shape = ir.Shape(out_dims)
        graph = ir.Graph([data], [out], nodes=[node], opset_imports={"": 21})
        return graph, node

    def test_records_split_divisibility(self):
        # [a,b,c] -> [a,b,2,floor(c/2)] records a*b*c == 2*a*b*floor(c/2).
        graph, _ = self._reshape_graph(["a", "b", "c"], ["a", "b", 2, "floor(c/2)"])
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        # After building replacements this must rewrite 2*floor(c/2) -> c.
        _, compound = _constraints._build_replacements(
            list(ctx.symbolic_equalities), ctx.is_generated_symbol
        )
        targets = {str(t) for _, t in compound}
        self.assertIn("c", targets)

    def test_trivial_reshape_records_nothing(self):
        # Element count identical on both sides -> no equality recorded.
        graph, _ = self._reshape_graph(["a", "b"], ["a", "b"])
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        self.assertEqual(list(ctx.symbolic_equalities), [])

    def test_unknown_dim_skips_recording(self):
        # An unknown (SymbolicDim(None)) dim makes the numel inexpressible.
        graph, node = self._reshape_graph(["a", "b"], None)
        node.outputs[0].shape = ir.Shape([ir.SymbolicDim(None)])
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        self.assertEqual(list(ctx.symbolic_equalities), [])

    def test_missing_shape_skips_recording(self):
        graph, _ = self._reshape_graph(None, ["a", "b"])
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        self.assertEqual(list(ctx.symbolic_equalities), [])

    def test_end_to_end_provenance_rewrites_inherited_floor(self):
        # A value declared [a,b,2*floor(c/2)] adopts c once the reshape numel
        # equality supplies the divisibility provenance.
        graph, _node = self._reshape_graph(["a", "b", "c"], ["a", "b", 2, "floor(c/2)"])
        # A second value that carries the flattened 2*floor(c/2) form.
        flat = ir.Value(
            name="flat",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["a", "b", "2*floor(c/2)"]),
        )
        graph.outputs.append(flat)
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        _constraints.propagate_symbolic_constraints(ctx, graph)
        self.assertEqual(flat.shape[-1], _sym("c"))

    def test_divisibility_rewrite_applies_wherever_symbol_appears(self):
        # A reshape [a,b,c] -> [a,b,2,floor(c/2)] proves c is even, i.e.
        # c == 2*floor(c/2).  That equality holds globally for the symbol c, so
        # a *separate* value shaped [2*floor(c/2)] must ALSO simplify to [c],
        # regardless of its total element count.
        graph, _node = self._reshape_graph(["a", "b", "c"], ["a", "b", 2, "floor(c/2)"])
        # Inherited from the reshape (same numel form).
        inherited = ir.Value(
            name="inherited",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["a", "b", "2*floor(c/2)"]),
        )
        # A separate value with a different numel but the same proven-even c.
        separate = ir.Value(
            name="separate",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["2*floor(c/2)"]),
        )
        graph.outputs.extend([inherited, separate])
        ctx = _context.ShapeInferenceContext()
        _constraints.record_reshape_numel_equalities(ctx, graph)
        _constraints.propagate_symbolic_constraints(ctx, graph)
        self.assertEqual(inherited.shape[-1], _sym("c"))
        self.assertEqual(separate.shape[0], _sym("c"))


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
        self.assertEqual(_constraints._canonical_name(names, _anon), expected)


class PropagateSymbolicConstraintsTest(unittest.TestCase):
    def test_no_equalities_is_noop(self):
        graph = _identity_graph({"x": ["_d0"], "y": ["_d0"]})
        ctx = _context.ShapeInferenceContext()
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    def test_equalities_without_renamable_symbols_is_noop(self):
        # Both sides declared: no generated symbol to rename, so no change.
        graph = _identity_graph({"x": ["A"], "y": ["A"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("A", "B")
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(graph.inputs[0].shape[0], _sym("A"))

    def test_leaf_rename_across_graph(self):
        graph = _identity_graph({"x": ["N", "_d0"], "y": ["N", "_d0"]})
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "TopK_k")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        for value in list(graph.inputs) + list(graph.outputs):
            self.assertEqual(value.shape[-1], _sym("TopK_k"))

    def test_transitive_rename(self):
        # _d0 == _d1 and _d1 == N collapse both anon symbols to N.
        graph = _identity_graph({"x": ["_d0"], "y": ["_d1"]})
        ctx = _generated_ctx("_d0", "_d1")
        ctx.add_symbolic_equality("_d0", "_d1")
        ctx.add_symbolic_equality("_d1", "N")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], _sym("N"))
        self.assertEqual(next(iter(graph.outputs)).shape[0], _sym("N"))

    def test_scaled_leaf_equality_rewrites_compound(self):
        graph = _identity_graph({"x": ["_d0"], "y": ["2*_d0"]})
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("2*_d0", "2*dnz")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], _sym("dnz"))
        self.assertEqual(next(iter(graph.outputs)).shape[0], _sym("2*dnz"))

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
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "dnz")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        out = next(iter(graph.outputs)).shape
        self.assertEqual(out[0], 4)
        self.assertEqual(out[1], _sym("M"))
        self.assertEqual(out[2], _sym("dnz"))

    def test_unknown_shape_is_skipped(self):
        # A value with no shape must not raise and stays unshaped.
        graph = _identity_graph({"x": ["_d0"], "y": None})
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "N")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertIsNone(next(iter(graph.outputs)).shape)
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

        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "B1")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        self.assertEqual(y.shape[0], _sym("B1"))
        self.assertEqual(body_out.shape[0], _sym("B1"))


class AuthorDeclaredPlaceholderTest(unittest.TestCase):
    """Regression tests for identity-based (not spelling-based) rename eligibility.

    A model author may legitimately declare a symbol literally named ``_d0``.
    Because eligibility is decided by the exact set of names THIS context minted
    (``ShapeInferenceContext.is_generated_symbol``) and NOT by matching the
    ``_dN`` spelling, such author-declared symbols must never be renamed.
    """

    def test_author_declared_underscore_d0_not_renamed(self):
        # Model authors `_d0` on its input; an Identity forwards it to output `K`.
        # An equality _d0 == K exists, but since the context never minted `_d0`
        # it must be treated as authoritative and left untouched.
        graph = _identity_graph({"_d0_in": ["_d0"], "K_out": ["_d0"]})
        ctx = _context.ShapeInferenceContext()  # nothing minted
        ctx.add_symbolic_equality("_d0", "K")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertFalse(changed)
        self.assertEqual(graph.inputs[0].shape[0], _sym("_d0"))
        self.assertEqual(next(iter(graph.outputs)).shape[0], _sym("_d0"))

    def test_minted_underscore_d0_is_renamed(self):
        # The mirror image: when the SAME spelling was actually minted by this
        # context, it IS eligible and gets renamed to the declared anchor.
        graph = _identity_graph({"x": ["_d0"], "y": ["_d0"]})
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "K")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        self.assertEqual(graph.inputs[0].shape[0], _sym("K"))

    def test_mixed_author_and_minted_only_minted_renamed(self):
        # `_d0` authored (input), `_d1` minted; only `_d1` should be renamed.
        graph = _identity_graph({"x": ["_d0", "_d1"], "y": ["_d0", "_d1"]})
        ctx = _generated_ctx("_d1")
        ctx.add_symbolic_equality("_d0", "K")
        ctx.add_symbolic_equality("_d1", "M")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        out = next(iter(graph.outputs)).shape
        self.assertEqual(out[0], _sym("_d0"))  # authored, preserved
        self.assertEqual(out[1], _sym("M"))  # minted, renamed


if __name__ == "__main__":
    unittest.main()
