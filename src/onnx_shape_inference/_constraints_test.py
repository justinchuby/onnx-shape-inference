# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the anchor/constraint propagation pass (_constraints.py)."""

from __future__ import annotations

import unittest
from unittest import mock

import onnx_ir as ir
import parameterized
import sympy

from onnx_shape_inference import _constraints, _context, _engine, _symbolic_shapes


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


class ConstraintRewriteMemoizationTest(unittest.TestCase):
    def test_deduplicates_rewrites_including_unchanged_and_unknown_dims(self):
        values = [
            ir.Value(
                name=f"x_{index}",
                shape=ir.Shape(["_d0", "_d0", 7, None, "M", _sym("7")]),
            )
            for index in range(64)
        ]
        graph = ir.Graph(values, [], nodes=[])
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "N")
        with mock.patch.object(
            _constraints, "_rewrite_dim", wraps=_constraints._rewrite_dim
        ) as rewrite:
            self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))

        # Each exact type/value is evaluated once, including cached None results
        # and the new anchor encountered on the second (unchanged) sweep.
        self.assertEqual(rewrite.call_count, 6)
        keys = [
            (
                type(call.args[0]),
                call.args[0] if isinstance(call.args[0], int) else call.args[0].value,
            )
            for call in rewrite.call_args_list
        ]
        self.assertEqual(len(set(keys)), 6)
        self.assertIn((int, 7), keys)
        self.assertIn((ir.SymbolicDim, "7"), keys)
        for value in values:
            self.assertEqual(value.shape, ir.Shape(["N", "N", 7, None, "M", _sym("7")]))
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    def test_exact_expression_text_is_not_conflated(self):
        graph = _identity_graph({"x": ["_d0", "N + 1", "1+N"], "y": ["_d0", "N + 1", "1+N"]})
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "anchor")
        with mock.patch.object(
            _constraints, "_rewrite_dim", wraps=_constraints._rewrite_dim
        ) as rewrite:
            self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(rewrite.call_count, 4)
        self.assertEqual(graph.inputs[0].shape, ir.Shape(["anchor", "N + 1", "1+N"]))

    def test_cache_does_not_survive_changed_equalities_on_same_context(self):
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "N")
        graph = _identity_graph({"x": ["_d0"], "y": ["_d0"]})
        self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(graph.inputs[0].shape, ir.Shape(["N"]))

        ctx.add_symbolic_equality("_d0", "A")
        fresh = _identity_graph({"x": ["_d0"], "y": ["_d0"]})
        self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, fresh))
        self.assertEqual(fresh.inputs[0].shape, ir.Shape(["A"]))

    def test_cache_does_not_cross_generated_identity_contexts(self):
        for generated, expected in [(("_d0", "_d1"), ["N", "M"]), (("_d1",), ["_d0", "M"])]:
            with self.subTest(generated=generated):
                ctx = _generated_ctx(*generated)
                ctx.add_symbolic_equality("_d0", "N")
                ctx.add_symbolic_equality("_d1", "M")
                graph = _identity_graph({"x": ["_d0", "_d1"], "y": ["_d0", "_d1"]})
                self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
                self.assertEqual(graph.inputs[0].shape, ir.Shape(expected))
                self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    @parameterized.parameterized.expand(
        [
            ("short_first", [("a+b", "short"), ("a+b+c", "long")], "c + short"),
            ("long_first", [("a+b+c", "long"), ("a+b", "short")], "long"),
        ]
    )
    def test_leaf_then_overlapping_compound_order(self, _name, equalities, expected):
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "a")
        for lhs, rhs in equalities:
            ctx.add_symbolic_equality(lhs, rhs)
        graph = _identity_graph({"x": ["_d0+b+c"] * 8, "y": ["_d0+b+c"] * 8})
        self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(graph.inputs[0].shape, ir.Shape([expected] * 8))
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    def test_repeated_dims_across_nested_graph_and_unknown_metadata(self):
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        body_value = ir.Value(
            name="body_value", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["_d0"])
        )
        body = ir.Graph([], [body_value], nodes=[], name="body")
        node = ir.Node(
            "",
            "If",
            [cond],
            num_outputs=1,
            attributes=[ir.Attr("then_branch", ir.AttributeType.GRAPH, body)],
        )
        node.outputs[0].shape = ir.Shape(["_d0"])
        unshaped = ir.Value(name="unshaped")
        unknown_shape = ir.Shape([None, "M"], denotations=["UNKNOWN", "CHANNEL"], frozen=True)
        unknown_type = ir.TensorType(ir.DataType.FLOAT, denotation="IMAGE")
        unknown = ir.Value(name="unknown", shape=unknown_shape, type=unknown_type)
        graph = ir.Graph([cond, unshaped, unknown], [node.outputs[0]], nodes=[node])
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "N")
        equalities = tuple(ctx.symbolic_equalities)
        generated = ctx.generated_dim_names

        self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(body_value.shape, ir.Shape(["N"]))
        self.assertEqual(node.outputs[0].shape, ir.Shape(["N"]))
        self.assertIsNone(unshaped.shape)
        self.assertIsNone(unshaped.type)
        self.assertIs(unknown.shape, unknown_shape)
        self.assertIs(unknown.type, unknown_type)
        self.assertEqual(unknown.shape.get_denotation(0), "UNKNOWN")
        self.assertEqual(unknown.shape.get_denotation(1), "CHANNEL")
        self.assertEqual(unknown.type.denotation, "IMAGE")
        self.assertTrue(unknown.shape.frozen)
        self.assertEqual(tuple(ctx.symbolic_equalities), equalities)
        self.assertEqual(ctx.generated_dim_names, generated)

    @parameterized.parameterized.expand(
        [("with_provenance", True, "c"), ("without_provenance", False, "2*floor(c/2)")]
    )
    def test_repeated_divisibility_needs_provenance(self, _name, provenance, expected):
        ctx = _generated_ctx("_d0")
        ctx.add_symbolic_equality("_d0", "N")
        if provenance:
            ctx.add_symbolic_equality("a*b*c", "2*a*b*floor(c/2)")
        graph = _identity_graph({"x": ["2*floor(c/2)"] * 8, "y": ["2*floor(c/2)"] * 8})
        self.assertEqual(_constraints.propagate_symbolic_constraints(ctx, graph), provenance)
        self.assertEqual(graph.inputs[0].shape, ir.Shape([expected] * 8))
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    @parameterized.parameterized.expand(
        [
            (policy, adopt)
            for policy in ("skip", "override", "refine", "strict")
            for adopt in (False, True)
        ]
    )
    def test_public_policy_and_adoption_flags(self, policy, adopt):
        x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 5]))
        nz = ir.Node("", "NonZero", [x], num_outputs=1)
        identities = [
            ir.Node("", "Identity", [nz.outputs[0]], num_outputs=1) for _ in range(4)
        ]
        for index, node in enumerate(identities):
            node.outputs[0].name = f"Y_{index}"
            node.outputs[0].type = ir.TensorType(ir.DataType.INT64)
            node.outputs[0].shape = ir.Shape([2, "dnz"])
        model = ir.Model(
            ir.Graph(
                [x],
                [node.outputs[0] for node in identities],
                nodes=[nz, *identities],
                opset_imports={"": 21},
            ),
            ir_version=10,
        )
        _engine.infer_symbolic_shapes(model, policy=policy, adopt_declared_symbols=adopt)
        actual = nz.outputs[0].shape[1]
        # Skip/override do not record the anchor equality in this fixture.
        if adopt and policy in ("refine", "strict"):
            self.assertEqual(actual, _sym("dnz"))
        else:
            self.assertNotEqual(actual, _sym("dnz"))
            self.assertIsInstance(actual, ir.SymbolicDim)


class IndependentLeafSubstitutionTest(unittest.TestCase):
    def test_avoids_reprocessing_full_dictionary_for_each_dimension(self):
        class CountedDict(dict):
            items_calls = 0

            def items(self):
                self.items_calls += 1
                return super().items()

        replacements = CountedDict(
            {
                _symbolic_shapes.parse_symbolic_expression(
                    f"unused_{i}"
                ): _symbolic_shapes.parse_symbolic_expression(f"anchor_{i}")
                for i in range(16)
            }
        )
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("source", "anchor")
        values = [ir.Value(shape=ir.Shape([f"dimension_{i}"])) for i in range(8)]
        graph = ir.Graph(values, [], nodes=[])
        with mock.patch.object(
            _constraints, "_build_replacements", return_value=(replacements, [])
        ):
            self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))
        self.assertEqual(replacements.items_calls, 1)
        self.assertEqual(
            [v.shape for v in values], [ir.Shape([f"dimension_{i}"]) for i in range(8)]
        )

    def test_only_relevant_keys_reach_substitution(self):
        symbol = _symbolic_shapes.parse_symbolic_expression
        replacements = {symbol(f"unused_{i}"): symbol(f"anchor_{i}") for i in range(682)}
        replacements[symbol("source")] = symbol("anchor")
        values = [ir.Value(shape=ir.Shape([f"source + dimension_{i}"])) for i in range(8)]
        graph = ir.Graph(values, [], nodes=[])
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("source", "anchor")
        actual_key_counts = []
        original = _constraints._relevant_symbol_map

        def record(expr, mapping):
            selected = original(expr, mapping)
            actual_key_counts.append(len(selected))
            return selected

        with (
            mock.patch.object(
                _constraints, "_build_replacements", return_value=(replacements, [])
            ),
            mock.patch.object(_constraints, "_relevant_symbol_map", side_effect=record),
        ):
            self.assertTrue(_constraints.propagate_symbolic_constraints(ctx, graph))
        # Eight unique inputs and their eight unique rewritten outputs: not
        # 683 dictionary entries processed by SymPy on each helper invocation.
        self.assertEqual(actual_key_counts, [1] * 8 + [0] * 8)

    @parameterized.parameterized.expand(
        [
            ("leaf", "a"),
            ("sum", "a + z"),
            ("product", "a*b*z"),
            ("power", "a**b"),
            ("floor_max", "Max(a, z) + floor((a + 1)/8)"),
            ("min_mod", "Min(a, z) + floor(a/2) + Mod(b, 4)"),
            ("fraction", "a/(a - b)"),
            ("negative_power", "(a - b)**(-2)"),
        ]
    )
    def test_independent_renames_match_original_dictionary(self, _, text):
        symbol = _symbolic_shapes.parse_symbolic_expression
        mapping = {
            symbol("a"): symbol("q"),
            symbol("b"): symbol("q"),
            symbol("unused"): symbol("u"),
        }
        expr = symbol(text)
        prepared = _constraints._IndependentSymbolMap(mapping)
        selected = _constraints._relevant_symbol_map(expr, prepared)
        self.assertNotIn(symbol("unused"), selected)
        self.assertEqual(expr.subs(selected), expr.subs(mapping))

    def test_same_name_different_assumptions_and_noncommutative_targets(self):
        positive = sympy.Symbol("a", positive=True)
        negative = sympy.Symbol("a", negative=True)
        x, y = sympy.symbols("x y", commutative=False)
        mapping = {negative: y, positive: x}
        expr = positive**2 + positive
        selected = _constraints._relevant_symbol_map(
            expr, _constraints._IndependentSymbolMap(mapping)
        )
        self.assertEqual(selected, {positive: x})
        self.assertEqual(expr.subs(selected), expr.subs(mapping))

    def test_binding_and_unknown_expression_types_retain_full_map(self):
        a, b, c = sympy.symbols("a b c")
        mapping = _constraints._IndependentSymbolMap({a: c})

        class Opaque(sympy.Function):
            @property
            def free_symbols(self):
                return {b}

        for expr in (
            sympy.Integral(a, (a, 0, b)),
            sympy.Sum(a, (a, 1, b)),
            sympy.Lambda(a, a + b),
            sympy.Derivative(a * b, a, evaluate=False),
            sympy.Piecewise((a, b > 0), (b, True)),
            Opaque(a),
        ):
            with self.subTest(expression_type=type(expr).__name__):
                selected = _constraints._relevant_symbol_map(expr, mapping)
                self.assertIs(selected, mapping)
                self.assertEqual(expr.subs(selected), expr.subs(dict(mapping)))
        self.assertNotIn(a, Opaque(a).free_symbols)
        self.assertEqual(Opaque(a).subs(mapping), Opaque(c))

    @parameterized.parameterized.expand(
        [
            ("introduce_later", {"a": "b", "b": "c"}),
            ("introduce_earlier", {"b": "a", "a": "c"}),
            ("cycle", {"a": "b", "b": "a"}),
            ("identity", {"a": "a"}),
            ("constant_source", {"2": "c"}),
            ("infinite_target", {"a": "oo"}),
            ("compound_target", {"a": "b + c"}),
        ]
    )
    def test_broader_maps_keep_original_path_and_sweep_results(self, _, strings):
        parse = _symbolic_shapes.parse_symbolic_expression
        mapping = {parse(source): parse(target) for source, target in strings.items()}
        if _ == "infinite_target":
            mapping[parse("a")] = sympy.oo
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("source", "anchor")
        original_selector = _constraints._relevant_symbol_map
        optimized = []

        def inspect(expr, replacements):
            optimized.append(isinstance(replacements, _constraints._IndependentSymbolMap))
            selected = original_selector(expr, replacements)
            self.assertIs(selected, replacements)
            return selected

        def run(selector):
            graph = ir.Graph([ir.Value(shape=ir.Shape(["a + b", "b", "a**2"]))], [], nodes=[])
            with (
                mock.patch.object(
                    _constraints, "_build_replacements", return_value=(mapping, [])
                ),
                mock.patch.object(_constraints, "_relevant_symbol_map", side_effect=selector),
            ):
                modified = _constraints.propagate_symbolic_constraints(ctx, graph)
            return modified, [v.shape for v in graph.inputs]

        expected = run(lambda expr, replacements: replacements)
        self.assertEqual(run(inspect), expected)
        self.assertFalse(any(optimized))

    def test_compound_partial_matches_and_singularities_keep_original_order(self):
        parse = _symbolic_shapes.parse_symbolic_expression
        mapping = {parse("a"): parse("q"), parse("b"): parse("q"), parse("unused"): parse("u")}
        compounds = [
            (parse("2*q"), parse("u")),
            (parse("u + q"), parse("v")),
            (parse("q**2"), parse("w")),
            (sympy.Integer(2), parse("constant_anchor")),
        ]
        for text in ("3*a", "a + 2*b", "a**4", "a/(a-b)", "1/(a-b) + a"):
            with self.subTest(text=text):
                expected = _constraints._rewrite_dim(_sym(text), mapping, compounds)
                actual = _constraints._rewrite_dim(
                    _sym(text), _constraints._IndependentSymbolMap(mapping), compounds
                )
                self.assertEqual(actual, expected)

    def test_symbol_subclasses_and_invalid_dimensions_preserve_fallback(self):
        class CustomSymbol(sympy.Symbol):
            pass

        source = CustomSymbol("a")
        mapping = {source: sympy.Symbol("z")}
        self.assertIs(_constraints._relevant_symbol_map(source, mapping), mapping)
        prepared = _constraints._IndependentSymbolMap({sympy.Symbol("a"): sympy.Symbol("z")})
        self.assertIs(_constraints._relevant_symbol_map(source, prepared), prepared)
        for dim in (17, ir.Shape([None]).dims[0]):
            with self.subTest(dim_type=type(dim)):
                self.assertEqual(
                    _constraints._rewrite_dim(dim, mapping, []),
                    _constraints._rewrite_dim(dim, prepared, []),
                )
        with self.assertRaises(ValueError) as baseline:
            _constraints._rewrite_dim(_sym("unsupported("), mapping, [])
        with self.assertRaises(ValueError) as candidate:
            _constraints._rewrite_dim(_sym("unsupported("), prepared, [])
        self.assertEqual(str(baseline.exception), str(candidate.exception))


if __name__ == "__main__":
    unittest.main()
