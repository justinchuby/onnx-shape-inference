# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the anchor/constraint propagation pass (_constraints.py)."""

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnx_shape_inference import _constraints, _context


def _graph_with_shapes(shapes: dict[str, list]) -> ir.Graph:
    """Build a trivial single-node graph whose values carry *shapes*."""
    values = {
        name: ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(dims),
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


class PropagateSymbolicConstraintsTest(unittest.TestCase):
    def test_no_equalities_is_noop(self):
        graph = _graph_with_shapes({"x": ["_d0"], "y": ["_d0"]})
        ctx = _context.ShapeInferenceContext()
        self.assertFalse(_constraints.propagate_symbolic_constraints(ctx, graph))

    def test_leaf_rename_across_graph(self):
        # x: [_d0], y: [_d0]; anchor equality _d0 == TopK_k renames both.
        graph = _graph_with_shapes({"x": ["N", "_d0"], "y": ["N", "_d0"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("_d0", "TopK_k")

        changed = _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertTrue(changed)
        for value in list(graph.inputs) + list(graph.outputs):
            self.assertEqual(value.shape[-1], ir.SymbolicDim("TopK_k"))

    def test_scaled_leaf_equality(self):
        # 2*_d0 == 2*dnz should imply _d0 == dnz, rewriting compound occurrence.
        graph = _graph_with_shapes({"x": ["_d0"], "y": ["2*_d0"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("2*_d0", "2*dnz")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], ir.SymbolicDim("dnz"))
        self.assertEqual(list(graph.outputs)[0].shape[0], ir.SymbolicDim("2*dnz"))

    def test_compound_replacement(self):
        # past_seq + seq declared as total_seq: compound substitution.
        graph = _graph_with_shapes(
            {"x": ["batch", "past_seq + seq"], "y": ["batch", "past_seq + seq"]}
        )
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("past_seq + seq", "total_seq")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        for value in list(graph.inputs) + list(graph.outputs):
            self.assertEqual(value.shape[-1], ir.SymbolicDim("total_seq"))

    def test_declared_names_never_renamed(self):
        # An equality between two declared names must not rewrite either.
        graph = _graph_with_shapes({"x": ["A"], "y": ["A"]})
        ctx = _context.ShapeInferenceContext()
        ctx.add_symbolic_equality("A", "B")

        _constraints.propagate_symbolic_constraints(ctx, graph)

        self.assertEqual(graph.inputs[0].shape[0], ir.SymbolicDim("A"))


if __name__ == "__main__":
    unittest.main()
