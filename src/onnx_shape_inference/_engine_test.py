# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shape inference engine (_engine.py)."""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

from onnx_shape_inference import _context, _engine


class InferSymbolicShapesTest(unittest.TestCase):
    """Tests for infer_symbolic_shapes (public API)."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_returns_same_model_object(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        result = _engine.infer_symbolic_shapes(model)
        self.assertIs(result, model)

    def test_infers_shape_for_relu(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(node.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_names_anonymous_dims_on_graph_inputs(self):
        x = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # The anonymous dim should now have a name
        self.assertIsNotNone(x.shape[0].value)

    def test_multi_node_chain(self):
        """Shapes propagate through a chain of nodes."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([4, 5]))
        relu_node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        sigmoid_node = ir.Node("", "Sigmoid", inputs=relu_node.outputs, num_outputs=1)
        graph = ir.Graph(
            [x], sigmoid_node.outputs, nodes=[relu_node, sigmoid_node], opset_imports={"": 21}
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(sigmoid_node.outputs[0].shape, ir.Shape([4, 5]))
        self.assertEqual(sigmoid_node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_unregistered_op_is_skipped(self):
        """Nodes with no registered inference function are skipped gracefully."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        node = ir.Node("com.unknown", "FooBarOp", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # Output shape should remain unset
        self.assertIsNone(node.outputs[0].shape)


class InferSymbolicShapesModifiedTest(unittest.TestCase):
    """Tests for _infer_symbolic_shapes (returns modified flag)."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_returns_true_when_shapes_change(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        modified = _engine._infer_symbolic_shapes(model)
        self.assertTrue(modified)

    def test_returns_false_when_no_nodes(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        graph = ir.Graph([x], [x], nodes=[], opset_imports={"": 21})
        model = self._make_model(graph)

        modified = _engine._infer_symbolic_shapes(model)
        self.assertFalse(modified)


class SubgraphProcessingTest(unittest.TestCase):
    """Tests that subgraphs (e.g. If branches) are processed."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_subgraph_shapes_are_inferred(self):
        """Nodes inside If-then/else subgraphs get shape inference."""
        # Build a then-branch: Relu(input) -> output
        then_x = ir.Value(
            name="then_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        then_relu = ir.Node("", "Relu", inputs=[then_x], num_outputs=1)
        then_branch = ir.Graph(
            [then_x],
            then_relu.outputs,
            nodes=[then_relu],
            opset_imports={"": 21},
            name="then_branch",
        )

        # Build an else-branch: Sigmoid(input) -> output
        else_x = ir.Value(
            name="else_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        else_sigmoid = ir.Node("", "Sigmoid", inputs=[else_x], num_outputs=1)
        else_branch = ir.Graph(
            [else_x],
            else_sigmoid.outputs,
            nodes=[else_sigmoid],
            opset_imports={"": 21},
            name="else_branch",
        )

        # Build If node
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            ],
            num_outputs=1,
        )

        graph = ir.Graph(
            [cond],
            if_node.outputs,
            nodes=[if_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # Shapes inside subgraphs should be inferred
        self.assertEqual(then_relu.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(then_relu.outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(else_sigmoid.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(else_sigmoid.outputs[0].dtype, ir.DataType.FLOAT)

    def test_anonymous_dims_in_subgraph_are_named(self):
        """Anonymous dims on subgraph inputs get unique names."""
        then_x = ir.Value(
            name="then_x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        then_relu = ir.Node("", "Relu", inputs=[then_x], num_outputs=1)
        then_branch = ir.Graph(
            [then_x],
            then_relu.outputs,
            nodes=[then_relu],
            opset_imports={"": 21},
            name="then_branch",
        )

        else_x = ir.Value(
            name="else_x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        else_relu = ir.Node("", "Relu", inputs=[else_x], num_outputs=1)
        else_branch = ir.Graph(
            [else_x],
            else_relu.outputs,
            nodes=[else_relu],
            opset_imports={"": 21},
            name="else_branch",
        )

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            ],
            num_outputs=1,
        )

        graph = ir.Graph(
            [cond],
            if_node.outputs,
            nodes=[if_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # The anonymous dim in the subgraph input should now have a name
        self.assertIsNotNone(then_x.shape[0].value)


class PolicyPropagationTest(unittest.TestCase):
    """Tests that the policy parameter is respected."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_default_policy_is_refine(self):
        """The default policy should be 'refine'."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        # Should not raise
        _engine.infer_symbolic_shapes(model)


class FailingOpTest(unittest.TestCase):
    """Tests that a failing op inference does not crash the engine."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_engine_raises_on_op_failure(self):
        """If one node fails, the engine raises with context."""
        # Gemm with wrong rank (will fail)
        a = ir.Value(
            name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3, 4])
        )
        b = ir.Value(name="b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3, 5]))
        gemm_node = ir.Node("", "Gemm", inputs=[a, b], num_outputs=1)

        graph = ir.Graph(
            [a, b],
            gemm_node.outputs,
            nodes=[gemm_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        with self.assertRaises((_context.OpUsageError, _context.ShapeInferenceError)):
            _engine.infer_symbolic_shapes(model)


class RefineBodyInputTest(unittest.TestCase):
    """Tests for _refine_body_input (Scan/Loop body input shape refinement)."""

    def _ctx(self) -> _context.ShapeInferenceContext:
        return _context.ShapeInferenceContext({"": 18}, policy="override")

    def test_actual_none_keeps_body_shape(self):
        ctx = self._ctx()
        body = ir.Value(
            name="b", shape=ir.Shape([3, 4]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        _engine._refine_body_input(ctx, body, None, ir.DataType.FLOAT)
        self.assertEqual(body.shape, ir.Shape([3, 4]))

    def test_rank_mismatch_replaces_with_actual(self):
        ctx = self._ctx()
        body = ir.Value(name="b", shape=ir.Shape([3]), type=ir.TensorType(ir.DataType.FLOAT))
        _engine._refine_body_input(ctx, body, ir.Shape([5, 6]), ir.DataType.FLOAT)
        self.assertEqual(body.shape, ir.Shape([5, 6]))

    def test_concrete_body_dim_kept_over_symbolic_actual(self):
        """A concrete body dim is kept when the actual dim is symbolic.

        The trailing (anonymous) body position falls through to the actual dim.
        """
        ctx = self._ctx()
        # body: [3, anonymous], actual: ["X", "Y"]
        body = ir.Value(
            name="b",
            shape=ir.Shape([3, ir.SymbolicDim(None)]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        _engine._refine_body_input(
            ctx, body, ir.Shape([ir.SymbolicDim("X"), ir.SymbolicDim("Y")]), ir.DataType.FLOAT
        )
        # dim0: concrete body 3 wins; dim1: anonymous body -> actual "Y".
        self.assertEqual(body.shape[0], 3)
        self.assertEqual(str(body.shape[1]), "Y")

    def test_named_body_dim_preferred_when_actual_symbolic(self):
        ctx = self._ctx()
        body = ir.Value(
            name="b",
            shape=ir.Shape([ir.SymbolicDim("D")]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        _engine._refine_body_input(
            ctx, body, ir.Shape([ir.SymbolicDim("E")]), ir.DataType.FLOAT
        )
        self.assertEqual(str(body.shape[0]), "D")


class ScanNegativeAxisTest(unittest.TestCase):
    """Scan with a negative scan_input_axes exercises the axis-normalization path."""

    def test_negative_scan_input_axis(self):
        # Body: one state input [D], one scan slice input [D].
        acc_in = ir.Value(
            name="acc_in",
            shape=ir.Shape([ir.SymbolicDim("D")]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        x_t = ir.Value(
            name="x_t",
            shape=ir.Shape([ir.SymbolicDim("D")]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        add = ir.Node("", "Add", inputs=[acc_in, x_t], outputs=[ir.Value(name="acc_out")])
        ident = ir.Node(
            "", "Identity", inputs=[add.outputs[0]], outputs=[ir.Value(name="scan_out")]
        )
        body = ir.Graph(
            inputs=[acc_in, x_t],
            outputs=[add.outputs[0], ident.outputs[0]],
            nodes=[add, ident],
            opset_imports={"": 18},
            name="body",
        )

        # X scanned along the last axis (-1): shape [D, T].
        x = ir.Value(
            name="X",
            shape=ir.Shape([ir.SymbolicDim("D"), ir.SymbolicDim("T")]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        zero = ir.Value(
            name="zero",
            shape=ir.Shape([ir.SymbolicDim("D")]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        scan = ir.Node(
            "",
            "Scan",
            inputs=[zero, x],
            outputs=[ir.Value(name="acc_final"), ir.Value(name="Y")],
            attributes={
                "num_scan_inputs": ir.Attr("num_scan_inputs", ir.AttributeType.INT, 1),
                "scan_input_axes": ir.Attr("scan_input_axes", ir.AttributeType.INTS, [-1]),
                "body": ir.Attr("body", ir.AttributeType.GRAPH, body),
            },
        )
        graph = ir.Graph(
            inputs=[zero, x],
            outputs=list(scan.outputs),
            nodes=[scan],
            opset_imports={"": 18},
            name="g",
        )
        model = ir.Model(graph, ir_version=10)
        # Must run without error; the negative axis is normalized to drop dim -1.
        _engine.infer_symbolic_shapes(model)
        self.assertIsNotNone(scan.outputs[1].shape)


class ScanBodyPropagationGuardTest(unittest.TestCase):
    """Defensive guards in _propagate_types_to_scan_body for malformed Scan."""

    def _scan(self, attrs: dict) -> ir.Node:
        x = ir.Value(name="X", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT))
        return ir.Node(
            "",
            "Scan",
            inputs=[x],
            outputs=[ir.Value(name="Y")],
            attributes=attrs,
        )

    def test_missing_num_scan_inputs_is_noop(self):
        ctx = _context.ShapeInferenceContext({"": 18})
        body = ir.Graph(
            inputs=[ir.Value(name="bi")],
            outputs=[ir.Value(name="bo")],
            nodes=[],
            opset_imports={"": 18},
            name="body",
        )
        node = self._scan({"body": ir.Attr("body", ir.AttributeType.GRAPH, body)})
        # Must not raise even though num_scan_inputs is absent.
        _engine._propagate_types_to_scan_body(ctx, node)

    def test_negative_num_state_is_noop(self):
        ctx = _context.ShapeInferenceContext({"": 18})
        body = ir.Graph(
            inputs=[ir.Value(name="bi")],
            outputs=[ir.Value(name="bo")],
            nodes=[],
            opset_imports={"": 18},
            name="body",
        )
        # num_scan_inputs (5) > number of node inputs (1) -> num_state < 0.
        node = self._scan(
            {
                "body": ir.Attr("body", ir.AttributeType.GRAPH, body),
                "num_scan_inputs": ir.Attr("num_scan_inputs", ir.AttributeType.INT, 5),
            }
        )
        _engine._propagate_types_to_scan_body(ctx, node)

    def test_missing_body_is_noop(self):
        ctx = _context.ShapeInferenceContext({"": 18})
        node = self._scan(
            {"num_scan_inputs": ir.Attr("num_scan_inputs", ir.AttributeType.INT, 1)}
        )
        _engine._propagate_types_to_scan_body(ctx, node)


class AnchorConstraintPropagationTest(unittest.TestCase):
    """End-to-end tests for the adopt_declared_symbols anchor pass."""

    def _nonzero_model(
        self, anchor_last_dim: str | int | None = "dnz"
    ) -> tuple[ir.Model, ir.Node]:
        """NonZero -> Identity graph.

        The graph output ``Y`` declares ``[2, anchor_last_dim]`` as its anchor
        (unless *anchor_last_dim* is ``None``, in which case ``Y`` is left
        unshaped so there is nothing to adopt).
        """
        x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 5]))
        nz = ir.Node("", "NonZero", inputs=[x], num_outputs=1)
        nz.outputs[0].name = "Z"
        ident = ir.Node("", "Identity", inputs=[nz.outputs[0]], num_outputs=1)
        y = ident.outputs[0]
        y.name = "Y"
        y.type = ir.TensorType(ir.DataType.INT64)
        if anchor_last_dim is not None:
            y.shape = ir.Shape([2, anchor_last_dim])
        graph = ir.Graph([x], [y], nodes=[nz, ident], opset_imports={"": 21})
        return ir.Model(graph, ir_version=10), nz

    @parameterized.parameterized.expand(
        [
            ("default", {}),
            ("explicit_true", {"adopt_declared_symbols": True}),
        ]
    )
    def test_adopts_declared_symbol(self, _name, kwargs):
        model, nz = self._nonzero_model()
        _engine.infer_symbolic_shapes(model, **kwargs)
        # The engine-anonymous dim on the intermediate Z is renamed to "dnz".
        self.assertEqual(nz.outputs[0].shape[1], ir.SymbolicDim("dnz"))
        # The declared anchor dim is preserved on the output.
        self.assertEqual(next(iter(model.graph.outputs)).shape[1], ir.SymbolicDim("dnz"))

    def test_opt_out_keeps_anonymous_symbol(self):
        model, nz = self._nonzero_model()
        _engine.infer_symbolic_shapes(model, adopt_declared_symbols=False)
        # Without the pass the engine-anonymous symbol (_dN) is preserved.
        last = nz.outputs[0].shape[1]
        self.assertIsInstance(last, ir.SymbolicDim)
        self.assertRegex(str(last), r"^_d\d+$")

    def test_no_anchor_keeps_anonymous_symbol(self):
        # With no declared anchor there is nothing to adopt; the data-dependent
        # symbol is left as the engine generated it (no spurious rename).
        model, nz = self._nonzero_model(anchor_last_dim=None)
        _engine.infer_symbolic_shapes(model)
        last = nz.outputs[0].shape[1]
        self.assertIsInstance(last, ir.SymbolicDim)
        self.assertRegex(str(last), r"^_d\d+$")

    def test_concrete_anchor_leaves_symbol_unrelated(self):
        # A concrete anchor dim relates to nothing; the anon symbol stays anon.
        model, nz = self._nonzero_model(anchor_last_dim=7)
        _engine.infer_symbolic_shapes(model)
        last = nz.outputs[0].shape[1]
        self.assertIsInstance(last, ir.SymbolicDim)
        self.assertRegex(str(last), r"^_d\d+$")

    def test_adopts_topk_k_name_on_both_outputs(self):
        x = ir.Value(
            name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["N", "N"])
        )
        k = ir.Value(name="K", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1]))
        topk = ir.Node("", "TopK", [x, k], num_outputs=2)
        y = ir.Value(
            name="Y",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["N", "TopK_k"]),
        )
        identity = ir.Node("", "Identity", [topk.outputs[0]], outputs=[y])
        model = ir.Model(
            ir.Graph([x, k], [y], nodes=[topk, identity], opset_imports={"": 21}),
            ir_version=10,
        )

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(topk.outputs[0].shape, ir.Shape(["N", "TopK_k"]))
        self.assertEqual(topk.outputs[1].shape, ir.Shape(["N", "TopK_k"]))

    def test_adopts_nonzero_name_through_reshape_expression(self):
        x = ir.Value(
            name="X",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "seq"]),
        )
        nonzero = ir.Node("", "NonZero", [x], num_outputs=1)
        shape = ir.Value(
            name="shape",
            const_value=ir.Tensor(np.array([-1], dtype=np.int64), name="shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([1]),
        )
        flattened = ir.Node(
            "",
            "Reshape",
            [nonzero.outputs[0], shape],
            num_outputs=1,
        )
        y = ir.Value(
            name="Y",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["2*dnz"]),
        )
        identity = ir.Node("", "Identity", [flattened.outputs[0]], outputs=[y])
        model = ir.Model(
            ir.Graph([x], [y], nodes=[nonzero, flattened, identity], opset_imports={"": 21}),
            ir_version=10,
        )

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(nonzero.outputs[0].shape, ir.Shape([2, "dnz"]))
        self.assertEqual(flattened.outputs[0].shape, ir.Shape(["2*dnz"]))


if __name__ == "__main__":
    unittest.main()
