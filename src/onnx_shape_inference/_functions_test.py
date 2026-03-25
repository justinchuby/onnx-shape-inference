# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ONNX local function shape inference."""

from __future__ import annotations

import json
import os
import unittest

import onnx_ir as ir

from onnx_shape_inference import _context, infer_symbolic_shapes

FLOAT = ir.DataType.FLOAT
FLOAT16 = ir.DataType.FLOAT16
INT64 = ir.DataType.INT64


# ---------------------------------------------------------------------------
# Helpers for building minimal models with local functions
# ---------------------------------------------------------------------------


def _make_value(name: str, shape=None, dtype=None) -> ir.Value:
    """Create an ir.Value with optional shape and dtype."""
    type_ = ir.TensorType(dtype) if dtype is not None else None
    shape_ = ir.Shape(shape) if shape is not None else None
    return ir.Value(name=name, shape=shape_, type=type_)


def _make_model(
    main_nodes: list[ir.Node],
    main_inputs: list[ir.Value],
    main_outputs: list[ir.Value],
    functions: list[ir.Function],
    *,
    main_opset: int = 20,
    extra_domain: str = "test",
) -> ir.Model:
    """Build a minimal ir.Model from pre-constructed pieces."""
    opset_imports: dict[str, int] = {"": main_opset}
    if extra_domain:
        opset_imports[extra_domain] = 1
    graph = ir.Graph(
        inputs=main_inputs,
        outputs=main_outputs,
        nodes=main_nodes,
        opset_imports=opset_imports,
        name="main",
    )
    return ir.Model(graph, ir_version=10, functions=functions)


def _identity_function(domain: str = "test", name: str = "MyIdentity") -> ir.Function:
    """Build a function that wraps a single Identity op."""
    f_input = ir.Value(name="x")
    node = ir.Node("", "Identity", inputs=[f_input], outputs=[ir.Value(name="y")])
    graph = ir.Graph(
        inputs=[f_input],
        outputs=node.outputs,
        nodes=[node],
        opset_imports={"": 20},
        name=f"{name}_body",
    )
    return ir.Function(domain, name, "", graph=graph, attributes=[])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestFunctionShapeInference(unittest.TestCase):
    # 1. Basic identity wrapper -------------------------------------------
    def test_identity_wrapper(self):
        """Output shape and dtype match the input for an Identity-wrapping function."""
        func = _identity_function()
        inp = _make_value("input", shape=[3, 4], dtype=FLOAT)
        call = ir.Node("test", "MyIdentity", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertEqual(out.shape, ir.Shape([3, 4]))
        self.assertEqual(out.dtype, FLOAT)

    # 2. Type transformation (Cast) ----------------------------------------
    def test_cast_dtype_propagates(self):
        """A function wrapping Cast propagates the target dtype to the caller output."""
        f_input = ir.Value(name="x")
        cast_node = ir.Node(
            "",
            "Cast",
            inputs=[f_input],
            outputs=[ir.Value(name="y")],
            attributes={"to": ir.Attr("to", ir.AttributeType.INT, int(FLOAT))},
        )
        func_graph = ir.Graph(
            inputs=[f_input],
            outputs=cast_node.outputs,
            nodes=[cast_node],
            opset_imports={"": 20},
            name="cast_body",
        )
        func = ir.Function("test", "MyCast", "", graph=func_graph, attributes=[])

        inp = _make_value("input", shape=[2, 5], dtype=FLOAT16)
        call = ir.Node("test", "MyCast", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertEqual(out.shape, ir.Shape([2, 5]))
        self.assertEqual(out.dtype, FLOAT)

    # 3. Shape arithmetic (Transpose) --------------------------------------
    def test_transpose_shape_permuted(self):
        """A function wrapping Transpose produces correctly permuted output shape."""
        f_input = ir.Value(name="x")
        perm = [2, 0, 1]
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[f_input],
            outputs=[ir.Value(name="y")],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, perm)},
        )
        func_graph = ir.Graph(
            inputs=[f_input],
            outputs=transpose_node.outputs,
            nodes=[transpose_node],
            opset_imports={"": 20},
            name="transpose_body",
        )
        func = ir.Function("test", "MyTranspose", "", graph=func_graph, attributes=[])

        inp = _make_value("input", shape=[3, 4, 5], dtype=FLOAT)
        call = ir.Node("test", "MyTranspose", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        # perm [2,0,1] on shape [3,4,5] → [5,3,4]
        self.assertEqual(out.shape, ir.Shape([5, 3, 4]))
        self.assertEqual(out.dtype, FLOAT)

    # 4. Unknown input shape — graceful degradation -----------------------
    def test_unknown_input_shape_no_crash(self):
        """When input shape is None, output stays None; no exception is raised."""
        func = _identity_function()
        inp = _make_value("input", dtype=FLOAT)  # shape=None
        call = ir.Node("test", "MyIdentity", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertIsNone(out.shape)
        self.assertEqual(out.dtype, FLOAT)

    # 5. Multiple callers, different shapes --------------------------------
    def test_multiple_callers_different_shapes(self):
        """Same function called with two different input shapes produces independent outputs.

        This is the key correctness test: stale state from call 1 must not leak
        into call 2 (and vice versa).
        """
        func = _identity_function()

        inp1 = _make_value("input1", shape=[3, 4], dtype=FLOAT)
        inp2 = _make_value("input2", shape=[7, 8], dtype=FLOAT)
        call1 = ir.Node("test", "MyIdentity", inputs=[inp1], outputs=[ir.Value(name="out1")])
        call2 = ir.Node("test", "MyIdentity", inputs=[inp2], outputs=[ir.Value(name="out2")])
        model = _make_model(
            [call1, call2], [inp1, inp2], call1.outputs + call2.outputs, [func]
        )

        infer_symbolic_shapes(model, warn_on_missing=False)

        self.assertEqual(call1.outputs[0].shape, ir.Shape([3, 4]))
        self.assertEqual(call2.outputs[0].shape, ir.Shape([7, 8]))

    # 6. Multiple callers, same shapes — cache hit -------------------------
    def test_multiple_callers_same_shapes_cache_hit(self):
        """Repeated calls with identical shapes hit the cache and produce correct results."""
        func = _identity_function()

        inp1 = _make_value("input1", shape=[5, 6], dtype=FLOAT)
        inp2 = _make_value("input2", shape=[5, 6], dtype=FLOAT)
        call1 = ir.Node("test", "MyIdentity", inputs=[inp1], outputs=[ir.Value(name="out1")])
        call2 = ir.Node("test", "MyIdentity", inputs=[inp2], outputs=[ir.Value(name="out2")])
        model = _make_model(
            [call1, call2], [inp1, inp2], call1.outputs + call2.outputs, [func]
        )

        infer_symbolic_shapes(model, warn_on_missing=False)

        self.assertEqual(call1.outputs[0].shape, ir.Shape([5, 6]))
        self.assertEqual(call2.outputs[0].shape, ir.Shape([5, 6]))
        self.assertEqual(call1.outputs[0].dtype, FLOAT)
        self.assertEqual(call2.outputs[0].dtype, FLOAT)

    # 7. Nested functions --------------------------------------------------
    def test_nested_functions(self):
        """Function A calls function B; shapes propagate through both levels."""
        # Inner function B: Identity
        func_b = _identity_function(domain="test", name="FuncB")

        # Outer function A: calls FuncB
        fa_input = ir.Value(name="a_in")
        inner_call = ir.Node(
            "test", "FuncB", inputs=[fa_input], outputs=[ir.Value(name="a_out")]
        )
        func_a_graph = ir.Graph(
            inputs=[fa_input],
            outputs=inner_call.outputs,
            nodes=[inner_call],
            opset_imports={"": 20, "test": 1},
            name="func_a_body",
        )
        func_a = ir.Function("test", "FuncA", "", graph=func_a_graph, attributes=[])

        inp = _make_value("input", shape=[9, 3], dtype=FLOAT)
        call = ir.Node("test", "FuncA", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func_a, func_b])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertEqual(out.shape, ir.Shape([9, 3]))
        self.assertEqual(out.dtype, FLOAT)

    # 8. Recursion guard ---------------------------------------------------
    def test_recursion_guard_no_infinite_loop(self):
        """A self-referencing function body triggers a warning and gracefully produces None."""
        f_input = ir.Value(name="x")
        self_call = ir.Node(
            "test", "RecursiveFunc", inputs=[f_input], outputs=[ir.Value(name="y")]
        )
        func_graph = ir.Graph(
            inputs=[f_input],
            outputs=self_call.outputs,
            nodes=[self_call],
            opset_imports={"": 20, "test": 1},
            name="recursive_body",
        )
        func = ir.Function("test", "RecursiveFunc", "", graph=func_graph, attributes=[])

        inp = _make_value("input", shape=[2, 2], dtype=FLOAT)
        call = ir.Node("test", "RecursiveFunc", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        # Must not raise; output shape remains None because the recursion is blocked
        with self.assertLogs("onnx_shape_inference._functions", level="WARNING") as cm:
            infer_symbolic_shapes(model, warn_on_missing=False)

        self.assertIsNone(call.outputs[0].shape)
        self.assertTrue(
            any("Recursive" in msg for msg in cm.output),
            f"Expected recursion warning, got: {cm.output}",
        )

    # 9. Per-function opset ------------------------------------------------
    def test_per_function_opset(self):
        """Function with opset_imports={'': 24} dispatches body nodes using opset 24."""
        # Build a function that uses Cast (available in all recent opsets)
        # but with an explicit opset_imports={'': 24}
        f_input = ir.Value(name="x")
        cast_node = ir.Node(
            "",
            "Cast",
            inputs=[f_input],
            outputs=[ir.Value(name="y")],
            attributes={"to": ir.Attr("to", ir.AttributeType.INT, int(FLOAT))},
        )
        func_graph = ir.Graph(
            inputs=[f_input],
            outputs=cast_node.outputs,
            nodes=[cast_node],
            opset_imports={"": 24},  # different from model's opset 17
            name="cast_body_opset24",
        )
        func = ir.Function("test", "Cast24", "", graph=func_graph, attributes=[])

        inp = _make_value("input", shape=[4, 4], dtype=FLOAT16)
        call = ir.Node("test", "Cast24", inputs=[inp], outputs=[ir.Value(name="out")])
        graph = ir.Graph(
            inputs=[inp],
            outputs=call.outputs,
            nodes=[call],
            opset_imports={"": 17, "test": 1},  # model uses opset 17
            name="main",
        )
        model = ir.Model(graph, ir_version=10, functions=[func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertEqual(out.shape, ir.Shape([4, 4]))
        self.assertEqual(out.dtype, FLOAT)

    # 10. sym_data propagation ---------------------------------------------
    def test_sym_data_propagates_from_function_output(self):
        """SYM_DATA_KEY metadata flows from a function's output to the caller's output."""
        # Build function: Shape(x) → output is a 1-D int64 shape tensor
        f_input = ir.Value(name="x")
        shape_node = ir.Node(
            "",
            "Shape",
            inputs=[f_input],
            outputs=[ir.Value(name="shape_out")],
        )
        func_graph = ir.Graph(
            inputs=[f_input],
            outputs=shape_node.outputs,
            nodes=[shape_node],
            opset_imports={"": 20},
            name="shape_func_body",
        )
        func = ir.Function("test", "GetShape", "", graph=func_graph, attributes=[])

        inp = _make_value("input", shape=[3, 4, 5], dtype=FLOAT)
        call = ir.Node("test", "GetShape", inputs=[inp], outputs=[ir.Value(name="out")])
        model = _make_model([call], [inp], call.outputs, [func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        # Shape op on [3,4,5] produces a 1-D tensor [3] with sym_data [3,4,5]
        self.assertIn(_context.SYM_DATA_KEY, out.metadata_props)
        sym_data = json.loads(out.metadata_props[_context.SYM_DATA_KEY])
        self.assertEqual(sym_data, [3, 4, 5])

    # 11. Integration test -------------------------------------------------
    def test_integration_qwen35_2b(self):
        """infer_symbolic_shapes resolves all function-call outputs on qwen3.5-2B."""
        # Set ONNX_TEST_MODEL env var or place the model at the default path.
        # This test is skipped in CI if the file does not exist.
        model_path = os.environ.get(
            "ONNX_TEST_MODEL",
            "/home/justinchu/dev/mobius-human/output/qwen3.5-2B-f16/model.onnx",
        )
        if not os.path.exists(model_path):
            self.skipTest(f"Integration model not found: {model_path}")

        model = ir.load(model_path)
        infer_symbolic_shapes(model, warn_on_missing=False)

        func_keys = {(k[0], k[1]) for k in model.functions}
        unknown = [
            out
            for node in model.graph
            if (node.domain, node.op_type) in func_keys
            for out in node.outputs
            if out.shape is None
        ]
        self.assertEqual(
            len(unknown),
            0,
            f"{len(unknown)} function-call outputs still have shape=None after inference",
        )

        known = sum(1 for n in model.graph for o in n.outputs if o.shape is not None)
        self.assertGreaterEqual(
            known,
            1683,
            f"Expected ≥1683 known shapes, got {known}",
        )

    # 12. RefAttr substitution ---------------------------------------------
    def test_ref_attr_substitution(self):
        """A body node with a RefAttr receives the resolved value from the call site."""
        # Function: Gather(data, indices, axis=@my_axis)
        # where @my_axis is a RefAttr pointing to function attribute 'my_axis'.
        # Call site passes my_axis=1; axis=1 should select columns of a [3,4] input.
        f_data = ir.Value(name="data")
        f_indices = ir.Value(name="indices")

        gather_node = ir.Node(
            "",
            "Gather",
            inputs=[f_data, f_indices],
            outputs=[ir.Value(name="gathered")],
            attributes={
                "axis": ir.RefAttr("axis", "my_axis", ir.AttributeType.INT),
            },
        )
        func_graph = ir.Graph(
            inputs=[f_data, f_indices],
            outputs=gather_node.outputs,
            nodes=[gather_node],
            opset_imports={"": 20},
            name="gather_func_body",
        )
        # Function declares 'my_axis' with default 0
        func = ir.Function(
            "test",
            "GatherWithAxis",
            "",
            graph=func_graph,
            attributes=[ir.Attr("my_axis", ir.AttributeType.INT, 0)],
        )

        # Call site: data=[3,4], indices=[2], my_axis=1
        # Gather on axis=1: output shape = data[:axis] + indices.shape + data[axis+1:]
        #                              = [3] + [2] + [] = [3, 2]
        data_inp = _make_value("data_inp", shape=[3, 4], dtype=FLOAT)
        indices_inp = _make_value("indices_inp", shape=[2], dtype=INT64)
        call = ir.Node(
            "test",
            "GatherWithAxis",
            inputs=[data_inp, indices_inp],
            outputs=[ir.Value(name="out")],
            attributes={"my_axis": ir.Attr("my_axis", ir.AttributeType.INT, 1)},
        )
        graph = ir.Graph(
            inputs=[data_inp, indices_inp],
            outputs=call.outputs,
            nodes=[call],
            opset_imports={"": 20, "test": 1},
            name="main",
        )
        model = ir.Model(graph, ir_version=10, functions=[func])

        infer_symbolic_shapes(model, warn_on_missing=False)

        out = call.outputs[0]
        self.assertEqual(out.shape, ir.Shape([3, 2]), f"Expected [3,2], got {out.shape}")
        self.assertEqual(out.dtype, FLOAT)

    # 13. Double-run idempotency ------------------------------------------
    def test_double_run_idempotent(self):
        """Running infer_symbolic_shapes twice on the same model produces identical results.

        Verifies that the type aliasing bug (dtype setter mutating shared TensorType)
        and the subgraph save/restore are both correct: no caller input dtypes are
        corrupted on the second pass.
        """
        func = _identity_function()

        inp1 = _make_value("input1", shape=[3, 4], dtype=FLOAT)
        inp2 = _make_value("input2", shape=[7, 8], dtype=FLOAT)
        call1 = ir.Node("test", "MyIdentity", inputs=[inp1], outputs=[ir.Value(name="out1")])
        call2 = ir.Node("test", "MyIdentity", inputs=[inp2], outputs=[ir.Value(name="out2")])
        model = _make_model(
            [call1, call2], [inp1, inp2], call1.outputs + call2.outputs, [func]
        )

        infer_symbolic_shapes(model, warn_on_missing=False)
        shape1_run1 = call1.outputs[0].shape
        shape2_run1 = call2.outputs[0].shape
        inp1_dtype_run1 = inp1.dtype
        inp2_dtype_run1 = inp2.dtype

        # Second run — must not corrupt caller input dtypes
        infer_symbolic_shapes(model, warn_on_missing=False)

        self.assertEqual(call1.outputs[0].shape, shape1_run1)
        self.assertEqual(call2.outputs[0].shape, shape2_run1)
        # Caller input dtypes must be unchanged by the second run
        self.assertEqual(inp1.dtype, inp1_dtype_run1, "inp1 dtype was corrupted on second run")
        self.assertEqual(inp2.dtype, inp2_dtype_run1, "inp2 dtype was corrupted on second run")


if __name__ == "__main__":
    unittest.main()
