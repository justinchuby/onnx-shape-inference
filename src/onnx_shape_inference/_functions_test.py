# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ONNX local function shape inference."""

from __future__ import annotations

import json
import os
import pathlib
import unittest

import onnx
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


def _single_op_function(
    domain: str,
    name: str,
    op_type: str,
    *,
    body_attrs: dict[str, ir.Attr] | None = None,
    n_inputs: int = 1,
    opset: int = 20,
    func_attrs: list[ir.Attr] | None = None,
) -> ir.Function:
    """Build a function wrapping a single ONNX op.

    ``body_attrs`` are attributes on the inner op node (e.g. ``{"to": ...}`` for Cast).
    ``func_attrs`` are the function-level attribute declarations exposed to callers
    (used with ``ir.RefAttr`` substitution); most single-op wrappers don't need them.

    The key distinction: ``func_attrs`` declares the function's *formal attribute
    parameters* — its public interface as stored on ``ir.Function`` itself — while
    ``body_attrs`` supplies *actual attribute values* baked directly into the inner
    body node at construction time.
    """
    f_inputs = [ir.Value(name=f"x{i}") for i in range(n_inputs)]
    node = ir.Node(
        "",
        op_type,
        inputs=f_inputs,
        outputs=[ir.Value(name="y")],
        attributes=body_attrs or {},
    )
    graph = ir.Graph(
        inputs=f_inputs,
        outputs=node.outputs,
        nodes=[node],
        opset_imports={"": opset},
        name=f"{name}_body",
    )
    return ir.Function(domain, name, "", graph=graph, attributes=func_attrs or [])


def _identity_function(domain: str = "test", name: str = "MyIdentity") -> ir.Function:
    """Build a function that wraps a single Identity op."""
    return _single_op_function(domain, name, "Identity")


def _run_function_call(
    func: ir.Function,
    inputs: list[ir.Value],
    extra_functions: list[ir.Function] | None = None,
    extra_attrs: dict[str, ir.Attr] | None = None,
) -> list[ir.Value]:
    """Build a minimal model calling *func*, run shape inference, return output Values."""
    output_values = [ir.Value(name=f"out{i}") for i in range(len(func.outputs))]
    call = ir.Node(
        func.domain,
        func.name,
        inputs=inputs,
        outputs=output_values,
        attributes=extra_attrs or {},
    )
    all_functions = [func] + (extra_functions or [])
    model = _make_model([call], inputs, output_values, all_functions, extra_domain=func.domain)
    infer_symbolic_shapes(model, warn_on_missing=False)
    return output_values


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestFunctionShapeInference(unittest.TestCase):
    # 1. Basic identity wrapper -------------------------------------------
    def test_identity_wrapper(self):
        """Output shape and dtype match the input for an Identity-wrapping function."""
        inp = _make_value("x", shape=[3, 4], dtype=FLOAT)
        [out] = _run_function_call(_identity_function(), [inp])
        self.assertEqual(out.shape, ir.Shape([3, 4]))
        self.assertEqual(out.dtype, FLOAT)

    # 2. Type transformation (Cast) ----------------------------------------
    def test_cast_dtype_propagates(self):
        """A function wrapping Cast propagates the target dtype to the caller output."""
        func = _single_op_function(
            "test",
            "MyCast",
            "Cast",
            body_attrs={"to": ir.Attr("to", ir.AttributeType.INT, int(FLOAT))},
        )
        inp = _make_value("x", shape=[2, 5], dtype=FLOAT16)
        [out] = _run_function_call(func, [inp])
        self.assertEqual(out.shape, ir.Shape([2, 5]))
        self.assertEqual(out.dtype, FLOAT)

    # 3. Shape arithmetic (Transpose) --------------------------------------
    def test_transpose_shape_permuted(self):
        """A function wrapping Transpose produces correctly permuted output shape."""
        func = _single_op_function(
            "test",
            "MyTranspose",
            "Transpose",
            body_attrs={"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 0, 1])},
        )
        inp = _make_value("x", shape=[3, 4, 5], dtype=FLOAT)
        [out] = _run_function_call(func, [inp])
        # perm [2,0,1] on [3,4,5] → [5,3,4]
        self.assertEqual(out.shape, ir.Shape([5, 3, 4]))
        self.assertEqual(out.dtype, FLOAT)

    # 4. Unknown input shape — graceful degradation -----------------------
    def test_unknown_input_shape_no_crash(self):
        """When input shape is None, output stays None; no exception is raised."""
        inp = _make_value("x", dtype=FLOAT)  # shape=None
        [out] = _run_function_call(_identity_function(), [inp])
        self.assertIsNone(out.shape)
        self.assertEqual(out.dtype, FLOAT)

    # 5. Multiple callers, different shapes --------------------------------
    def test_multiple_callers_different_shapes(self):
        """Same function called twice with different shapes produces independent outputs.

        Stale state from call 1 must not leak into call 2 (and vice versa).
        """
        func = _identity_function()
        inp1 = _make_value("x1", shape=[3, 4], dtype=FLOAT)
        inp2 = _make_value("x2", shape=[7, 8], dtype=FLOAT)
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
        inp1 = _make_value("x1", shape=[5, 6], dtype=FLOAT)
        inp2 = _make_value("x2", shape=[5, 6], dtype=FLOAT)
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
        func_b = _identity_function(domain="test", name="FuncB")

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

        inp = _make_value("x", shape=[9, 3], dtype=FLOAT)
        [out] = _run_function_call(func_a, [inp], extra_functions=[func_b])
        self.assertEqual(out.shape, ir.Shape([9, 3]))
        self.assertEqual(out.dtype, FLOAT)

    # 8. Recursion guard ---------------------------------------------------
    def test_recursion_guard_no_infinite_loop(self):
        """A self-referencing function body triggers a warning and produces None output."""
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

        inp = _make_value("x", shape=[2, 2], dtype=FLOAT)
        with self.assertLogs("onnx_shape_inference._functions", level="WARNING") as cm:
            [out] = _run_function_call(func, [inp])
        self.assertIsNone(out.shape)
        self.assertTrue(
            any("Recursive" in msg for msg in cm.output),
            f"Expected recursion warning, got: {cm.output}",
        )

    # 9. Per-function opset ------------------------------------------------
    def test_per_function_opset(self):
        """Function with opset_imports={'': 24} dispatches body nodes using opset 24."""
        func = _single_op_function(
            "test",
            "Cast24",
            "Cast",
            body_attrs={"to": ir.Attr("to", ir.AttributeType.INT, int(FLOAT))},
            opset=24,
        )
        inp = _make_value("x", shape=[4, 4], dtype=FLOAT16)
        call = ir.Node("test", "Cast24", inputs=[inp], outputs=[ir.Value(name="out")])
        graph = ir.Graph(
            inputs=[inp],
            outputs=call.outputs,
            nodes=[call],
            opset_imports={"": 17, "test": 1},  # model uses opset 17, function uses 24
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
        func = _single_op_function("test", "GetShape", "Shape")
        inp = _make_value("x", shape=[3, 4, 5], dtype=FLOAT)
        [out] = _run_function_call(func, [inp])
        # Shape op on [3,4,5] produces a 1-D tensor [3] with sym_data [3,4,5]
        self.assertIn(_context.SYM_DATA_KEY, out.metadata_props)
        sym_data = json.loads(out.metadata_props[_context.SYM_DATA_KEY])
        self.assertEqual(sym_data, [3, 4, 5])

    # 11. Integration test -------------------------------------------------
    def test_integration_qwen35_2b(self):
        """infer_symbolic_shapes resolves all function-call outputs on qwen3.5-2B."""
        # Resolution order:
        # 1. ONNX_TEST_MODEL env var (CI / developer override)
        # 2. testdata/qwen3.5-2B-f16.onnx.pbtxt relative to the repo root
        _repo_root = pathlib.Path(__file__).parent.parent.parent
        _testdata_model = _repo_root / "testdata" / "qwen3.5-2B-f16.onnx.pbtxt"
        model_path = pathlib.Path(os.environ.get("ONNX_TEST_MODEL") or _testdata_model)
        if not model_path.exists():
            self.skipTest(
                f"Integration model not found at {model_path}. "
                "Set ONNX_TEST_MODEL to override."
            )

        model = ir.load(str(model_path))
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
        # Call site passes my_axis=1: Gather axis=1 on [3,4] with indices [2] → [3,2]
        f_data = ir.Value(name="data")
        f_indices = ir.Value(name="indices")
        gather_node = ir.Node(
            "",
            "Gather",
            inputs=[f_data, f_indices],
            outputs=[ir.Value(name="gathered")],
            attributes={"axis": ir.RefAttr("axis", "my_axis", ir.AttributeType.INT)},
        )
        func_graph = ir.Graph(
            inputs=[f_data, f_indices],
            outputs=gather_node.outputs,
            nodes=[gather_node],
            opset_imports={"": 20},
            name="gather_func_body",
        )
        func = ir.Function(
            "test",
            "GatherWithAxis",
            "",
            graph=func_graph,
            attributes=[ir.Attr("my_axis", ir.AttributeType.INT, 0)],  # default axis=0
        )

        data_inp = _make_value("data", shape=[3, 4], dtype=FLOAT)
        indices_inp = _make_value("indices", shape=[2], dtype=INT64)
        [out] = _run_function_call(
            func,
            [data_inp, indices_inp],
            extra_attrs={"my_axis": ir.Attr("my_axis", ir.AttributeType.INT, 1)},
        )
        self.assertEqual(out.shape, ir.Shape([3, 2]), f"Expected [3,2], got {out.shape}")
        self.assertEqual(out.dtype, FLOAT)

    # 13. Trailing optional inputs (fewer call-site inputs than function declares) -----
    def test_trailing_optional_inputs_omitted(self):
        """Call site may provide fewer inputs than the function declares.

        ONNX allows trailing function inputs to be optional.  The call site
        omits them (providing fewer inputs); this must not raise an error.
        The function body uses only the provided inputs, and the output shape
        must still be inferred correctly.
        """
        # Function: Add(x, y, z) → out   (z is unused; only x + y are consumed)
        f_x = ir.Value(name="x")
        f_y = ir.Value(name="y")
        f_z = ir.Value(name="z")  # trailing optional — not used in the body
        add_node = ir.Node("", "Add", inputs=[f_x, f_y], outputs=[ir.Value(name="out")])
        func_graph = ir.Graph(
            inputs=[f_x, f_y, f_z],
            outputs=add_node.outputs,
            nodes=[add_node],
            opset_imports={"": 20},
            name="myfunc_body",
        )
        func = ir.Function("test", "MyFunc", "", graph=func_graph, attributes=[])

        # Call site provides only 2 of the 3 declared inputs
        x_inp = _make_value("A", shape=[2, 3], dtype=FLOAT)
        y_inp = _make_value("B", shape=[2, 3], dtype=FLOAT)
        [out] = _run_function_call(func, [x_inp, y_inp])
        self.assertEqual(out.shape, ir.Shape([2, 3]))
        self.assertEqual(out.dtype, FLOAT)

    # 14. Double-run idempotency ------------------------------------------
    def test_double_run_idempotent(self):
        """Running infer_symbolic_shapes twice produces identical results with no dtype corruption.

        Verifies that the type aliasing fix (dtype setter mutating shared TensorType)
        holds across runs: caller input dtypes must not be corrupted on the second pass.
        """
        func = _identity_function()
        inp1 = _make_value("x1", shape=[3, 4], dtype=FLOAT)
        inp2 = _make_value("x2", shape=[7, 8], dtype=FLOAT)
        call1 = ir.Node("test", "MyIdentity", inputs=[inp1], outputs=[ir.Value(name="out1")])
        call2 = ir.Node("test", "MyIdentity", inputs=[inp2], outputs=[ir.Value(name="out2")])
        model = _make_model(
            [call1, call2], [inp1, inp2], call1.outputs + call2.outputs, [func]
        )

        infer_symbolic_shapes(model, warn_on_missing=False)
        shape1, shape2 = call1.outputs[0].shape, call2.outputs[0].shape

        infer_symbolic_shapes(model, warn_on_missing=False)

        self.assertEqual(call1.outputs[0].shape, shape1)
        self.assertEqual(call2.outputs[0].shape, shape2)
        self.assertEqual(call1.outputs[0].dtype, FLOAT)
        self.assertEqual(call2.outputs[0].dtype, FLOAT)
        self.assertEqual(inp1.dtype, FLOAT, "inp1 dtype was corrupted on second run")
        self.assertEqual(inp2.dtype, FLOAT, "inp2 dtype was corrupted on second run")


class OpSchemaFunctionExpansionTest(unittest.TestCase):
    """Inference for standard ops via their ONNX op-schema function bodies.

    Newer ONNX operators define their semantics as a (often context-dependent)
    function of more primitive ops.  When we have no hand-written rule for such
    an op, the engine materializes and runs shape inference on that function
    body, which lets us track new ops automatically.
    """

    def _build_attention(self, opset: int = 24):
        b, h, s, d = 2, 4, 8, 16
        helper = onnx.helper

        def vi(name):
            return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [b, h, s, d])

        node = helper.make_node("Attention", ["Q", "K", "V"], ["Y"])
        graph = helper.make_graph(
            [node],
            "g",
            [vi("Q"), vi("K"), vi("V")],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
        )
        proto = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
        )
        return ir.serde.deserialize_model(proto), (b, h, s, d)

    def _build_hardswish(self, opset: int = 22):
        helper = onnx.helper
        x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 4])
        node = helper.make_node("HardSwish", ["X"], ["Y"])
        graph = helper.make_graph(
            [node],
            "g",
            [x],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
        )
        proto = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
        )
        return ir.serde.deserialize_model(proto)

    def test_fallback_used_when_no_native_rule(self):
        """The op-schema fallback recovers the shape when no native rule exists.

        ``HardSwish`` has a native (elementwise) rule, so to exercise the
        *fallback* end to end we mask that rule (return ``None`` from the
        registry lookup) and confirm the engine materializes and runs the op's
        static function body to recover the output shape.
        """
        import unittest.mock as mock

        from onnx_shape_inference import _registry

        try:
            schema = onnx.defs.get_schema("HardSwish", max_inclusive_version=22, domain="")
        except Exception:
            self.skipTest("HardSwish op not available in this onnx build")
        if not schema.has_function:
            self.skipTest("HardSwish has no function body in this onnx build")

        model = self._build_hardswish()
        real_get = _registry.registry.get

        def fake_get(domain, op_type, version):
            if op_type == "HardSwish":
                return None  # pretend we have no native rule
            return real_get(domain, op_type, version)

        with mock.patch.object(_registry.registry, "get", side_effect=fake_get):
            infer_symbolic_shapes(model, warn_on_missing=False)

        out = model.graph.outputs[0]
        self.assertEqual(out.dtype, FLOAT)
        self.assertEqual(out.shape, ir.Shape([3, 4]))

    def test_attention_native_rule_infers_output(self):
        """With the native rule active, ``Attention`` infers the output shape."""
        try:
            onnx.defs.get_schema("Attention", max_inclusive_version=24, domain="")
        except Exception:
            self.skipTest("Attention op (opset 24) not available in this onnx build")

        model, (b, h, s, d) = self._build_attention()
        infer_symbolic_shapes(model, warn_on_missing=False)
        out = model.graph.outputs[0]
        self.assertEqual(out.dtype, FLOAT)
        self.assertEqual(out.shape, ir.Shape([b, h, s, d]))

    def test_unknown_op_still_warns(self):
        """An op with no registered rule and no function body is a safe no-op.

        The output stays unset and inference does not raise.
        """
        x = _make_value("x", shape=[2, 3], dtype=FLOAT)
        node = ir.Node(
            "com.example", "TotallyMadeUpOp", inputs=[x], outputs=[ir.Value(name="y")]
        )
        graph = ir.Graph(
            inputs=[x],
            outputs=node.outputs,
            nodes=[node],
            opset_imports={"": 24, "com.example": 1},
            name="g",
        )
        model = ir.Model(graph, ir_version=10)
        infer_symbolic_shapes(model, warn_on_missing=False)
        self.assertIsNone(node.outputs[0].shape)


class OpSchemaFunctionUnitTest(unittest.TestCase):
    """White-box tests for the op-schema function fallback machinery."""

    @staticmethod
    def _tiny_function(n_in: int, n_out: int) -> ir.Function:
        ins = [ir.Value(name=f"fi{i}") for i in range(n_in)]
        outs = [ir.Value(name=f"fo{i}") for i in range(n_out)]
        nodes = []
        if n_in and n_out:
            nodes = [ir.Node("", "Identity", inputs=[ins[0]], outputs=[outs[0]])]
        graph = ir.Graph(
            inputs=ins, outputs=outs, nodes=nodes, opset_imports={"": 18}, name="fb"
        )
        return ir.Function(domain="", name="Foo", graph=graph, attributes=[])

    def _seed(self, cache, node, opset, f):
        from onnx_shape_inference import _functions

        key = (
            node.domain or "",
            node.op_type,
            opset,
            _functions._input_type_signature(node),
            _functions._make_attr_signature(node),
        )
        cache[key] = f

    def _call(self, node, *, process_graph_fn, cache, opset=18, active=None):
        from onnx_shape_inference import _functions

        ctx = _context.ShapeInferenceContext({"": opset})
        return _functions.infer_via_op_schema_function(
            ctx,
            node,
            process_graph_fn=process_graph_fn,
            warn_on_missing=False,
            opset_version=opset,
            active_functions=active,
            op_function_cache=cache,
        )

    def test_happy_path_copies_shape_dtype_and_sym_data(self):
        x = _make_value("x", shape=[2, 3], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[x], outputs=[ir.Value(name="y")])
        f = self._tiny_function(1, 1)

        def pg(child_ctx, graph, **kw):
            for o in graph.outputs:
                o.shape = ir.Shape([5, 6])
                o.dtype = FLOAT
                child_ctx.set_symbolic_value(o, [5, 6])
            return True

        cache: dict = {}
        self._seed(cache, node, 18, f)
        ok = self._call(node, process_graph_fn=pg, cache=cache)
        self.assertTrue(ok)
        self.assertEqual(node.outputs[0].shape, ir.Shape([5, 6]))
        self.assertEqual(node.outputs[0].dtype, FLOAT)
        self.assertIn(_context.SYM_DATA_KEY, node.outputs[0].metadata_props)

    def test_recursion_guard_returns_false(self):
        x = _make_value("x", shape=[2], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[x], outputs=[ir.Value(name="y")])

        def pg(child_ctx, graph, **kw):  # pragma: no cover - must not run
            raise AssertionError("body should not run under recursion guard")

        ok = self._call(
            node, process_graph_fn=pg, cache={}, active=frozenset({("", "Foo", "")})
        )
        self.assertFalse(ok)

    def test_degenerate_body_rejected(self):
        x = _make_value("x", shape=[2], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[x], outputs=[ir.Value(name="y")])
        f = self._tiny_function(1, 0)  # no outputs

        def pg(child_ctx, graph, **kw):  # pragma: no cover - must not run
            raise AssertionError("body should not run for degenerate function")

        cache: dict = {}
        self._seed(cache, node, 18, f)
        self.assertFalse(self._call(node, process_graph_fn=pg, cache=cache))

    def test_more_inputs_than_formals_rejected(self):
        a = _make_value("a", shape=[2], dtype=FLOAT)
        b = _make_value("b", shape=[2], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[a, b], outputs=[ir.Value(name="y")])
        f = self._tiny_function(1, 1)  # only one formal input
        cache: dict = {}
        self._seed(cache, node, 18, f)
        self.assertFalse(self._call(node, process_graph_fn=lambda *a, **k: True, cache=cache))

    def test_op_usage_error_propagates(self):
        x = _make_value("x", shape=[2], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[x], outputs=[ir.Value(name="y")])
        f = self._tiny_function(1, 1)

        def pg(child_ctx, graph, **kw):
            raise _context.OpUsageError(node, "boom")

        cache: dict = {}
        self._seed(cache, node, 18, f)
        with self.assertRaises(_context.OpUsageError):
            self._call(node, process_graph_fn=pg, cache=cache)

    def test_generic_exception_returns_false(self):
        x = _make_value("x", shape=[2], dtype=FLOAT)
        node = ir.Node("", "Foo", inputs=[x], outputs=[ir.Value(name="y")])
        f = self._tiny_function(1, 1)

        def pg(child_ctx, graph, **kw):
            raise ValueError("unexpected")

        cache: dict = {}
        self._seed(cache, node, 18, f)
        self.assertFalse(self._call(node, process_graph_fn=pg, cache=cache))

    def test_materialize_unknown_op_returns_none(self):
        from onnx_shape_inference import _functions

        x = _make_value("x", shape=[2], dtype=FLOAT)
        node = ir.Node("com.example", "NoSuchOp", inputs=[x], outputs=[ir.Value(name="y")])
        self.assertIsNone(_functions._materialize_op_schema_function(node, 18))

    def test_materialize_static_function_body(self):
        """An op with a *static* function body materializes to an ir.Function."""
        from onnx_shape_inference import _functions

        try:
            schema = onnx.defs.get_schema("HardSwish", max_inclusive_version=22, domain="")
        except Exception:
            self.skipTest("HardSwish not available")
        if not schema.has_function or schema.has_context_dependent_function:
            self.skipTest("HardSwish is not a static-function op in this build")

        x = _make_value("x", shape=[2, 3], dtype=FLOAT)
        node = ir.Node("", "HardSwish", inputs=[x], outputs=[ir.Value(name="y")])
        f = _functions._materialize_op_schema_function(node, 22)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(len(f.graph), 1)

    def test_materialize_handles_deserialize_failure(self):
        """If body materialization/deserialization raises, the helper returns None."""
        import unittest.mock as mock

        from onnx_shape_inference import _functions

        x = _make_value("x", shape=[2, 8, 96], dtype=FLOAT)
        node = ir.Node("", "Attention", inputs=[x], outputs=[ir.Value(name="y")])
        with mock.patch.object(
            ir.serde, "deserialize_function", side_effect=RuntimeError("boom")
        ):
            self.assertIsNone(_functions._materialize_op_schema_function(node, 24))


class TestChildContextAnchorAdoption(unittest.TestCase):
    """Ensure child-context dimensions are visible to the parent engine.

    A dim minted inside a function body must be visible to the parent engine so
    anchor renaming and reservation work across the boundary.

    These exercise the shared symbol allocator between parent and child contexts.
    """

    def test_child_minted_dim_adopts_parent_output_anchor(self):
        # The function wraps NonZero, whose second output dim is data-dependent
        # and is minted *inside* the function body (a child context).
        func = _single_op_function("test", "MyNonZero", "NonZero", opset=20)
        x = _make_value("X", shape=[2, 5], dtype=FLOAT)
        # The main graph declares a user-visible symbol `dnz` on the output.
        out = _make_value("Y", shape=[2, "dnz"], dtype=INT64)
        call = ir.Node("test", "MyNonZero", inputs=[x], outputs=[out], attributes={})
        model = _make_model([call], [x], [out], [func], extra_domain="test")

        infer_symbolic_shapes(model, warn_on_missing=False)

        # The child-minted anonymous dim was recognised as engine-generated at the
        # parent level, so it adopted the declared anchor name instead of `_dN`.
        self.assertEqual(out.shape, ir.Shape([2, "dnz"]))

    def test_function_body_does_not_mint_author_reserved_name(self):
        # The main graph authors a symbol literally spelled `_d0`; the function
        # body's minted data-dependent dim must not collide with it.
        func = _single_op_function("test", "MyNonZero", "NonZero", opset=20)
        x = _make_value("X", shape=["_d0", 5], dtype=FLOAT)
        out = _make_value("Y", shape=None, dtype=None)
        call = ir.Node("test", "MyNonZero", inputs=[x], outputs=[out], attributes={})
        model = _make_model([call], [x], [out], [func], extra_domain="test")

        infer_symbolic_shapes(model, warn_on_missing=False)

        # NonZero output is [rank, nnz]; the minted nnz dim must not reuse `_d0`.
        self.assertIsNotNone(out.shape)
        minted = out.shape[1]
        self.assertNotEqual(str(minted), "_d0")


if __name__ == "__main__":
    unittest.main()
