# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Whole-model shape-inference cases ported from onnx-light.

These mirror the manually-built ``cases_for_shapes/inference/*.cc`` test
models from
https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_backend_test/cases_for_shapes/inference

Each model is built directly with ``onnx_ir`` and run through
:func:`infer_symbolic_shapes`.  The assertions check, for every value of
interest, that:

* the inferred element type matches exactly,
* every **concrete** (integer) dimension matches exactly, and
* every **symbolic** dimension is *mathematically equivalent* to the expected
  expression (compared via SymPy), so the relationships between related
  shapes are verified — e.g. ``concat_out`` really is ``2 * d_model`` and
  ``flat_nz`` really is ``2 * nnz``.

Genuinely data-dependent dims that shape inference must invent (the ``nnz``
count from ``NonZero``, a merged ``If``-branch axis) appear as engine-anonymous
``_dN`` symbols.  These are bound to the expected wildcard symbol on first
sight and then enforced consistently across the rest of the model, so a
relationship such as ``flat_nz == 2 * nnz`` is still checked end to end.
"""

from __future__ import annotations

import re
import unittest

import numpy as np
import onnx_ir as ir
import sympy

from onnx_shape_inference import infer_symbolic_shapes
from onnx_shape_inference._symbolic_shapes import parse_symbolic_expression

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64
BOOL = ir.DataType.BOOL

# Engine-generated anonymous dims (genuinely data-dependent / unknowable, e.g.
# the ``nnz`` count from NonZero or a merged If-branch axis).  These are matched
# by *binding*: the first time one appears it is bound to the corresponding
# expected wildcard symbol, and that binding is then enforced consistently
# across every other value in the same model (so e.g. ``flat_nz == 2 * nnz``).
_ANON_DIM = re.compile(r"^_d\d+$")


def _dims_equiv(got: sympy.Expr, expected: sympy.Expr) -> bool:
    """Return whether two integer dim expressions are mathematically equal.

    Falls back to evaluating both over several positive-integer assignments of
    their free symbols, so floor/ceil identities (which ``sympy.simplify`` does
    not always reduce) are still recognised — e.g. ``-floor(-b/2 - c/2)`` equals
    ``(1 + b + c)//2``.
    """
    diff = sympy.simplify(got - expected)
    if diff == 0:
        return True
    symbols = sorted(diff.free_symbols, key=lambda s: s.name)
    # Two distinct deterministic assignments guard against coincidences.
    for base in (3, 8):
        subs = {s: base + i for i, s in enumerate(symbols)}
        value = diff.subs(subs)
        if not value.is_number or value != 0:
            return False
    return True


def _val(name: str, dtype: ir.DataType | None = None, shape=None) -> ir.Value:
    type_ = ir.TensorType(dtype) if dtype is not None else None
    shape_ = ir.Shape(shape) if shape is not None else None
    return ir.Value(name=name, shape=shape_, type=type_)


def _init(name: str, arr: np.ndarray) -> ir.Value:
    arr = np.asarray(arr)
    tensor = ir.Tensor(arr, name=name)
    return ir.Value(
        name=name,
        const_value=tensor,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(arr.shape)),
    )


def _attr(name: str, value, typ: ir.AttributeType = ir.AttributeType.INT) -> ir.Attr:
    return ir.Attr(name, typ, value)


def _node(op: str, inputs, outputs, attrs=None, domain: str = "") -> ir.Node:
    outs = [o if isinstance(o, ir.Value) else ir.Value(name=o) for o in outputs]
    return ir.Node(domain, op, inputs=inputs, outputs=outs, attributes=attrs or {})


def _build_model(nodes, inputs, outputs, inits=None, opset=18, functions=None, ir_version=10):
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=inits or [],
        opset_imports={"": opset},
        name="g",
    )
    return ir.Model(graph, ir_version=ir_version, functions=functions or [])


class ShapeInferenceCasesTest(unittest.TestCase):
    """Ported onnx-light whole-model shape-inference cases."""

    def _assert_infer(self, model: ir.Model, expected: dict) -> None:
        """Run inference and compare against ``{name: (dtype, dims)}``.

        Concrete (int) dims must match exactly.  Symbolic dims are compared by
        SymPy equivalence to the expected expression, so the *relationships*
        between related shapes are verified (e.g. ``concat_out`` really is
        ``2 * d_model``, ``flat_nz`` really is ``2 * nnz``).  Engine-anonymous
        ``_dN`` dims are bound to the expected wildcard symbol on first sight
        and then enforced consistently across the whole model.  ``dims=None``
        means unknown rank (shape must be unset).
        """
        infer_symbolic_shapes(model)

        by_name: dict[str, ir.Value] = {}
        for v in list(model.graph.inputs) + list(model.graph.outputs):
            by_name[v.name] = v
        for n in model.graph:
            for o in n.outputs:
                if o is not None and o.name:
                    by_name[o.name] = o

        # Bindings from engine-anonymous dim symbols to expected wildcard exprs.
        bindings: dict[sympy.Symbol, sympy.Expr] = {}

        for name, (edtype, edims) in expected.items():
            v = by_name.get(name)
            self.assertIsNotNone(v, f"value {name!r} not found in graph")
            got_dtype = v.type.dtype if v.type is not None else None
            self.assertEqual(got_dtype, edtype, f"{name}: dtype mismatch")

            if edims is None:
                self.assertIsNone(v.shape, f"{name}: expected unknown rank")
                continue
            self.assertIsNotNone(v.shape, f"{name}: expected shape {edims}, got None")
            got = list(v.shape)
            self.assertEqual(
                len(got), len(edims), f"{name}: rank mismatch, expected {edims}, got {got}"
            )
            for e, g in zip(edims, got):
                self._assert_dim(name, e, g, bindings)

    def _assert_dim(self, name, expected, got, bindings) -> None:
        if isinstance(expected, int):
            self.assertEqual(
                got, expected, f"{name}: expected concrete {expected}, got {got!r}"
            )
            return

        expected_expr = parse_symbolic_expression(expected)
        got_expr = (
            sympy.Integer(got) if isinstance(got, int) else parse_symbolic_expression(str(got))
        )
        got_subst = got_expr.subs(bindings)

        if _dims_equiv(got_subst, expected_expr):
            return

        # Bind a fresh engine-anonymous dim to the expected wildcard symbol.
        if (
            isinstance(got_expr, sympy.Symbol)
            and _ANON_DIM.match(got_expr.name)
            and got_expr not in bindings
        ):
            bindings[got_expr] = expected_expr
            return

        self.fail(f"{name}: dim mismatch, expected {expected!r}, got {got!r}")

    # ------------------------------------------------------------------
    # cases_add_concat_reshape.cc
    # ------------------------------------------------------------------
    def test_add_concat_reshape(self):
        x = _val("X", FLOAT, ["batch", "seq", "d_model"])
        y = _val("Y", FLOAT, ["batch", "seq", "d_model"])
        reshape_shape = _init("reshape_shape", np.array([0, 0, -1], dtype=np.int64))
        n1 = _node("Add", [x, y], ["added"])
        n2 = _node("Concat", [n1.outputs[0], x], ["concat_out"], {"axis": _attr("axis", 2)})
        n3 = _node("Reshape", [n2.outputs[0], reshape_shape], ["Z_pre_abs"])
        n4 = _node("Abs", [n3.outputs[0]], ["Z"])
        model = _build_model([n1, n2, n3, n4], [x, y], [n4.outputs[0]], inits=[reshape_shape])
        self._assert_infer(
            model,
            {
                "added": (FLOAT, ["batch", "seq", "d_model"]),
                "concat_out": (FLOAT, ["batch", "seq", "2*d_model"]),
                "Z": (FLOAT, ["batch", "seq", "2*d_model"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_shape_identity_unsqueeze.cc
    # ------------------------------------------------------------------
    def test_shape_identity_unsqueeze(self):
        k = 15
        inp = _val("input", FLOAT, [1] * k)
        axes = _init("unsq_axes", np.arange(k, dtype=np.int64))
        n1 = _node("Shape", [inp], ["shape_out"])
        n2 = _node("Identity", [n1.outputs[0]], ["identity_out"])
        n3 = _node("Unsqueeze", [n2.outputs[0], axes], ["output_pre_abs"])
        n4 = _node("Abs", [n3.outputs[0]], ["output"])
        model = _build_model([n1, n2, n3, n4], [inp], [n4.outputs[0]], inits=[axes])
        self._assert_infer(
            model,
            {
                "shape_out": (INT64, [k]),
                "identity_out": (INT64, [k]),
                "output": (INT64, [1] * k + [k]),
            },
        )

    # ------------------------------------------------------------------
    # cases_nonzero_chain.cc — NonZero -> Transpose -> Cast chain
    # ------------------------------------------------------------------
    def test_nonzero_chain(self):
        x = _val("X", FLOAT, ["batch", "seq"])
        n1 = _node("Abs", [x], ["abs_out"])
        n2 = _node("Relu", [n1.outputs[0]], ["relu_out"])
        n3 = _node("Add", [n2.outputs[0], n2.outputs[0]], ["double_out"])
        n4 = _node("Mul", [n3.outputs[0], n2.outputs[0]], ["mul_out"])
        n5 = _node("NonZero", [n4.outputs[0]], ["nz_pre_abs"])
        n6 = _node("Abs", [n5.outputs[0]], ["nz"])
        n7 = _node("Transpose", [n6.outputs[0]], ["transposed_nz"])
        n8 = _node(
            "Cast", [n7.outputs[0]], ["nz_float_pre_abs"], {"to": _attr("to", int(FLOAT))}
        )
        n9 = _node("Abs", [n8.outputs[0]], ["nz_float"])
        model = _build_model(
            [n1, n2, n3, n4, n5, n6, n7, n8, n9], [x], [n6.outputs[0], n9.outputs[0]]
        )
        self._assert_infer(
            model,
            {
                "nz_pre_abs": (INT64, [2, "nnz"]),
                "nz": (INT64, [2, "nnz"]),
                "transposed_nz": (INT64, ["nnz", 2]),
                "nz_float": (FLOAT, ["nnz", 2]),
            },
        )

    # ------------------------------------------------------------------
    # cases_nonzero_chain.cc — NonZero -> Reshape([-1]) dimension expression
    # ------------------------------------------------------------------
    def test_nonzero_plus_expression(self):
        x = _val("X", FLOAT, ["batch", "seq"])
        m1 = _init("m1", np.array([-1], dtype=np.int64))
        n1 = _node("Abs", [x], ["abs_out"])
        n2 = _node("NonZero", [n1.outputs[0]], ["nz"])
        n3 = _node("Reshape", [n2.outputs[0], m1], ["flat_nz"])
        n4 = _node("Neg", [n3.outputs[0]], ["Y_pre_abs"])
        n5 = _node("Abs", [n4.outputs[0]], ["Y"])
        model = _build_model([n1, n2, n3, n4, n5], [x], [n5.outputs[0]], inits=[m1])
        self._assert_infer(
            model,
            {
                "nz": (INT64, [2, "dnz"]),
                "flat_nz": (INT64, ["2*dnz"]),
                "Y": (INT64, ["2*dnz"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_local_function.cc — single model-local function call
    # ------------------------------------------------------------------
    def test_local_function_add(self):
        a, b = ir.Value(name="a"), ir.Value(name="b")
        body = _node("Add", [a, b], ["c"])
        fgraph = ir.Graph(
            inputs=[a, b],
            outputs=[body.outputs[0]],
            nodes=[body],
            opset_imports={"": 18},
            name="func_add",
        )
        func = ir.Function(domain="local", name="func_add", graph=fgraph, attributes=[])

        x = _val("X", FLOAT, ["batch", "d_model"])
        y = _val("Y", FLOAT, ["batch", "d_model"])
        call = _node("func_add", [x, y], ["Z_pre_abs"], domain="local")
        ab = _node("Abs", [call.outputs[0]], ["Z"])
        graph = ir.Graph(
            inputs=[x, y],
            outputs=[ab.outputs[0]],
            nodes=[call, ab],
            opset_imports={"": 18, "local": 1},
            name="g",
        )
        model = ir.Model(graph, ir_version=10, functions=[func])
        self._assert_infer(
            model,
            {
                "Z_pre_abs": (FLOAT, ["batch", "d_model"]),
                "Z": (FLOAT, ["batch", "d_model"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_nested_local_function.cc — function calling another function
    # ------------------------------------------------------------------
    def test_nested_local_function_add(self):
        a, b = ir.Value(name="a"), ir.Value(name="b")
        inner_body = _node("Add", [a, b], ["c"])
        inner_g = ir.Graph(
            inputs=[a, b],
            outputs=[inner_body.outputs[0]],
            nodes=[inner_body],
            opset_imports={"": 18},
            name="func_inner_add",
        )
        inner = ir.Function(
            domain="local", name="func_inner_add", graph=inner_g, attributes=[]
        )

        a2, b2 = ir.Value(name="a"), ir.Value(name="b")
        outer_body = _node("func_inner_add", [a2, b2], ["c"], domain="local")
        outer_g = ir.Graph(
            inputs=[a2, b2],
            outputs=[outer_body.outputs[0]],
            nodes=[outer_body],
            opset_imports={"local": 1},
            name="func_outer_add",
        )
        outer = ir.Function(
            domain="local", name="func_outer_add", graph=outer_g, attributes=[]
        )

        x = _val("X", FLOAT, ["batch", "d_model"])
        y = _val("Y", FLOAT, ["batch", "d_model"])
        call = _node("func_outer_add", [x, y], ["Z_pre_abs"], domain="local")
        ab = _node("Abs", [call.outputs[0]], ["Z"])
        graph = ir.Graph(
            inputs=[x, y],
            outputs=[ab.outputs[0]],
            nodes=[call, ab],
            opset_imports={"": 18, "local": 1},
            name="g",
        )
        model = ir.Model(graph, ir_version=10, functions=[inner, outer])
        self._assert_infer(
            model,
            {
                "Z_pre_abs": (FLOAT, ["batch", "d_model"]),
                "Z": (FLOAT, ["batch", "d_model"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_if_symbolic_shapes.cc — If with branch-merged symbolic shapes
    # ------------------------------------------------------------------
    def test_if_symbolic_shapes(self):
        cond = _val("cond", BOOL, [])
        a_then = _val("a_then", FLOAT, [3, 4])
        a_else = _val("a_else", FLOAT, [5, 4])
        c_else = _val("c_else", BOOL, [5])
        b_then = _val("b_then", INT64, [3])
        b_else = _val("b_else", INT64, [5])

        t1 = _node("Identity", [a_then], ["out_a_then"])
        t2 = _node("Identity", [b_then], ["out_b_then"])
        then_g = ir.Graph(
            inputs=[],
            outputs=[t1.outputs[0], t2.outputs[0]],
            nodes=[t1, t2],
            opset_imports={"": 13},
            name="then_branch",
        )
        e1 = _node("Compress", [a_else, c_else], ["out_a_else"], {"axis": _attr("axis", 0)})
        e2 = _node("Identity", [b_else], ["out_b_else"])
        else_g = ir.Graph(
            inputs=[],
            outputs=[e1.outputs[0], e2.outputs[0]],
            nodes=[e1, e2],
            opset_imports={"": 13},
            name="else_branch",
        )
        if_node = _node(
            "If",
            [cond],
            ["out_a", "out_b"],
            {
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_g),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_g),
            },
        )
        graph = ir.Graph(
            inputs=[cond, a_then, a_else, c_else, b_then, b_else],
            outputs=list(if_node.outputs),
            nodes=[if_node],
            opset_imports={"": 13},
            name="g",
        )
        model = ir.Model(graph, ir_version=10)
        self._assert_infer(
            model,
            {
                # leading axis differs between branches -> fresh symbolic dim,
                # trailing concrete 4 preserved.
                "out_a": (FLOAT, ["If_out_a_d0", 4]),
                "out_b": (INT64, ["If_out_b_d0"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_loop_pairwise_distance.cc — Loop with outer-scope capture
    # ------------------------------------------------------------------
    def test_loop_pairwise_distance(self):
        x = _val("X", FLOAT, ["batch", "feat"])
        zero_idx = _init("zero_idx", np.array([0], dtype=np.int64))
        unsq_axes = _init("unsqueeze_axes", np.array([0], dtype=np.int64))
        red_axes = _init("reduce_axes", np.array([-1], dtype=np.int64))
        cond_init = _init("cond_init", np.array([True]))

        iter_count = _val("iter_count", INT64, [])
        cond_in = _val("cond_in", BOOL, [])
        bi = _node("Identity", [cond_in], ["cond_out"])
        bu = _node("Unsqueeze", [iter_count, unsq_axes], ["iter_1d"])
        bg = _node("Gather", [x, bu.outputs[0]], ["x_i"], {"axis": _attr("axis", 0)})
        bs = _node("Sub", [x, bg.outputs[0]], ["diff"])
        bm = _node("Mul", [bs.outputs[0], bs.outputs[0]], ["sq"])
        br = _node(
            "ReduceSum",
            [bm.outputs[0], red_axes],
            ["sum_sq"],
            {"keepdims": _attr("keepdims", 0)},
        )
        bsq = _node("Sqrt", [br.outputs[0]], ["dist"])
        bo = _node("Identity", [bsq.outputs[0]], ["scan_out"])
        body = ir.Graph(
            inputs=[iter_count, cond_in],
            outputs=[bi.outputs[0], bo.outputs[0]],
            nodes=[bi, bu, bg, bs, bm, br, bsq, bo],
            opset_imports={"": 18},
            name="pairwise_distance_body",
        )

        n1 = _node("Shape", [x], ["shape_X"])
        n2 = _node(
            "Gather", [n1.outputs[0], zero_idx], ["trip_count"], {"axis": _attr("axis", 0)}
        )
        loop = _node(
            "Loop",
            [n2.outputs[0], cond_init],
            ["Y_pre_abs"],
            {"body": ir.Attr("body", ir.AttributeType.GRAPH, body)},
        )
        ab = _node("Abs", [loop.outputs[0]], ["Y"])
        model = _build_model(
            [n1, n2, loop, ab],
            [x],
            [ab.outputs[0]],
            inits=[zero_idx, unsq_axes, red_axes, cond_init],
        )
        self._assert_infer(
            model,
            {
                "shape_X": (INT64, [2]),
                "trip_count": (INT64, [1]),
                "Y": (FLOAT, ["batch", "batch"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_shape_builder.cc — Shape/Concat -> Reshape/MatMul chain
    # ------------------------------------------------------------------
    def test_check_shape(self):
        x = _val("X", FLOAT, ["D32", "D64"])
        y = _val("Y", FLOAT, ["batch", "channel", "D128", "D64"])
        zero = _init("zero", np.array([0], dtype=np.int64))
        un = _init("un", np.array([1], dtype=np.int64))
        c0 = _init("c0", np.array([0], dtype=np.int64))
        cm1 = _init("cm1", np.array([-1], dtype=np.int64))

        sx = _node("Shape", [x], ["x_last_dim"], {"start": _attr("start", -1)})
        s1 = _node("Concat", [c0, cm1, sx.outputs[0]], ["shape1"], {"axis": _attr("axis", 0)})
        sy = _node(
            "Shape", [y], ["y_dim2"], {"start": _attr("start", 2), "end": _attr("end", 3)}
        )
        s2 = _node(
            "Concat",
            [cm1, sx.outputs[0], sy.outputs[0]],
            ["shape2"],
            {"axis": _attr("axis", 0)},
        )
        yf = _node(
            "Shape", [y], ["y_first2"], {"start": _attr("start", 0), "end": _attr("end", 2)}
        )
        yl = _node("Shape", [y], ["y_last_dim"], {"start": _attr("start", -1)})
        s3 = _node(
            "Concat",
            [yf.outputs[0], sx.outputs[0], yl.outputs[0]],
            ["shape3"],
            {"axis": _attr("axis", 0)},
        )

        u1 = _node("Unsqueeze", [x, zero], ["xu1"])
        u2 = _node("Unsqueeze", [u1.outputs[0], un], ["xu2"])
        r1 = _node("Reshape", [u2.outputs[0], s1.outputs[0]], ["xm1"])
        r2 = _node("Reshape", [y, s2.outputs[0]], ["xm2c"])
        cst = _node("Cast", [r2.outputs[0]], ["xm2"], {"to": _attr("to", int(FLOAT))})
        mm = _node("MatMul", [r1.outputs[0], cst.outputs[0]], ["xm"])
        rz = _node("Reshape", [mm.outputs[0], s3.outputs[0]], ["Z"])
        model = _build_model(
            [sx, s1, sy, s2, yf, yl, s3, u1, u2, r1, r2, cst, mm, rz],
            [x, y],
            [rz.outputs[0]],
            inits=[zero, un, c0, cm1],
        )
        self._assert_infer(
            model,
            {
                "shape1": (INT64, [3]),
                "shape2": (INT64, [3]),
                "shape3": (INT64, [4]),
                "xu1": (FLOAT, [1, "D32", "D64"]),
                "xu2": (FLOAT, [1, 1, "D32", "D64"]),
                "xm1": (FLOAT, [1, "D32", "D64"]),
                "xm2c": (FLOAT, ["batch*channel", "D64", "D128"]),
                "xm": (FLOAT, ["batch*channel", "D32", "D128"]),
                "Z": (FLOAT, ["batch", "channel", "D64", "D64"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_shape_builder.cc — Reshape([0,0,2,-1]) -> Reshape([0,0,-1])
    # ------------------------------------------------------------------
    def test_reshape_reshape(self):
        x = _val("X", FLOAT, ["a", "b", "c"])
        s1 = _init("shape1", np.array([0, 0, 2, -1], dtype=np.int64))
        s2 = _init("shape2", np.array([0, 0, -1], dtype=np.int64))
        one = _init("one", np.array([1.0], dtype=np.float32))
        r1 = _node("Reshape", [x, s1], ["xr"])
        r2 = _node("Reshape", [r1.outputs[0], s2], ["xrr"])
        ad = _node("Add", [r2.outputs[0], one], ["Y"])
        model = _build_model([r1, r2, ad], [x], [ad.outputs[0]], inits=[s1, s2, one])
        self._assert_infer(
            model,
            {
                "xr": (FLOAT, ["a", "b", 2, "c//2"]),
                # Exact symbolic simplification collapses 2*(c//2) to c.
                "xrr": (FLOAT, ["a", "b", "c"]),
                "Y": (FLOAT, ["a", "b", "c"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_shape_builder.cc — value-as-shape -> Reshape -> Transpose
    # ------------------------------------------------------------------
    def test_value_as_shape_builder(self):
        ids = _val("ids_weight", FLOAT, ["batch", "seq", 256])
        init328 = _init("init328", np.array([32, 8], dtype=np.int64))
        a = _init("A", np.zeros((256, 256), dtype=np.float32))
        b = _init("B", np.zeros((256, 256), dtype=np.float32))
        c = _init("C", np.zeros((256, 256), dtype=np.float32))
        sh = _node(
            "Shape", [ids], ["shape"], {"start": _attr("start", 0), "end": _attr("end", 2)}
        )
        ns = _node(
            "Concat", [sh.outputs[0], init328], ["new_shape"], {"axis": _attr("axis", 0)}
        )
        a1 = _node("MatMul", [ids, a], ["A1"])
        b1 = _node("MatMul", [ids, b], ["B1"])
        c1 = _node("MatMul", [ids, c], ["C1"])
        ar = _node("Reshape", [a1.outputs[0], ns.outputs[0]], ["Areshaped"])
        bre = _node("Reshape", [b1.outputs[0], ns.outputs[0]], ["Breshaped"])
        cr = _node("Reshape", [c1.outputs[0], ns.outputs[0]], ["Creshaped"])
        perm = ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 1, 3])
        at = _node("Transpose", [ar.outputs[0]], ["At"], {"perm": perm})
        bt = _node("Transpose", [bre.outputs[0]], ["Bt"], {"perm": perm})
        ct = _node("Transpose", [cr.outputs[0]], ["Ct"], {"perm": perm})
        model = _build_model(
            [sh, ns, a1, b1, c1, ar, bre, cr, at, bt, ct],
            [ids],
            [at.outputs[0], bt.outputs[0], ct.outputs[0]],
            inits=[init328, a, b, c],
        )
        self._assert_infer(
            model,
            {
                "shape": (INT64, [2]),
                "new_shape": (INT64, [4]),
                "A1": (FLOAT, ["batch", "seq", 256]),
                "Areshaped": (FLOAT, ["batch", "seq", 32, 8]),
                "At": (FLOAT, ["batch", 32, "seq", 8]),
                "Bt": (FLOAT, ["batch", 32, "seq", 8]),
                "Ct": (FLOAT, ["batch", 32, "seq", 8]),
            },
        )

    # ------------------------------------------------------------------
    # cases_shape_builder.cc — Concat -> Split -> Concat (even / odd)
    # ------------------------------------------------------------------
    def _build_concat_split(self, even: bool) -> ir.Model:
        if even:
            x = _val("X", FLOAT, ["a", "2*b"])
            y = _val("Y", FLOAT, ["a", "2*c"])
        else:
            x = _val("X", FLOAT, ["a", "b"])
            y = _val("Y", FLOAT, ["a", "c"])
        c1 = _node("Concat", [x, y], ["xy"], {"axis": _attr("axis", 1)})
        sp = _node(
            "Split",
            [c1.outputs[0]],
            ["S1", "S2"],
            {"axis": _attr("axis", 1), "num_outputs": _attr("num_outputs", 2)},
        )
        c2 = _node(
            "Concat", [sp.outputs[1], sp.outputs[0]], ["zs"], {"axis": _attr("axis", 1)}
        )
        rl = _node("Relu", [c2.outputs[0]], ["Z"])
        return _build_model([c1, sp, c2, rl], [x, y], [rl.outputs[0]])

    def test_concat_split_even(self):
        model = self._build_concat_split(even=True)
        self._assert_infer(
            model,
            {
                "xy": (FLOAT, ["a", "2*b+2*c"]),
                "S1": (FLOAT, ["a", "b+c"]),
                "S2": (FLOAT, ["a", "b+c"]),
                "Z": (FLOAT, ["a", "2*b+2*c"]),
            },
        )

    def test_concat_split_odd(self):
        model = self._build_concat_split(even=False)
        self._assert_infer(
            model,
            {
                "xy": (FLOAT, ["a", "b+c"]),
                "S1": (FLOAT, ["a", "(1+b+c)//2"]),
                "S2": (FLOAT, ["a", "(b+c)//2"]),
                "Z": (FLOAT, ["a", "b+c"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_value_as_shape.cc — value-as-shape through Add/Sub -> Expand
    # ------------------------------------------------------------------
    def test_value_as_shape(self):
        """Value-as-shape data propagates through ``Add``/``Sub`` to ``Expand``.

        Broadcasting of the length-1 ``one`` operand lets the shape tensor
        ``shape2`` resolve to ``[N, 1]``, so ``Expand`` recovers ``(N, 1)``.
        """
        x = _val("x", FLOAT, ["N", 1])
        y1 = _val("y1", FLOAT, [1, "B"])
        y2 = _val("y2", FLOAT, [1, "B"])
        y3 = _val("y3", FLOAT, [1, "B"])
        one = _init("one", np.array([1], dtype=np.int64))
        nn = _node("Shape", [x], ["n"], {"start": _attr("start", 0), "end": _attr("end", 1)})
        nb = _node("Shape", [x], ["b"], {"start": _attr("start", 1), "end": _attr("end", 2)})
        nc = _node(
            "Concat", [nn.outputs[0], nb.outputs[0]], ["shape"], {"axis": _attr("axis", 0)}
        )
        na = _node("Add", [nc.outputs[0], one], ["shape1"])
        ns = _node("Sub", [na.outputs[0], one], ["shape2"])
        ne = _node("Expand", [x, ns.outputs[0]], ["expanded"])
        z1 = _node("Add", [ne.outputs[0], y1], ["z1"])
        z2 = _node("Add", [ne.outputs[0], y2], ["z2"])
        z3 = _node("Add", [ne.outputs[0], y3], ["z3"])
        z12 = _node("Add", [z1.outputs[0], z2.outputs[0]], ["z12"])
        zpa = _node("Add", [z12.outputs[0], z3.outputs[0]], ["z_pre_abs"])
        zz = _node("Abs", [zpa.outputs[0]], ["z"])
        model = _build_model(
            [nn, nb, nc, na, ns, ne, z1, z2, z3, z12, zpa, zz],
            [x, y1, y2, y3],
            [zz.outputs[0]],
            inits=[one],
            opset=20,
        )
        self._assert_infer(
            model,
            {
                "expanded": (FLOAT, ["N", 1]),
                "z1": (FLOAT, ["N", "B"]),
                "z": (FLOAT, ["N", "B"]),
            },
        )

    # ------------------------------------------------------------------
    # cases_scan_running_sum.cc — Scan with concrete state initializer
    # ------------------------------------------------------------------
    def test_scan_running_sum(self):
        """Concrete Scan state dims from an initializer flow into the body.

        The ``[3]`` shape of the ``zero_acc`` initializer is merged into the
        symbolic body input ``acc_in: [D]``, so the state and stacked
        scan-output dim resolve to the concrete ``3``.
        """
        x = _val("X", FLOAT, ["T", "D"])
        zero_acc = _init("zero_acc", np.zeros(3, dtype=np.float32))

        acc_in = _val("acc_in", FLOAT, ["D"])
        x_t = _val("x_t", FLOAT, ["D"])
        ba = _node("Add", [acc_in, x_t], ["acc_out"])
        bo = _node("Identity", [ba.outputs[0]], ["scan_out"])
        body = ir.Graph(
            inputs=[acc_in, x_t],
            outputs=[ba.outputs[0], bo.outputs[0]],
            nodes=[ba, bo],
            opset_imports={"": 18},
            name="running_sum_body",
        )
        scan = _node(
            "Scan",
            [zero_acc, x],
            ["acc_final", "Y_pre_abs"],
            {
                "num_scan_inputs": _attr("num_scan_inputs", 1),
                "body": ir.Attr("body", ir.AttributeType.GRAPH, body),
            },
        )
        ab = _node("Abs", [scan.outputs[1]], ["Y"])
        model = _build_model([scan, ab], [x], [ab.outputs[0]], inits=[zero_acc])
        self._assert_infer(
            model,
            {
                "acc_final": (FLOAT, [3]),
                "Y_pre_abs": (FLOAT, ["T", 3]),
                "Y": (FLOAT, ["T", 3]),
            },
        )


if __name__ == "__main__":
    unittest.main()
