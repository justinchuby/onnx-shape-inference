# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""FLOAT6 metadata inference against released ONNX opset 28 schemas."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir
import parameterized

from onnx_shape_inference import ShapeInferenceError, infer_symbolic_shapes
from onnx_shape_inference._ops._testing import run_shape_inference, ts

_FLOAT6_NAMES = ("FLOAT6E2M3", "FLOAT6E3M2")
_LAYOUTS = (
    ("scalar", [], [], {}),
    ("tensor", [2, 16], [], {}),
    ("axis", ["B", 16], [16], {"axis": 1}),
    ("blocked", ["B", 16], ["B", 4], {"axis": 1, "block_size": 4}),
    ("unknown", None, [], {}),
)


@unittest.skipUnless(
    all(name in ir.DataType.__members__ for name in _FLOAT6_NAMES),
    "FLOAT6 requires onnx-ir main or a release exposing FLOAT6E2M3/FLOAT6E3M2; "
    "onnx-ir 1.0.0 cannot represent these dtypes",
)
class Float6Test(unittest.TestCase):
    def _check(self, op, inputs, attrs, expected):
        actual = run_shape_inference(
            "",
            op,
            inputs,
            {name: ir.AttrInt64(name, value) for name, value in attrs.items()},
            opset_version=28,
        )
        self.assertEqual(actual, [expected])
        graph_inputs = [
            onnx.helper.make_tensor_value_info(
                f"in{i}",
                spec.type.dtype.value,
                [dim if isinstance(dim, int) else str(dim) for dim in spec.shape]
                if spec.shape is not None
                else None,
            )
            for i, spec in enumerate(inputs)
        ]
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(op, [v.name for v in graph_inputs], ["y"], **attrs)],
            "float6",
            graph_inputs,
            [onnx.ValueInfoProto(name="y")],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 28)])
        reference = ir.serde.deserialize_model(
            onnx.shape_inference.infer_shapes(model, strict_mode=True)
        ).graph.outputs[0]
        self.assertEqual(actual, [ir.TypeAndShape(reference.type, reference.shape)])

    @parameterized.parameterized.expand(
        [
            (f"{name}_{label}_{direction}", name, shape, direction)
            for name in _FLOAT6_NAMES
            for label, shape in (
                ("scalar", []),
                ("concrete", [2, 16]),
                ("symbolic", ["B", 16]),
                ("unknown", None),
            )
            for direction in ("to", "from")
        ]
    )
    def test_cast(self, _name, name, shape, direction):
        dtype = ir.DataType[name]
        source, target = (
            (ir.DataType.FLOAT, dtype) if direction == "to" else (dtype, ir.DataType.FLOAT16)
        )
        self._check("Cast", [ts(source, shape)], {"to": target.value}, ts(target, shape))

    @parameterized.parameterized.expand(
        [
            (f"{name}_{label}_{source}", name, shape, scale_shape, attrs, source)
            for name in _FLOAT6_NAMES
            for label, shape, scale_shape, attrs in _LAYOUTS
            for source in ("attribute", "zero_point", "default_attribute", "both")
        ]
    )
    def test_quantize(self, _name, name, shape, scale_shape, layout_attrs, source):
        dtype = ir.DataType[name]
        inputs = [ts(ir.DataType.FLOAT, shape), ts(ir.DataType.FLOAT, scale_shape)]
        attrs = dict(layout_attrs)
        if source != "attribute":
            inputs.append(ts(dtype, scale_shape))
        if source in ("attribute", "both"):
            attrs["output_dtype"] = dtype.value
        elif source == "default_attribute":
            attrs["output_dtype"] = onnx.TensorProto.UNDEFINED
        self._check("QuantizeLinear", inputs, attrs, ts(dtype, shape))

    @parameterized.parameterized.expand(
        [
            (f"{name}_{label}_{source}", name, shape, scale_shape, attrs, source)
            for name in _FLOAT6_NAMES
            for label, shape, scale_shape, attrs in _LAYOUTS
            for source in ("scale", "default_attribute", "attribute")
        ]
    )
    def test_dequantize(self, _name, name, shape, scale_shape, layout_attrs, source):
        dtype = ir.DataType[name]
        attrs = dict(layout_attrs)
        output_dtype = ir.DataType.FLOAT16
        if source == "attribute":
            attrs["output_dtype"] = ir.DataType.BFLOAT16.value
            output_dtype = ir.DataType.BFLOAT16
        elif source == "default_attribute":
            attrs["output_dtype"] = onnx.TensorProto.UNDEFINED
        self._check(
            "DequantizeLinear",
            [ts(dtype, shape), ts(ir.DataType.FLOAT16, scale_shape), ts(dtype, scale_shape)],
            attrs,
            ts(output_dtype, shape),
        )

    @parameterized.parameterized.expand([(name,) for name in _FLOAT6_NAMES])
    def test_round_trip_graph_and_type_serialization(self, name):
        dtype = ir.DataType[name]
        x = ir.Value(
            name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["B", 16])
        )
        scale = ir.Value(
            name="scale", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([])
        )
        quantize = ir.Node(
            "",
            "QuantizeLinear",
            [x, scale],
            num_outputs=1,
            attributes={"output_dtype": ir.AttrInt64("output_dtype", dtype.value)},
        )
        cast = ir.Node(
            "",
            "Cast",
            quantize.outputs,
            num_outputs=1,
            attributes={"to": ir.AttrInt64("to", dtype.value)},
        )
        dequantize = ir.Node(
            "",
            "DequantizeLinear",
            [cast.outputs[0], scale],
            num_outputs=1,
        )
        model = ir.Model(
            ir.Graph(
                [x, scale],
                dequantize.outputs,
                nodes=[quantize, cast, dequantize],
                opset_imports={"": 28},
                name="float6_round_trip",
            ),
            ir_version=onnx.IR_VERSION,
        )
        infer_symbolic_shapes(model)
        for node, expected_dtype in (
            (quantize, dtype),
            (cast, dtype),
            (dequantize, ir.DataType.FLOAT),
        ):
            self.assertEqual(
                ir.TypeAndShape(node.outputs[0].type, node.outputs[0].shape),
                ts(expected_dtype, ["B", 16]),
            )
        proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(proto, full_check=True)
        restored = ir.serde.deserialize_model(proto)
        self.assertEqual(
            [node.outputs[0].dtype for node in restored.graph],
            [dtype, dtype, ir.DataType.FLOAT],
        )


class QuantizationDtypeContractTest(unittest.TestCase):
    @parameterized.parameterized.expand([(name,) for name in _FLOAT6_NAMES])
    def test_released_schema_constraints(self, name):
        tensor_type = f"tensor({name.lower()})"
        for op, parameter in (
            ("Cast", "T1"),
            ("Cast", "T2"),
            ("QuantizeLinear", "T3"),
            ("DequantizeLinear", "T1"),
        ):
            schema = onnx.defs.get_schema(op, 28)
            allowed = next(
                constraint.allowed_type_strs
                for constraint in schema.type_constraints
                if constraint.type_param_str == parameter
            )
            self.assertIn(tensor_type, allowed)

    @parameterized.parameterized.expand(
        [
            ("QuantizeLinear", ir.DataType.FLOAT, ir.DataType.INT8, ir.DataType.INT8),
            ("DequantizeLinear", ir.DataType.UINT8, ir.DataType.UINT8, ir.DataType.FLOAT16),
        ]
    )
    def test_explicit_default_dtype(self, op, input_dtype, zero_dtype, expected_dtype):
        actual = run_shape_inference(
            "",
            op,
            [ts(input_dtype, ["B", 16]), ts(ir.DataType.FLOAT16, []), ts(zero_dtype, [])],
            {"output_dtype": ir.AttrInt64("output_dtype", 0)},
            opset_version=28,
        )
        self.assertEqual(actual, [ts(expected_dtype, ["B", 16])])

    def test_default_without_zero_point(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "QuantizeLinear",
                [ts(ir.DataType.FLOAT, [2, 16]), ts(ir.DataType.FLOAT)],
                {"output_dtype": ir.AttrInt64("output_dtype", 0)},
                opset_version=28,
            ),
            [ts(ir.DataType.UINT8, [2, 16])],
        )

    def test_untyped_zero_point_does_not_imply_uint8(self):
        self.assertEqual(
            run_shape_inference(
                "",
                "QuantizeLinear",
                [ts(ir.DataType.FLOAT, [2, 16]), ts(ir.DataType.FLOAT), ts()],
                opset_version=28,
            ),
            [ts(None, [2, 16])],
        )

    @parameterized.parameterized.expand([("override",), ("skip",)])
    def test_mismatched_output_dtype(self, policy):
        expected = (
            self.assertRaises(ShapeInferenceError)
            if policy == "override"
            else self.assertLogs("onnx_shape_inference._context", level="WARNING")
        )
        with expected:
            run_shape_inference(
                "",
                "QuantizeLinear",
                [ts(ir.DataType.FLOAT, [2, 16]), ts(ir.DataType.FLOAT), ts(ir.DataType.INT8)],
                {"output_dtype": ir.AttrInt64("output_dtype", ir.DataType.UINT8.value)},
                opset_version=28,
                policy=policy,
            )
