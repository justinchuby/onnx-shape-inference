# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic schema-driven fuzz graph generation."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx_ir as ir
import parameterized

from onnx_shape_inference import _registry
from tests.fuzz import _generator
from tests.fuzz._oracles import CrashOracle
from tests.fuzz._types import DATA_DEPENDENT_OPS


def _seed_for_planner(op_type: str) -> int:
    return sorted(_generator._op_planners).index(("", op_type))


def _first_seed_emitting(op_type: str, limit: int = 300) -> int:
    for seed in range(limit):
        case = _generator.generate(seed)
        if any(selected[1] == op_type for selected in case.selected_ops):
            return seed
    raise AssertionError(f"No seed below {limit} emitted {op_type}")


def _proto(seed: int) -> onnx.ModelProto:
    return ir.serde.serialize_model(_generator.generate(seed).model)


def _constant(value: ir.Value | None) -> np.ndarray:
    assert value is not None
    tensor = ir.convenience.get_const_tensor(value)
    assert tensor is not None
    return tensor.numpy()


def _node_by_name(case, name: str) -> ir.Node:
    return next(node for node in case.model.graph if node.name == name)


def _is_unidirectional_broadcastable(secondary: ir.Shape, primary: ir.Shape) -> bool:
    """Return True when ``secondary`` can broadcast *into* ``primary``.

    A secondary tensor is unidirectionally broadcastable to a primary tensor
    when its rank does not exceed the primary rank and each trailing-aligned
    dimension is either ``1`` or equal to the corresponding primary dimension.
    """
    if secondary.rank() > primary.rank():
        return False
    offset = primary.rank() - secondary.rank()
    for index in range(secondary.rank()):
        sec_dim = secondary[index]
        prim_dim = primary[offset + index]
        if sec_dim == 1:
            continue
        if sec_dim != prim_dim:
            return False
    return True


def _isolated_planner_model(
    op_type: str, opset: int, seed: int
) -> tuple[onnx.ModelProto, ir.Node]:
    """Build a single-node model from a planner for isolated validation."""
    generator = _generator._Generator(seed)
    generator.opset = opset
    generator._seed_port_pool()
    ports = generator._add_operator(("", op_type))
    node = generator.nodes[-1]
    graph = ir.Graph(
        inputs=generator.graph_inputs,
        outputs=[port.value for port in ports],
        nodes=[node],
        initializers=generator.initializers,
        opset_imports={"": opset},
        name="planner_isolated",
    )
    model = ir.Model(graph, ir_version=10)
    return ir.serde.serialize_model(model), node


class GeneratorTest(unittest.TestCase):
    @parameterized.parameterized.expand([(seed,) for seed in range(32)])
    def test_generated_model_passes_checker_and_roundtrips(self, seed):
        proto = _proto(seed)
        onnx.checker.check_model(proto)

        roundtripped = ir.serde.serialize_model(ir.serde.deserialize_model(proto))
        onnx.checker.check_model(roundtripped)
        self.assertEqual(
            proto.SerializeToString(deterministic=True),
            roundtripped.SerializeToString(deterministic=True),
        )

    @parameterized.parameterized.expand([(0,), (1,), (7,), (42,), (999,)])
    def test_generation_is_byte_deterministic(self, seed):
        first = _proto(seed).SerializeToString(deterministic=True)
        second = _proto(seed).SerializeToString(deterministic=True)
        self.assertEqual(first, second)

    def test_selection_is_registry_driven_and_covers_broad_slice(self):
        _registry.registry.collect()
        supported = {
            (domain, op_type): versions
            for domain, op_type, versions in _registry.registry.iter_supported()
        }
        observed: set[tuple[str, str]] = set()
        observed_boundaries: set[int] = set()
        observed_opsets: set[int] = set()
        for seed in range(100):
            case = _generator.generate(seed)
            observed_opsets.add(case.opset_imports[""])
            for domain, op_type, version in case.selected_ops:
                observed.add((domain, op_type))
                observed_boundaries.add(version)
                self.assertIn((domain, op_type), supported)
                self.assertIn(version, supported[(domain, op_type)])
                self.assertLessEqual(version, case.opset_imports[domain])

        self.assertGreaterEqual(len(observed), 100)
        self.assertGreaterEqual(len(observed_boundaries), 10)
        self.assertGreaterEqual(len(observed_opsets), 10)

    @parameterized.parameterized.expand(
        [
            (
                0,
                "shape_slice_concat_reshape",
                ("Shape", "Slice", "Concat", "Reshape"),
            ),
            (
                1,
                "shape_gather_unsqueeze_concat_reshape",
                ("Shape", "Gather", "Unsqueeze", "Concat", "Reshape"),
            ),
            (2, "constant_of_shape", ("ConstantOfShape",)),
            (3, "shape_slice_range", ("Shape", "Slice", "Squeeze", "Range")),
        ]
    )
    def test_sym_data_templates_are_first_class_recipes(self, seed, template, expected):
        case = _generator.generate(seed)
        self.assertEqual(case.metadata["template"], template)
        op_types = tuple(node.op_type for node in case.model.graph)
        cursor = 0
        for op_type in op_types:
            if cursor < len(expected) and op_type == expected[cursor]:
                cursor += 1
        self.assertEqual(cursor, len(expected), op_types)

    @parameterized.parameterized.expand(
        [(seed, op_type) for seed, (_, op_type) in enumerate(sorted(_generator._op_planners))]
    )
    def test_constant_input_planners_emit_coherent_values(self, seed, expected_op_type):
        case = _generator.generate(seed)
        self.assertEqual(case.metadata["planner_op"], ("", expected_op_type))
        node = _node_by_name(case, case.metadata["planner_node"])
        self.assertEqual(node.op_type, expected_op_type)

        if expected_op_type == "ConstantOfShape":
            shape = _constant(node.inputs[0])
            self.assertEqual(shape.ndim, 1)
            self.assertTrue(np.all(shape >= 0))
        elif expected_op_type == "Expand":
            data, shape = node.inputs
            self.assertGreaterEqual(_constant(shape).size, data.shape.rank())
        elif expected_op_type == "OneHot":
            _, depth, values = node.inputs
            self.assertGreater(int(_constant(depth)), 0)
            self.assertEqual(_constant(values).shape, (2,))
        elif expected_op_type == "Pad":
            rank = node.inputs[0].shape.rank()
            if len(node.inputs) > 1:
                self.assertEqual(_constant(node.inputs[1]).size, 2 * rank)
            else:
                pads = node.attributes.get("pads") or node.attributes.get("paddings")
                self.assertIsNotNone(pads)
                self.assertEqual(len(pads.as_ints()), 2 * rank)
        elif expected_op_type == "Range":
            start, limit, delta = (_constant(value) for value in node.inputs)
            self.assertEqual(start.ndim, 0)
            self.assertEqual(limit.ndim, 0)
            self.assertNotEqual(float(delta), 0.0)
        elif expected_op_type == "Reshape":
            data, shape = node.inputs
            target = _constant(shape)
            self.assertLessEqual(np.count_nonzero(target == -1), 1)
            self.assertEqual(target.ndim, 1)
            input_numel = int(np.prod(list(data.shape)))
            if -1 not in target:
                self.assertEqual(int(np.prod(target)), input_numel)
        elif expected_op_type == "Resize":
            data = node.inputs[0]
            resize_arg = node.inputs[-1]
            self.assertEqual(_constant(resize_arg).size, data.shape.rank())
        elif expected_op_type == "Slice":
            lengths = [_constant(value).size for value in node.inputs[1:]]
            self.assertEqual(len(set(lengths)), 1)
        elif expected_op_type == "Tile":
            data, repeats = node.inputs
            repeat_values = _constant(repeats)
            self.assertEqual(repeat_values.size, data.shape.rank())
            self.assertTrue(np.all(repeat_values >= 1))
        elif expected_op_type == "TopK":
            data, k = node.inputs
            axis = node.attributes["axis"].as_int()
            k_value = int(_constant(k)[0])
            self.assertGreaterEqual(k_value, 1)
            self.assertLessEqual(k_value, data.shape[axis])
        elif expected_op_type == "Conv":
            x, weight = node.inputs
            kernel = _constant(weight)
            self.assertEqual(kernel.ndim, 4)
            self.assertEqual(x.shape.rank(), 4)
            self.assertEqual(int(x.shape[1]), kernel.shape[1])
            self.assertEqual(list(node.attributes["kernel_shape"].as_ints()), [3, 3])
        elif expected_op_type in ("MaxPool", "AveragePool"):
            (x,) = node.inputs
            self.assertEqual(x.shape.rank(), 4)
            self.assertEqual(list(node.attributes["kernel_shape"].as_ints()), [2, 2])
        elif expected_op_type == "MatMul":
            a, b = node.inputs
            weight = _constant(b)
            self.assertEqual(a.shape.rank(), 2)
            self.assertEqual(weight.ndim, 2)
            self.assertEqual(int(a.shape[1]), weight.shape[0])
        elif expected_op_type == "Gemm":
            a, b = node.inputs[0], node.inputs[1]
            weight = _constant(b)
            self.assertEqual(a.shape.rank(), 2)
            self.assertEqual(int(a.shape[1]), weight.shape[0])
        elif expected_op_type == "Gather":
            data, indices = node.inputs
            axis = node.attributes["axis"].as_int()
            values = _constant(indices)
            extent = int(data.shape[axis])
            self.assertTrue(np.all(values < extent))
            self.assertTrue(np.all(values >= -extent))
        elif expected_op_type == "Concat":
            first, second = node.inputs
            axis = node.attributes["axis"].as_int()
            self.assertEqual(first.shape.rank(), second.shape.rank())
            for index in range(first.shape.rank()):
                if index != axis:
                    self.assertEqual(int(first.shape[index]), int(second.shape[index]))
        elif expected_op_type == "Split":
            self.assertEqual(len(node.outputs), 2)
            self.assertIn("axis", node.attributes)
        elif expected_op_type in ("PRelu", "LayerNormalization", "RMSNormalization"):
            primary = node.inputs[0]
            self.assertGreaterEqual(primary.shape.rank(), 1)
            for secondary in node.inputs[1:]:
                if secondary is None:
                    continue
                self.assertTrue(
                    _is_unidirectional_broadcastable(secondary.shape, primary.shape),
                    f"{secondary.shape} not broadcastable to {primary.shape}",
                )
                self.assertEqual(secondary.dtype, primary.dtype)
        else:
            self.fail(f"Missing planner assertion for {expected_op_type}")

        onnx.checker.check_model(ir.serde.serialize_model(case.model))

    def test_input_pool_mixes_shared_symbols_scalars_and_vectors(self):
        case = _generator.generate(4)
        input_shapes = [value.shape for value in case.model.graph.inputs]
        self.assertTrue(any(shape.rank() == 0 for shape in input_shapes))
        self.assertTrue(any(shape.rank() == 1 for shape in input_shapes))
        first_dims = [
            str(shape[0])
            for shape in input_shapes
            if shape.rank() > 0 and isinstance(shape[0], ir.SymbolicDim)
        ]
        self.assertLess(len(set(first_dims)), len(first_dims))
        self.assertEqual(tuple(case.symbolic_dims), tuple(sorted(case.symbolic_dims)))
        self.assertEqual(set(case.symbolic_dims), set(case.symbol_constraints))

    @parameterized.parameterized.expand(
        [
            ("DepthToSpace", (4,)),
            ("SpaceToDepth", (2, 2)),
        ]
    )
    def test_semantic_planners_record_divisibility_constraints(self, op_type, divisors):
        generator = _generator._Generator(0)
        generator._seed_port_pool()
        generator._add_operator(("", op_type))

        recorded = sorted(
            constraint.divisible_by
            for constraint in generator.symbol_constraints.values()
            if constraint.divisible_by > 1
        )
        self.assertEqual(recorded, sorted(divisors))

    @parameterized.parameterized.expand(
        [
            (sorted(_generator._op_planners).index(("", "Range")), "Range"),
            (sorted(_generator._op_planners).index(("", "TopK")), "TopK"),
        ]
    )
    def test_data_dependent_planner_outputs_are_marked(self, seed, op_type):
        case = _generator.generate(seed)
        node = _node_by_name(case, case.metadata["planner_node"])
        self.assertEqual(node.op_type, op_type)
        self.assertIn(("", op_type), DATA_DEPENDENT_OPS)
        self.assertTrue(
            {output.name for output in node.outputs}.issubset(case.data_dependent_values)
        )


class ExpandedCoverageTest(unittest.TestCase):
    """Tests for the planner, control-flow, dtype, and non-degenerate widening."""

    @parameterized.parameterized.expand(
        [
            ("Conv",),
            ("MatMul",),
            ("Gemm",),
            ("Gather",),
            ("Concat",),
            ("Split",),
            ("MaxPool",),
            ("AveragePool",),
        ]
    )
    def test_new_planners_generate_and_pass_crash_oracle(self, op_type):
        seed = _seed_for_planner(op_type)
        case = _generator.generate(seed)
        self.assertEqual(case.metadata["planner_op"], ("", op_type))
        onnx.checker.check_model(ir.serde.serialize_model(case.model))
        result = CrashOracle().check(case)
        self.assertEqual(result.status, "PASS", result.reason)

    @parameterized.parameterized.expand([("If",), ("Loop",), ("Scan",)])
    def test_control_flow_emitted_during_generation(self, op_type):
        seed = _first_seed_emitting(op_type)
        case = _generator.generate(seed)
        self.assertTrue(
            any(node.op_type == op_type for node in case.model.graph),
            f"seed {seed} did not emit {op_type}",
        )
        self.assertIn(("", op_type), {op[:2] for op in case.selected_ops})

    @parameterized.parameterized.expand([("_add_if",), ("_add_loop",), ("_add_scan",)])
    def test_control_flow_subgraphs_are_valid(self, builder):
        import onnx_shape_inference as osi

        generator = _generator._Generator(0)
        generator.opset = 18
        getattr(generator, builder)()
        node = generator.nodes[-1]
        graph_attrs = [
            attr for attr in node.attributes.values() if attr.type == ir.AttributeType.GRAPH
        ]
        self.assertTrue(graph_attrs, "control-flow node needs subgraph attributes")
        for attr in graph_attrs:
            self.assertGreater(len(attr.as_graph()), 0)
        output = generator.ports[-1].value
        graph = ir.Graph(
            inputs=[],
            outputs=[output],
            nodes=generator.nodes,
            initializers=generator.initializers,
            opset_imports={"": 18},
            name="cf_isolated",
        )
        model = ir.Model(graph, ir_version=10)
        proto = ir.serde.serialize_model(model)
        # full_check validates the nested subgraphs via ONNX shape inference.
        onnx.checker.check_model(proto, full_check=True)
        osi.infer_symbolic_shapes(ir.serde.deserialize_model(proto))

    def test_split_planner_emits_two_outputs(self):
        case = _generator.generate(_seed_for_planner("Split"))
        node = _node_by_name(case, case.metadata["planner_node"])
        self.assertEqual(node.op_type, "Split")
        self.assertEqual(len(node.outputs), 2)

    @parameterized.parameterized.expand(
        [
            ("PRelu", 16),
            ("PRelu", 22),
            ("LayerNormalization", 17),
            ("RMSNormalization", 23),
        ]
    )
    def test_unidirectional_planners_emit_valid_broadcasts(self, op_type, opset):
        # Every secondary input a unidirectional-broadcast planner emits must be
        # broadcastable *into* the primary tensor, and the isolated single-op
        # model must pass the strict ONNX checker across supporting opsets.
        for seed in range(40):
            proto, node = _isolated_planner_model(op_type, opset, seed)
            onnx.checker.check_model(proto, full_check=True)
            primary = node.inputs[0]
            for secondary in node.inputs[1:]:
                if secondary is None:
                    continue
                self.assertTrue(
                    _is_unidirectional_broadcastable(secondary.shape, primary.shape),
                    f"seed {seed}: {secondary.shape} not broadcastable to {primary.shape}",
                )

    @parameterized.parameterized.expand(
        [
            ((1, 1), 16),
            ((2, 3), 16),
            ((2, 3, 4), 22),
        ]
    )
    def test_prelu_inference_agrees_with_runtime(self, x_shape, opset):
        import onnxruntime as ort

        import onnx_shape_inference as osi

        generator = _generator._Generator(0)
        x_shape = tuple(x_shape)
        slope_shape = _generator._unidirectional_broadcast_shape(generator, x_shape)
        x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(x_shape))
        slope = onnx.helper.make_tensor_value_info(
            "slope", onnx.TensorProto.FLOAT, list(slope_shape)
        )
        y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, list(x_shape))
        node = onnx.helper.make_node("PRelu", ["x", "slope"], ["y"])
        graph = onnx.helper.make_graph([node], "prelu", [x, slope], [y])
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
        )
        onnx.checker.check_model(model, full_check=True)

        inferred = osi.infer_symbolic_shapes(ir.serde.deserialize_model(model))
        inferred_shape = tuple(int(dim) for dim in inferred.graph.outputs[0].shape)

        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(0)
        outputs = session.run(
            None,
            {
                "x": rng.standard_normal(x_shape).astype(np.float32),
                "slope": rng.standard_normal(slope_shape).astype(np.float32),
            },
        )
        self.assertEqual(inferred_shape, tuple(outputs[0].shape))
        self.assertEqual(inferred_shape, x_shape)

    def test_random_shape_defaults_are_non_degenerate(self):
        generator = _generator._Generator(0)
        for _ in range(200):
            shape = generator._random_shape()
            self.assertGreaterEqual(len(shape), 1, "default rank must be >= 1")
            for dim in shape:
                if isinstance(dim, int):
                    self.assertGreaterEqual(dim, 1, "default extents must be >= 1")

    def test_random_shape_allow_zero_reintroduces_empty_extents(self):
        generator = _generator._Generator(3)
        saw_zero = any(
            dim == 0
            for _ in range(400)
            for dim in generator._random_shape(allow_zero=True)
            if isinstance(dim, int)
        )
        self.assertTrue(saw_zero)

    def test_default_planned_inputs_avoid_zero_length_dims(self):
        # Graph inputs created by the generic planner should never carry a
        # zero-length static extent (the intentional bool scalar in the seed
        # pool is rank-0 but not zero-length).
        for seed in range(120):
            case = _generator.generate(seed)
            for value in case.model.graph.inputs:
                if value.shape is None:
                    continue
                for dim in value.shape:
                    if isinstance(dim, int):
                        self.assertNotEqual(dim, 0, f"seed {seed} produced empty dim")

    def test_seed_pool_exercises_small_integer_dtypes(self):
        generator = _generator._Generator(0)
        generator._seed_port_pool()
        dtypes = {port.dtype for port in generator.ports}
        for dtype in (
            ir.DataType.INT8,
            ir.DataType.INT16,
            ir.DataType.UINT16,
        ):
            self.assertIn(dtype, dtypes)


if __name__ == "__main__":
    unittest.main()
