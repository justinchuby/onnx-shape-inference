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
from onnx_shape_inference._fuzz import _generator
from onnx_shape_inference._fuzz._types import DATA_DEPENDENT_OPS


def _proto(seed: int) -> onnx.ModelProto:
    return ir.serde.serialize_model(_generator.generate(seed).model)


def _constant(value: ir.Value | None) -> np.ndarray:
    assert value is not None
    tensor = ir.convenience.get_const_tensor(value)
    assert tensor is not None
    return tensor.numpy()


def _node_by_name(case, name: str) -> ir.Node:
    return next(node for node in case.model.graph if node.name == name)


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

    @parameterized.parameterized.expand([(4, "Range"), (9, "TopK")])
    def test_data_dependent_planner_outputs_are_marked(self, seed, op_type):
        case = _generator.generate(seed)
        node = _node_by_name(case, case.metadata["planner_node"])
        self.assertEqual(node.op_type, op_type)
        self.assertIn(("", op_type), DATA_DEPENDENT_OPS)
        self.assertTrue(
            {output.name for output in node.outputs}.issubset(case.data_dependent_values)
        )


if __name__ == "__main__":
    unittest.main()
