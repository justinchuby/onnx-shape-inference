# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fuzz-oracle outcomes and symbolic binding."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from unittest import mock

import onnx_ir as ir

from tests.fuzz._binding import bind_symbols, materialize_model
from tests.fuzz._harness import FuzzHarness
from tests.fuzz._oracles import (
    CrashOracle,
    DifferentialOracle,
    SimplificationOracle,
    SoundnessOracle,
)
from tests.fuzz._types import FuzzCase, OracleResult, SymbolConstraint


def _case(*, symbolic: bool = False) -> FuzzCase:
    shape = ["N", "M"] if symbolic else [2, 3]
    x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
    relu = ir.Node("", "Relu", [x], num_outputs=1)
    relu.outputs[0].name = "Y"
    model = ir.Model(
        ir.Graph([x], list(relu.outputs), nodes=[relu], opset_imports={"": 21}),
        ir_version=10,
    )
    return FuzzCase(model=model, seed=7, opset_imports={"": 21})


class CrashOracleTest(unittest.TestCase):
    def test_correct_graph_passes_and_is_idempotent(self):
        self.assertEqual(CrashOracle().check(_case()).status, "PASS")

    def test_malformed_mode_requires_op_usage_error(self):
        x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        malformed = ir.Node("", "Reshape", [x], num_outputs=1)
        model = ir.Model(
            ir.Graph([x], list(malformed.outputs), nodes=[malformed], opset_imports={"": 21}),
            ir_version=10,
        )
        case = FuzzCase(model=model, seed=0, opset_imports={"": 21})
        self.assertEqual(CrashOracle(malformed=True).check(case).status, "PASS")


class DifferentialOracleTest(unittest.TestCase):
    def test_correct_graph_passes(self):
        result = DifferentialOracle().check(_case())
        self.assertEqual(result.status, "PASS")


class BindingTest(unittest.TestCase):
    def test_shared_symbols_receive_one_distinct_prime_binding(self):
        case = _case(symbolic=True)
        bindings = bind_symbols(case)
        self.assertEqual(set(bindings), {"M", "N"})
        self.assertNotEqual(bindings["M"], bindings["N"])
        concrete = materialize_model(case, bindings)
        self.assertEqual(
            next(iter(concrete.graph.inputs)).shape,
            ir.Shape([bindings["N"], bindings["M"]]),
        )

    def test_divisible_constraint_uses_a_value_inside_its_bounds(self):
        case = _case(symbolic=True)
        case.symbol_constraints["N"] = SymbolConstraint(
            minimum=3,
            maximum=6,
            divisible_by=4,
        )
        bindings = bind_symbols(case)
        self.assertEqual(bindings["N"], 4)


class SimplificationOracleTest(unittest.TestCase):
    def test_equivalent_expression_passes(self):
        case = _case(symbolic=True)
        case.pre_simplify_dims[("Y", 0)] = "N"
        self.assertEqual(SimplificationOracle(samples=4).check(case).status, "PASS")

    def test_wrong_reference_fails(self):
        case = _case(symbolic=True)
        case.pre_simplify_dims[("Y", 0)] = "N + 1"
        result = SimplificationOracle(samples=4).check(case)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.value_name, "Y")


@unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime is optional")
class SoundnessOracleTest(unittest.TestCase):
    def test_correct_inference_passes(self):
        self.assertEqual(SoundnessOracle().check(_case()).status, "PASS")

    def test_unmapped_intermediate_dtype_does_not_false_fail(self):
        x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        cast = ir.Node(
            "",
            "Cast",
            [x],
            num_outputs=1,
            attributes={"to": ir.Attr("to", ir.AttributeType.INT, int(ir.DataType.FLOAT16))},
        )
        cast.outputs[0].name = "Y"
        model = ir.Model(
            ir.Graph([x], list(cast.outputs), nodes=[cast], opset_imports={"": 21}),
            ir_version=10,
        )
        case = FuzzCase(model=model, seed=0, opset_imports={"": 21})
        self.assertEqual(SoundnessOracle().check(case).status, "PASS")

    def test_wrong_inferred_shape_fails_against_runtime(self):
        case = _case()
        wrong = copy.deepcopy(case.model)
        wrong_output = next(iter(wrong.graph.outputs))
        wrong_output.shape = ir.Shape([99, 3])
        with mock.patch(
            "tests.fuzz._oracles._infer_ours",
            return_value=wrong,
        ):
            result = SoundnessOracle().check(case)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.value_name, "Y")

    def test_isolates_unsupported_node_failures(self):
        x = ir.Value(name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        relu = ir.Node("", "Relu", [x], num_outputs=1)
        relu.outputs[0].name = "relu"
        add = ir.Node("", "Add", [relu.outputs[0], x], num_outputs=1)
        add.outputs[0].name = "add"
        empty_shape = ir.Value(
            name="empty_shape",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([0]),
            const_value=ir.tensor([], dtype=ir.DataType.INT64),
        )
        invalid = ir.Node("", "CenterCropPad", [x, empty_shape], num_outputs=1)
        invalid.outputs[0].name = "invalid"
        model = ir.Model(
            ir.Graph(
                [x],
                [add.outputs[0], invalid.outputs[0]],
                nodes=[relu, add, invalid],
                initializers=[empty_shape],
                opset_imports={"": 18},
                name="mixed_node_soundness",
            ),
            ir_version=10,
        )
        result = SoundnessOracle().check(FuzzCase(model=model, seed=0, opset_imports={"": 18}))
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.details["nodes"], {"pass": 2, "skip": 1})


class HarnessTest(unittest.TestCase):
    def test_failure_contains_seed_and_oracle_name(self):
        class FailingOracle:
            name = "failing"

            def applicable(self, case):
                return True

            def check(self, case):
                return OracleResult.failed(
                    "failing", "deliberate", value_name="Y", kind="dtype"
                )

        harness = FuzzHarness(
            lambda seed: FuzzCase(model=_case().model, seed=seed, opset_imports={"": 21}),
            [FailingOracle()],
        )
        with self.assertRaisesRegex(AssertionError, r"oracle=failing seed=11"):
            harness.run([11])


if __name__ == "__main__":
    unittest.main()
