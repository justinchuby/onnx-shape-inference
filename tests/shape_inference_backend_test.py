# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test shape inference against ONNX backend node tests.

For each backend test model, the test:
1. Generates the model and input tensors using ONNX's backend test loader.
2. Injects the input tensors as graph initializers so that constant inputs
   (e.g. ``axes`` for Reduce ops) are available to shape inference.
3. Saves the expected output dtype and shape from the model proto.
4. Clears the output type/shape information.
5. Runs symbolic shape inference.
6. Asserts the inferred dtype and shape match the expected values.

When the inferred shape is ``None`` or symbolic where concrete is expected,
the test fails unless explicitly added to a skip list.
"""

from __future__ import annotations

import unittest

import onnx
import onnx.backend.test.case.test_case
import onnx.backend.test.loader
import onnx.numpy_helper
import onnx_ir as ir
import parameterized

from onnx_shape_inference import infer_symbolic_shapes

# Node cases are generated in memory; newer ONNX wheels do not bundle .pb files.
_test_args = [
    (case.name, case)
    for case in sorted(
        onnx.backend.test.loader.load_model_tests(kind="node"), key=lambda case: case.name
    )
]

# Tests where shape inference produces incorrect results due to data-dependent
# shapes (e.g. operator inputs like split sizes or axes are graph inputs, not
# constants). Each entry should be investigated individually.

# Expanded multi-op models where shape info is lost through If/Loop/complex
# subgraphs or where intermediate Constant/Shape ops don't preserve shapes.
_SKIP_EXPANDED_MODELS: set[str] = set()

_ALL_SKIPS = _SKIP_EXPANDED_MODELS


def _inject_inputs_as_initializers(
    proto: onnx.ModelProto,
    input_tensors: list[onnx.TensorProto],
) -> None:
    """Add test input tensors as graph initializers.

    This makes constant inputs (like ``axes`` for Reduce ops) visible to
    shape inference via ``get_const_tensor``.
    """
    existing_names = {init.name for init in proto.graph.initializer}
    input_names = [inp.name for inp in proto.graph.input]
    for tensor in input_tensors:
        name = tensor.name
        if not name:
            continue
        if name in existing_names:
            continue
        if name in input_names:
            proto.graph.initializer.append(tensor)


def _shapes_compatible(
    expected: ir.Shape,
    inferred: ir.Shape,
) -> tuple[bool, bool]:
    """Check that the inferred shape is compatible with the expected shape.

    Returns:
        (compatible, has_symbolic): compatible is True if shapes match,
        has_symbolic is True if inferred has symbolic dims where expected
        has concrete ones.
    """
    if len(expected) != len(inferred):
        return False, False
    has_symbolic = False
    for exp_dim, inf_dim in zip(expected, inferred):
        exp_val = exp_dim.value if isinstance(exp_dim, ir.SymbolicDim) else exp_dim
        inf_val = inf_dim.value if isinstance(inf_dim, ir.SymbolicDim) else inf_dim
        if isinstance(exp_val, int) and isinstance(inf_val, int):
            if exp_val != inf_val:
                return False, False
        elif isinstance(exp_val, int) and not isinstance(inf_val, int):
            has_symbolic = True
    return True, has_symbolic


class ShapeInferenceBackendTest(unittest.TestCase):
    @parameterized.parameterized.expand(_test_args)
    def test_shape_inference_matches_expected(
        self, test_name: str, case: onnx.backend.test.case.test_case.TestCase
    ) -> None:
        if test_name in _ALL_SKIPS:
            self.skipTest("See skip list for reason")

        self.assertIsNotNone(case.model)
        self.assertTrue(case.data_sets)
        # Input injection must not mutate the loader's shared test case.
        proto = onnx.ModelProto()
        proto.CopyFrom(case.model)

        # Inject test inputs as initializers so constant inputs are available
        input_tensors = [
            value
            if isinstance(value, onnx.TensorProto)
            else onnx.numpy_helper.from_array(value, input_info.name)
            for input_info, value in zip(proto.graph.input, case.data_sets[0][0])
            if input_info.type.HasField("tensor_type")
        ]
        _inject_inputs_as_initializers(proto, input_tensors)

        model = ir.serde.deserialize_model(proto)

        # Save expected output dtype and shape, then clear them
        expected: dict[
            str, tuple[ir.DataType | None, ir.Shape | None, ir.TypeProtocol | None]
        ] = {}
        for out in model.graph.outputs:
            expected[out.name] = (out.dtype, out.shape, out.type)
            out.type = None

        infer_symbolic_shapes(model)

        for out in model.graph.outputs:
            exp_dtype, exp_shape, exp_type = expected[out.name]

            # For sequence/optional types, compare type objects directly
            if exp_type is not None and not isinstance(exp_type, ir.TensorType):
                self.assertEqual(
                    out.type,
                    exp_type,
                    f"Output '{out.name}': type mismatch",
                )
                continue

            # Check dtype
            if exp_dtype is not None:
                self.assertIsNotNone(
                    out.dtype,
                    f"Output '{out.name}': dtype is None, expected {exp_dtype}",
                )
                self.assertEqual(
                    out.dtype,
                    exp_dtype,
                    f"Output '{out.name}': dtype mismatch",
                )

            # Check shape
            if exp_shape is not None:
                self.assertIsNotNone(
                    out.shape,
                    f"Output '{out.name}': shape is None, expected {exp_shape}",
                )
                compatible, has_symbolic = _shapes_compatible(exp_shape, out.shape)
                if has_symbolic:
                    self.fail(
                        f"Output '{out.name}': inferred symbolic dims where "
                        f"concrete expected: expected={exp_shape}, got={out.shape}. "
                        f"Add to _SKIP_EXPANDED_MODELS if this is expected."
                    )
                self.assertTrue(
                    compatible,
                    f"Output '{out.name}': shape mismatch: "
                    f"expected={exp_shape}, got={out.shape}",
                )


if __name__ == "__main__":
    unittest.main()
