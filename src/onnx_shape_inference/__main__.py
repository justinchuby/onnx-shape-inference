# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for onnx-shape-inference.

Usage::

    python -m onnx_shape_inference model.onnx
    python -m onnx_shape_inference model.onnx -o model_inferred.onnx
"""

from __future__ import annotations

import argparse

import onnx_ir as ir

from onnx_shape_inference import infer_symbolic_shapes


def _count_shapes(model: ir.Model) -> int:
    """Count the number of values with shapes in the model."""
    count = 0
    for node in model.graph.all_nodes():
        for output in node.outputs:
            if output.shape is not None:
                count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    """Run shape inference on an ONNX model."""
    parser = argparse.ArgumentParser(
        prog="onnx_shape_inference",
        description="Symbolic shape inference for ONNX models.",
    )
    parser.add_argument("model", help="Path to the input ONNX model.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the inferred model. If not provided, the model is not saved.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Save the inferred model back to the input path.",
    )
    parser.add_argument(
        "--policy",
        choices=["strict", "refine"],
        default="refine",
        help="Shape merge policy (default: refine).",
    )
    args = parser.parse_args(argv)

    if args.output and args.in_place:
        parser.error("--output and --in-place are mutually exclusive.")

    model = ir.load(args.model)
    shapes_before = _count_shapes(model)

    infer_symbolic_shapes(model, policy=args.policy)

    shapes_after = _count_shapes(model)
    new_shapes = shapes_after - shapes_before
    print(f"New shapes created: {new_shapes}")

    save_path = args.model if args.in_place else args.output
    if save_path:
        ir.save(model, save_path)
        print(f"Saved inferred model to {save_path}")


if __name__ == "__main__":
    main()
