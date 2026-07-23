# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic schema-driven graph generation for shape-inference fuzzing."""

from __future__ import annotations

__all__ = ["generate"]

import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import onnx
import onnx_ir as ir

from onnx_shape_inference import _registry
from tests.fuzz._types import (
    DATA_DEPENDENT_OPS,
    FuzzCase,
    SymbolConstraint,
)

_OpKey: TypeAlias = tuple[str, str]
_Dim: TypeAlias = int | str | None
_Shape: TypeAlias = tuple[_Dim, ...]

_TYPE_NAME_TO_DTYPE = {
    "bool": ir.DataType.BOOL,
    "double": ir.DataType.DOUBLE,
    "float": ir.DataType.FLOAT,
    "float16": ir.DataType.FLOAT16,
    "int8": ir.DataType.INT8,
    "int16": ir.DataType.INT16,
    "int32": ir.DataType.INT32,
    "int64": ir.DataType.INT64,
    "uint8": ir.DataType.UINT8,
    "uint16": ir.DataType.UINT16,
    "uint32": ir.DataType.UINT32,
    "uint64": ir.DataType.UINT64,
}
_DTYPE_TO_NUMPY = {
    ir.DataType.BOOL: np.bool_,
    ir.DataType.DOUBLE: np.float64,
    ir.DataType.FLOAT: np.float32,
    ir.DataType.FLOAT16: np.float16,
    ir.DataType.INT8: np.int8,
    ir.DataType.INT16: np.int16,
    ir.DataType.INT32: np.int32,
    ir.DataType.INT64: np.int64,
    ir.DataType.UINT8: np.uint8,
    ir.DataType.UINT16: np.uint16,
    ir.DataType.UINT32: np.uint32,
    ir.DataType.UINT64: np.uint64,
}
_PREFERRED_DTYPES = (
    ir.DataType.FLOAT,
    ir.DataType.INT64,
    ir.DataType.INT32,
    ir.DataType.BOOL,
    ir.DataType.DOUBLE,
    ir.DataType.FLOAT16,
    ir.DataType.INT8,
    ir.DataType.UINT8,
    ir.DataType.INT16,
    ir.DataType.UINT16,
    ir.DataType.UINT32,
    ir.DataType.UINT64,
)
_TENSOR_TYPE = re.compile(r"^tensor\(([^)]+)\)$")
_SUPPORTED_REQUIRED_ATTRS = frozenset(
    {
        onnx.defs.OpSchema.AttrType.FLOAT,
        onnx.defs.OpSchema.AttrType.INT,
        onnx.defs.OpSchema.AttrType.STRING,
        onnx.defs.OpSchema.AttrType.TENSOR,
        onnx.defs.OpSchema.AttrType.FLOATS,
        onnx.defs.OpSchema.AttrType.INTS,
        onnx.defs.OpSchema.AttrType.STRINGS,
        onnx.defs.OpSchema.AttrType.TENSORS,
    }
)
_GENERIC_EXCLUDED_OPS = frozenset(
    {
        "ArgMin",
        "ArgMax",
        "Attention",
        "Constant",
        "ConvInteger",
        "ConvTranspose",
        "DeformConv",
        "Einsum",
        "GRU",
        "GatherND",
        "GlobalAveragePool",
        "GlobalLpPool",
        "GlobalMaxPool",
        "GridSample",
        "If",
        "LSTM",
        "LinearAttention",
        "Loop",
        "LpPool",
        "MatMulInteger",
        "MaxRoiPool",
        "MaxUnpool",
        "QLinearMatMul",
        "QLinearConv",
        "RNN",
        "RoiAlign",
        "Scan",
        "SequenceMap",
        "STFT",
        "TensorScatter",
    }
)
_TEMPLATE_NAMES = (
    "shape_slice_concat_reshape",
    "shape_gather_unsqueeze_concat_reshape",
    "constant_of_shape",
    "shape_slice_range",
    None,
)


class _PlanningError(ValueError):
    """Raised when a schema cannot be instantiated by the current generator."""


@dataclass
class _Port:
    value: ir.Value
    dtype: ir.DataType
    shape: _Shape | None
    data_dependent: bool = False


@dataclass
class _InputPlan:
    inputs: list[_Port | None]
    attributes: dict[str, ir.Attr] = field(default_factory=dict)
    output_shapes: tuple[_Shape | None, ...] = ()
    output_count: int | None = None


InputPlanner: TypeAlias = Callable[
    ["_Generator", onnx.defs.OpSchema, dict[str, ir.DataType]], _InputPlan
]


def _shape_product(shape: Sequence[int]) -> int:
    return math.prod(shape)


def _plan_reshape(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "data",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    concrete_shape = tuple(int(dim) for dim in data.shape)
    numel = _shape_product(concrete_shape)
    target = (-1,) if numel > 0 and generator.rng.randrange(2) else (numel,)
    shape = generator._constant_port("reshape_shape", target, ir.DataType.INT64)
    return _InputPlan([data, shape], output_shapes=((numel,),))


def _plan_slice(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "data",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    axis = generator.rng.randrange(len(data.shape))
    extent = int(data.shape[axis])
    index_dtype = generator._dtype_for_type("Tind", schema, bindings)
    starts = generator._constant_port("slice_starts", [0], index_dtype)
    ends = generator._constant_port("slice_ends", [extent], index_dtype)
    axes = generator._constant_port("slice_axes", [axis], index_dtype)
    steps = generator._constant_port("slice_steps", [1], index_dtype)
    return _InputPlan([data, starts, ends, axes, steps], output_shapes=(data.shape,))


def _plan_tile(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "input",
        bindings,
        concrete_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    repeats = [1] * len(data.shape)
    repeats[generator.rng.randrange(len(repeats))] = 2
    repeats_port = generator._constant_port("tile_repeats", repeats, ir.DataType.INT64)
    output_shape = tuple(int(dim) * repeat for dim, repeat in zip(data.shape, repeats))
    return _InputPlan([data, repeats_port], output_shapes=(output_shape,))


def _plan_expand(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "input",
        bindings,
        concrete_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    target = tuple(int(dim) for dim in data.shape)
    shape = generator._constant_port("expand_shape", target, ir.DataType.INT64)
    return _InputPlan([data, shape], output_shapes=(target,))


def _plan_range(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = generator._dtype_for_type("T", schema, bindings)
    start = generator._constant_port("range_start", 0, dtype)
    limit = generator._constant_port("range_limit", 3, dtype)
    delta = generator._constant_port("range_delta", 1, dtype)
    return _InputPlan([start, limit, delta], output_shapes=((3,),))


def _plan_topk(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "X",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    axis = generator.rng.randrange(len(data.shape))
    k = max(1, min(2, int(data.shape[axis])))
    k_port = generator._constant_port("topk_k", [k], ir.DataType.INT64)
    output_shape = list(data.shape)
    output_shape[axis] = k
    attributes = {"axis": ir.Attr("axis", ir.AttributeType.INT, axis)}
    return _InputPlan(
        [data, k_port],
        attributes=attributes,
        output_shapes=(tuple(output_shape), tuple(output_shape)),
    )


def _plan_pad(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "data",
        bindings,
        concrete_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    pads = [0] * (2 * len(data.shape))
    pads[-1] = 1
    output_shape = list(data.shape)
    output_shape[-1] = int(output_shape[-1]) + 1
    if any(formal.name == "pads" for formal in schema.inputs):
        pads_port = generator._constant_port("pad_pads", pads, ir.DataType.INT64)
        inputs: list[_Port | None] = [data, pads_port]
        attributes = {}
    else:
        inputs = [data]
        attr_name = "paddings" if "paddings" in schema.attributes else "pads"
        attributes = {
            attr_name: ir.Attr(attr_name, ir.AttributeType.INTS, tuple(pads)),
        }
    return _InputPlan(inputs, attributes, output_shapes=(tuple(output_shape),))


def _plan_one_hot(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    bindings["T1"] = ir.DataType.INT64
    bindings["T2"] = ir.DataType.INT64
    bindings["T3"] = ir.DataType.FLOAT
    indices = generator._port_for_name(
        schema,
        "indices",
        bindings,
        concrete_shape=True,
        min_rank=1,
    )
    assert indices.shape is not None
    depth = generator._constant_port("onehot_depth", 3, ir.DataType.INT64)
    values = generator._constant_port("onehot_values", [0.0, 1.0], ir.DataType.FLOAT)
    output_shape = (*indices.shape, 3)
    return _InputPlan([indices, depth, values], output_shapes=(output_shape,))


def _plan_constant_of_shape(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    del schema
    bindings["T1"] = ir.DataType.INT64
    bindings.setdefault("T2", ir.DataType.FLOAT)
    shape = generator._constant_port("constant_of_shape_input", [2, 3], ir.DataType.INT64)
    return _InputPlan([shape], output_shapes=((2, 3),))


def _plan_resize(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "X",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=2,
        max_rank=4,
    )
    assert data.shape is not None
    output_shape = [int(dim) for dim in data.shape]
    output_shape[-1] *= 2
    input_names = [formal.name for formal in schema.inputs]
    if input_names == ["X", "scales"]:
        scales = [1.0] * len(data.shape)
        scales[-1] = 2.0
        scales_port = generator._constant_port("resize_scales", scales, ir.DataType.FLOAT)
        inputs: list[_Port | None] = [data, scales_port]
    else:
        roi = generator._constant_port("resize_roi", [], ir.DataType.FLOAT)
        scales = generator._constant_port("resize_scales", [], ir.DataType.FLOAT)
        sizes = generator._constant_port("resize_sizes", output_shape, ir.DataType.INT64)
        inputs = [data, roi, scales, sizes]
    return _InputPlan(inputs, output_shapes=(tuple(output_shape),))


def _plan_depth_to_space(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = generator._dtype_for_type("T", schema, bindings)
    channels = generator._new_symbol()
    generator._require_divisible(channels, 4)
    data = generator._graph_input(dtype, (1, channels, 2, 3))
    attributes = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
    return _InputPlan([data], attributes)


def _plan_space_to_depth(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = generator._dtype_for_type("T", schema, bindings)
    height = generator._new_symbol()
    width = generator._new_symbol()
    generator._require_divisible(height, 2)
    generator._require_divisible(width, 2)
    data = generator._graph_input(dtype, (1, 3, height, width))
    attributes = {"blocksize": ir.Attr("blocksize", ir.AttributeType.INT, 2)}
    return _InputPlan([data], attributes)


def _ints_attr(name: str, values: Sequence[int]) -> ir.Attr:
    return ir.Attr(name, ir.AttributeType.INTS, tuple(int(v) for v in values))


def _int_attr(name: str, value: int) -> ir.Attr:
    return ir.Attr(name, ir.AttributeType.INT, int(value))


def _numeric_dtype(
    generator: _Generator,
    type_str: str,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
    *,
    prefer: Sequence[ir.DataType] = (ir.DataType.FLOAT,),
) -> ir.DataType:
    """Pick a dtype for ``type_str`` that we can also build initializers for.

    Prefers the supplied dtypes (defaulting to float) so operators that only run
    numerically under ONNX Runtime, e.g. pooling and convolution, receive a
    runnable type when the schema allows it.
    """
    if type_str in bindings:
        return bindings[type_str]
    candidates = generator._dtype_candidates(type_str, schema)
    if not candidates:
        raise _PlanningError(f"No supported tensor dtype for {schema.name}.{type_str}")
    for wanted in prefer:
        if wanted in candidates:
            bindings[type_str] = wanted
            return wanted
    dtype = candidates[0]
    bindings[type_str] = dtype
    return dtype


def _unidirectional_broadcast_shape(
    generator: _Generator,
    primary_shape: _Shape,
) -> _Shape:
    """Return a shape that is unidirectionally broadcastable *to* ``primary_shape``.

    Several operators require a secondary parameter input (for example PRelu
    ``slope`` or LayerNormalization ``Scale``) to be unidirectionally
    broadcastable to a primary tensor. Per the ONNX spec that means the result
    has rank ``<= len(primary_shape)`` and, aligned on the trailing axes, every
    dimension is either ``1`` or exactly the corresponding primary dimension.

    The returned shape keeps the primary's (possibly symbolic) dimension when it
    is preserved so downstream oracles bind the same symbol, and falls back to
    ``1`` for unknown (``None``) dimensions since ``1`` broadcasts to anything.
    """
    if primary_shape is None:
        return ()
    rank = len(primary_shape)
    keep = generator.rng.randint(0, rank)
    if keep == 0:
        return ()
    tail = primary_shape[rank - keep :]
    dims: list[_Dim] = []
    for dim in tail:
        if dim is None or generator.rng.randrange(2):
            dims.append(1)
        else:
            dims.append(dim)
    return tuple(dims)


def _plan_prelu(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    x = generator._port_for_name(schema, "X", bindings, min_rank=1)
    slope_shape = _unidirectional_broadcast_shape(generator, x.shape)
    slope = generator._graph_input(x.dtype, slope_shape)
    return _InputPlan([x, slope], output_shapes=(x.shape,))


def _plan_layer_normalization(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    x = generator._port_for_name(schema, "X", bindings, min_rank=1)
    inputs: list[_Port | None] = [
        x,
        generator._graph_input(x.dtype, _unidirectional_broadcast_shape(generator, x.shape)),
    ]
    input_names = {formal.name for formal in schema.inputs}
    if "B" in input_names and generator.rng.randrange(2):
        inputs.append(
            generator._graph_input(
                x.dtype, _unidirectional_broadcast_shape(generator, x.shape)
            )
        )
    return _InputPlan(inputs, output_shapes=(x.shape,))


def _plan_rms_normalization(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    x = generator._port_for_name(schema, "X", bindings, min_rank=1)
    scale = generator._graph_input(
        x.dtype, _unidirectional_broadcast_shape(generator, x.shape)
    )
    bindings["V"] = x.dtype
    return _InputPlan([x, scale], output_shapes=(x.shape,))


def _plan_conv(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = _numeric_dtype(generator, "T", schema, bindings)
    channels = generator.rng.choice((1, 2, 3))
    features = generator.rng.choice((1, 2, 4))
    size = generator.rng.choice((6, 8))
    x = generator._graph_input(dtype, (1, channels, size, size))
    weight = np.ones((features, channels, 3, 3), dtype=_DTYPE_TO_NUMPY[dtype])
    w = generator._constant_port("conv_w", weight, dtype)
    attributes = {
        "kernel_shape": _ints_attr("kernel_shape", (3, 3)),
        "pads": _ints_attr("pads", (1, 1, 1, 1)),
        "strides": _ints_attr("strides", (1, 1)),
    }
    output_shape = (1, features, size, size)
    return _InputPlan([x, w], attributes, output_shapes=(output_shape,))


def _plan_pool(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = _numeric_dtype(generator, "T", schema, bindings)
    channels = generator.rng.choice((1, 2, 3))
    size = generator.rng.choice((6, 8))
    x = generator._graph_input(dtype, (1, channels, size, size))
    attributes = {
        "kernel_shape": _ints_attr("kernel_shape", (2, 2)),
        "strides": _ints_attr("strides", (2, 2)),
    }
    out = (size - 2) // 2 + 1
    output_shape = (1, channels, out, out)
    return _InputPlan([x], attributes, output_shapes=(output_shape,))


def _plan_matmul(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = _numeric_dtype(generator, "T", schema, bindings)
    m, k, n = (generator.rng.choice((2, 3, 4)) for _ in range(3))
    a = generator._graph_input(dtype, (m, k))
    weight = np.ones((k, n), dtype=_DTYPE_TO_NUMPY[dtype])
    b = generator._constant_port("matmul_b", weight, dtype)
    return _InputPlan([a, b], output_shapes=((m, n),))


def _plan_gemm(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = _numeric_dtype(generator, "T", schema, bindings)
    m, k, n = (generator.rng.choice((2, 3, 4)) for _ in range(3))
    a = generator._graph_input(dtype, (m, k))
    b = generator._constant_port(
        "gemm_b", np.ones((k, n), dtype=_DTYPE_TO_NUMPY[dtype]), dtype
    )
    inputs: list[_Port | None] = [a, b]
    if schema.min_input >= 3 or generator.rng.randrange(2):
        c = generator._constant_port(
            "gemm_c", np.ones((n,), dtype=_DTYPE_TO_NUMPY[dtype]), dtype
        )
        inputs.append(c)
    return _InputPlan(inputs, output_shapes=((m, n),))


def _plan_gather(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    data = generator._port_for_name(
        schema,
        "data",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=1,
    )
    assert data.shape is not None
    axis = generator.rng.randrange(len(data.shape))
    extent = int(data.shape[axis])
    indices = generator._constant_port(
        "gather_indices", [0, min(1, extent - 1)], ir.DataType.INT64
    )
    output_shape = (*data.shape[:axis], 2, *data.shape[axis + 1 :])
    attributes = {"axis": _int_attr("axis", axis)}
    return _InputPlan([data, indices], attributes, output_shapes=(output_shape,))


def _plan_concat(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    first = generator._port_for_name(
        schema,
        "inputs",
        bindings,
        concrete_shape=True,
        positive_shape=True,
        min_rank=1,
    )
    assert first.shape is not None
    dtype = first.dtype
    axis = generator.rng.randrange(len(first.shape))
    second_shape = [int(dim) for dim in first.shape]
    second_shape[axis] = generator.rng.choice((1, 2, 3))
    second = generator._graph_input(dtype, tuple(second_shape))
    output_shape = [int(dim) for dim in first.shape]
    output_shape[axis] = int(first.shape[axis]) + second_shape[axis]
    attributes = {"axis": _int_attr("axis", axis)}
    return _InputPlan([first, second], attributes, output_shapes=(tuple(output_shape),))


def _plan_split(
    generator: _Generator,
    schema: onnx.defs.OpSchema,
    bindings: dict[str, ir.DataType],
) -> _InputPlan:
    dtype = _numeric_dtype(
        generator, "T", schema, bindings, prefer=(ir.DataType.FLOAT, ir.DataType.INT64)
    )
    rank = generator.rng.randint(1, 3)
    shape = [generator.rng.choice((1, 2, 3)) for _ in range(rank)]
    axis = generator.rng.randrange(rank)
    shape[axis] = 4
    x = generator._graph_input(dtype, tuple(shape))
    half = list(shape)
    half[axis] = 2
    output_shape = tuple(half)
    attributes: dict[str, ir.Attr] = {"axis": _int_attr("axis", axis)}
    inputs: list[_Port | None] = [x]
    input_names = {formal.name for formal in schema.inputs}
    if "split" in input_names:
        inputs.append(generator._constant_port("split_sizes", [2, 2], ir.DataType.INT64))
    elif "num_outputs" in schema.attributes:
        attributes["num_outputs"] = _int_attr("num_outputs", 2)
    else:
        attributes["split"] = _ints_attr("split", (2, 2))
    return _InputPlan(
        inputs,
        attributes,
        output_shapes=(output_shape, output_shape),
        output_count=2,
    )


_op_planners: dict[_OpKey, InputPlanner] = {
    ("", "ConstantOfShape"): _plan_constant_of_shape,
    ("", "Expand"): _plan_expand,
    ("", "OneHot"): _plan_one_hot,
    ("", "Pad"): _plan_pad,
    ("", "Range"): _plan_range,
    ("", "Reshape"): _plan_reshape,
    ("", "Resize"): _plan_resize,
    ("", "Slice"): _plan_slice,
    ("", "Tile"): _plan_tile,
    ("", "TopK"): _plan_topk,
    ("", "AveragePool"): _plan_pool,
    ("", "Concat"): _plan_concat,
    ("", "Conv"): _plan_conv,
    ("", "Gather"): _plan_gather,
    ("", "Gemm"): _plan_gemm,
    ("", "LayerNormalization"): _plan_layer_normalization,
    ("", "MatMul"): _plan_matmul,
    ("", "MaxPool"): _plan_pool,
    ("", "PRelu"): _plan_prelu,
    ("", "RMSNormalization"): _plan_rms_normalization,
    ("", "Split"): _plan_split,
}
_semantic_planners: dict[_OpKey, InputPlanner] = {
    ("", "DepthToSpace"): _plan_depth_to_space,
    ("", "SpaceToDepth"): _plan_space_to_depth,
}


class _Generator:
    """Stateful implementation behind :func:`generate`."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes: list[ir.Node] = []
        self.graph_inputs: list[ir.Value] = []
        self.initializers: list[ir.Value] = []
        self.ports: list[_Port] = []
        self.symbolic_dims: dict[str, int | None] = {}
        self.symbol_constraints: dict[str, SymbolConstraint] = {}
        self.data_dependent_ports: set[str] = set()
        self.selected_ops: list[tuple[str, str, int]] = []
        self.op_counts: dict[_OpKey, int] = {}
        self.value_index = 0
        self.symbol_index = 0
        self.template_name = _TEMPLATE_NAMES[seed % len(_TEMPLATE_NAMES)]
        self.planner_key = sorted(_op_planners)[seed % len(_op_planners)]
        _registry.registry.collect()
        self.opset = self._choose_opset()

    def build(self) -> FuzzCase:
        """Build one deterministic fuzz case."""
        self._seed_port_pool()
        template_output = self._add_template()
        self._add_operator(self.planner_key)
        planner_node = self.nodes[-1].name
        for _ in range(6):
            self._add_weighted_operator()
        self._maybe_add_control_flow()

        graph_output = (
            template_output.value if template_output is not None else self.graph_inputs[0]
        )
        graph = ir.Graph(
            inputs=self.graph_inputs,
            outputs=[graph_output],
            nodes=self.nodes,
            initializers=self.initializers,
            opset_imports={"": self.opset},
            name=f"fuzz_{self.seed}",
        )
        model = ir.Model(
            graph,
            ir_version=10,
            producer_name="onnx-shape-inference-fuzzer",
        )
        return FuzzCase(
            model=model,
            seed=self.seed,
            opset_imports={"": self.opset},
            symbolic_dims=tuple(self.symbolic_dims),
            symbol_constraints=dict(self.symbol_constraints),
            data_dependent_values=frozenset(self.data_dependent_ports),
            selected_ops=tuple(self.selected_ops),
            metadata={
                "planner_node": planner_node,
                "planner_op": self.planner_key,
                "template": self.template_name,
            },
        )

    def _choose_opset(self) -> int:
        max_opset = onnx.defs.onnx_opset_version()
        template_minimum = {
            "shape_slice_concat_reshape": 10,
            "shape_gather_unsqueeze_concat_reshape": 10,
            "constant_of_shape": 9,
            "shape_slice_range": 11,
            None: 1,
        }[self.template_name]
        planner_versions = _registry.registry.version_boundaries(*self.planner_key)
        planner_minimum = planner_versions[0] if planner_versions else 1
        minimum = max(template_minimum, planner_minimum)
        boundaries = sorted(
            {
                version
                for domain, _, versions in _registry.registry.iter_supported()
                if domain == ""
                for version in versions
                if minimum <= version <= max_opset
            }
        )
        if not boundaries:
            return max(minimum, max_opset)
        boundary = self.rng.choice(boundaries)
        return max(minimum, min(max_opset, boundary + self.rng.choice((-1, 0, 0, 0, 1))))

    def _seed_port_pool(self) -> None:
        shared = self._new_symbol()
        self._graph_input(ir.DataType.FLOAT, (shared, self.rng.choice((1, 2, 3))))
        self._graph_input(ir.DataType.FLOAT, (shared, 1))
        self._graph_input(ir.DataType.INT64, (self.rng.choice((1, 2, 3)),))
        self._graph_input(ir.DataType.BOOL, ())
        # Seed a few small-integer typed ports so dtype-specific operator paths
        # (bitwise, quantized-adjacent, cast targets) get exercised with
        # reusable inputs rather than only float/int64/bool.
        for dtype in (
            ir.DataType.INT32,
            ir.DataType.INT8,
            ir.DataType.INT16,
            ir.DataType.UINT8,
            ir.DataType.UINT16,
        ):
            self._graph_input(dtype, (self.rng.choice((1, 2, 3)),))

    def _new_value_name(self, prefix: str) -> str:
        name = f"{prefix}_{self.value_index}"
        self.value_index += 1
        return name

    def _new_symbol(self) -> str:
        name = f"s{self.symbol_index}"
        self.symbol_index += 1
        self.symbolic_dims[name] = None
        self.symbol_constraints[name] = SymbolConstraint()
        return name

    def _require_divisible(self, symbol: str, divisor: int) -> None:
        constraint = self.symbol_constraints[symbol]
        constraint.divisible_by = math.lcm(constraint.divisible_by, divisor)

    def _random_shape(
        self,
        *,
        concrete: bool = False,
        positive: bool = False,
        allow_zero: bool = False,
        min_rank: int = 1,
        max_rank: int = 4,
    ) -> _Shape:
        rank = self.rng.randint(min_rank, max_rank)
        dims: list[_Dim] = []
        for _ in range(rank):
            if not concrete and self.rng.randrange(4) == 0:
                if self.symbolic_dims and self.rng.randrange(2):
                    dims.append(self.rng.choice(sorted(self.symbolic_dims)))
                elif len(self.symbolic_dims) < 6:
                    dims.append(self._new_symbol())
                else:
                    dims.append(self.rng.choice(sorted(self.symbolic_dims)))
            else:
                # Bias away from zero-length extents: they frequently make ONNX
                # Runtime unable to execute the graph, starving the soundness
                # oracle. Callers must opt in to zeros explicitly.
                if allow_zero and not positive:
                    values = (0, 1, 1, 2, 2, 3)
                else:
                    values = (1, 1, 2, 2, 3)
                dims.append(self.rng.choice(values))
        return tuple(dims)

    def _graph_input(self, dtype: ir.DataType, shape: _Shape) -> _Port:
        value = ir.Value(
            name=self._new_value_name("input"),
            type=ir.TensorType(dtype),
            shape=ir.Shape(shape),
        )
        port = _Port(value, dtype, shape)
        self.graph_inputs.append(value)
        self.ports.append(port)
        return port

    def _constant_port(
        self,
        prefix: str,
        data: object,
        dtype: ir.DataType,
    ) -> _Port:
        np_dtype = _DTYPE_TO_NUMPY.get(dtype)
        if np_dtype is None:
            raise _PlanningError(f"Cannot construct initializer with dtype {dtype}")
        array = np.asarray(data, dtype=np_dtype)
        name = self._new_value_name(prefix)
        tensor = ir.Tensor(array, name=name)
        value = ir.Value(
            name=name,
            type=ir.TensorType(dtype),
            shape=ir.Shape(array.shape),
            const_value=tensor,
        )
        port = _Port(value, dtype, tuple(int(dim) for dim in array.shape))
        self.initializers.append(value)
        return port

    def _dtype_candidates(
        self,
        type_str: str,
        schema: onnx.defs.OpSchema,
    ) -> tuple[ir.DataType, ...]:
        concrete_match = _TENSOR_TYPE.fullmatch(type_str)
        if concrete_match:
            dtype = _TYPE_NAME_TO_DTYPE.get(concrete_match.group(1))
            return (dtype,) if dtype is not None else ()
        constraints = {
            constraint.type_param_str: constraint.allowed_type_strs
            for constraint in schema.type_constraints
        }
        candidates = {
            _TYPE_NAME_TO_DTYPE[match.group(1)]
            for allowed in constraints.get(type_str, ())
            if (match := _TENSOR_TYPE.fullmatch(allowed))
            and match.group(1) in _TYPE_NAME_TO_DTYPE
        }
        return tuple(dtype for dtype in _PREFERRED_DTYPES if dtype in candidates)

    def _dtype_for_type(
        self,
        type_str: str,
        schema: onnx.defs.OpSchema,
        bindings: dict[str, ir.DataType],
    ) -> ir.DataType:
        if type_str in bindings:
            return bindings[type_str]
        candidates = self._dtype_candidates(type_str, schema)
        if not candidates:
            raise _PlanningError(f"No supported tensor dtype for {schema.name}.{type_str}")
        pool_dtypes = {port.dtype for port in self.ports}
        reusable = [dtype for dtype in candidates if dtype in pool_dtypes]
        dtype = self.rng.choice(reusable or list(candidates))
        if not _TENSOR_TYPE.fullmatch(type_str):
            bindings[type_str] = dtype
        return dtype

    def _port_for_name(
        self,
        schema: onnx.defs.OpSchema,
        name: str,
        bindings: dict[str, ir.DataType],
        *,
        concrete_shape: bool = False,
        positive_shape: bool = False,
        min_rank: int = 1,
        max_rank: int = 4,
    ) -> _Port:
        formal = next((formal for formal in schema.inputs if formal.name == name), None)
        if formal is None:
            raise _PlanningError(f"{schema.name} schema has no input named {name}")
        dtype = self._dtype_for_type(formal.type_str, schema, bindings)
        reusable = [
            port
            for port in self.ports
            if port.dtype == dtype
            and port.shape is not None
            and min_rank <= len(port.shape) <= max_rank
            and (not concrete_shape or all(isinstance(dim, int) for dim in port.shape))
            and (
                not positive_shape
                or all(isinstance(dim, int) and dim > 0 for dim in port.shape)
            )
        ]
        if reusable and self.rng.randrange(4):
            return self.rng.choice(reusable)
        shape = self._random_shape(
            concrete=concrete_shape,
            positive=positive_shape,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        return self._graph_input(dtype, shape)

    def _required_attribute(
        self,
        op_type: str,
        name: str,
        attr_type: onnx.defs.OpSchema.AttrType,
    ) -> ir.Attr:
        if attr_type == onnx.defs.OpSchema.AttrType.FLOAT:
            return ir.Attr(name, ir.AttributeType.FLOAT, 1.0)
        if attr_type == onnx.defs.OpSchema.AttrType.INT:
            if name == "to":
                value = int(ir.DataType.FLOAT)
            elif name == "axis":
                value = 0
            elif name == "blocksize":
                value = 2
            else:
                value = 1
            return ir.Attr(name, ir.AttributeType.INT, value)
        if attr_type == onnx.defs.OpSchema.AttrType.STRING:
            value = {
                ("BitShift", "direction"): "LEFT",
                ("Einsum", "equation"): "i->i",
                ("TfIdfVectorizer", "mode"): "TF",
            }.get((op_type, name), "NOTSET")
            return ir.Attr(name, ir.AttributeType.STRING, value)
        if attr_type == onnx.defs.OpSchema.AttrType.INTS:
            if name in {"kernel_shape", "pooled_shape", "shape"}:
                value = (1, 1)
            else:
                value = (0,)
            return ir.Attr(name, ir.AttributeType.INTS, value)
        if attr_type == onnx.defs.OpSchema.AttrType.FLOATS:
            return ir.Attr(name, ir.AttributeType.FLOATS, (1.0,))
        if attr_type == onnx.defs.OpSchema.AttrType.STRINGS:
            return ir.Attr(name, ir.AttributeType.STRINGS, ("x",))
        if attr_type == onnx.defs.OpSchema.AttrType.TENSOR:
            tensor = ir.Tensor(np.asarray([1.0], dtype=np.float32), name=f"{name}_value")
            return ir.Attr(name, ir.AttributeType.TENSOR, tensor)
        if attr_type == onnx.defs.OpSchema.AttrType.TENSORS:
            tensor = ir.Tensor(np.asarray([1.0], dtype=np.float32), name=f"{name}_value")
            return ir.Attr(name, ir.AttributeType.TENSORS, (tensor,))
        raise _PlanningError(f"Unsupported required attribute type: {attr_type}")

    def _default_plan(
        self,
        schema: onnx.defs.OpSchema,
        bindings: dict[str, ir.DataType],
    ) -> _InputPlan:
        inputs: list[_Port | None] = []
        required_after = [
            sum(
                later.option == onnx.defs.OpSchema.FormalParameterOption.Single
                for later in schema.inputs[index + 1 :]
            )
            for index in range(len(schema.inputs))
        ]
        for index, formal in enumerate(schema.inputs):
            if formal.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                continue
            count = 1
            if formal.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                count = max(1, schema.min_input - len(inputs) - required_after[index])
            for _ in range(count):
                dtype = self._dtype_for_type(formal.type_str, schema, bindings)
                reusable = [port for port in self.ports if port.dtype == dtype]
                if reusable and self.rng.randrange(5):
                    inputs.append(self.rng.choice(reusable))
                else:
                    inputs.append(self._graph_input(dtype, self._random_shape()))
        attributes = {
            name: self._required_attribute(schema.name, name, attr.type)
            for name, attr in sorted(schema.attributes.items())
            if attr.required
        }
        return _InputPlan(inputs, attributes)

    def _schema(self, op_key: _OpKey) -> onnx.defs.OpSchema:
        domain, op_type = op_key
        return onnx.defs.get_schema(
            op_type,
            max_inclusive_version=self.opset,
            domain=domain,
        )

    def _can_generate(self, op_key: _OpKey) -> bool:
        domain, op_type = op_key
        if domain or op_type in _GENERIC_EXCLUDED_OPS:
            return False
        boundaries = _registry.registry.version_boundaries(domain, op_type)
        if not boundaries or boundaries[0] > self.opset:
            return False
        try:
            schema = self._schema(op_key)
        except onnx.onnx_cpp2py_export.defs.SchemaError:
            return False
        if schema.deprecated:
            return False
        if any(
            attr.required and attr.type not in _SUPPORTED_REQUIRED_ATTRS
            for attr in schema.attributes.values()
        ):
            return False
        formals = (*schema.inputs, *schema.outputs)
        return all(self._dtype_candidates(formal.type_str, schema) for formal in formals)

    def _candidate_ops(self) -> list[_OpKey]:
        return [
            (domain, op_type)
            for domain, op_type, _ in _registry.registry.iter_supported()
            if self._can_generate((domain, op_type))
        ]

    def _add_weighted_operator(self) -> None:
        candidates = self._candidate_ops()
        if not candidates:
            raise _PlanningError(f"No generatable operators for opset {self.opset}")
        remaining = list(candidates)
        while remaining:
            minimum_count = min(self.op_counts.get(op_key, 0) for op_key in remaining)
            least_used = [
                op_key
                for op_key in remaining
                if self.op_counts.get(op_key, 0) == minimum_count
            ]
            op_key = self.rng.choice(least_used)
            try:
                self._add_operator(op_key)
            except _PlanningError:
                remaining.remove(op_key)
            else:
                return
        raise _PlanningError("All candidate operator plans failed")

    def _add_operator(self, op_key: _OpKey) -> list[_Port]:
        if not self._can_generate(op_key):
            raise _PlanningError(
                f"Operator is not generatable at opset {self.opset}: {op_key}"
            )
        schema = self._schema(op_key)
        bindings: dict[str, ir.DataType] = {}
        planner = _op_planners.get(op_key) or _semantic_planners.get(op_key)
        plan = (
            planner(self, schema, bindings)
            if planner is not None
            else self._default_plan(schema, bindings)
        )
        output_formals: list[onnx.defs.OpSchema.FormalParameter] = []
        for formal in schema.outputs:
            if formal.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
                continue
            count = 1
            if formal.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                if plan.output_count is not None:
                    count = max(1, plan.output_count - len(output_formals))
                else:
                    count = max(1, schema.min_output - len(output_formals))
            output_formals.extend([formal] * count)
        outputs: list[ir.Value] = []
        ports: list[_Port] = []
        data_dependent = op_key in DATA_DEPENDENT_OPS
        for index, formal in enumerate(output_formals):
            dtype = self._dtype_for_type(formal.type_str, schema, bindings)
            shape = plan.output_shapes[index] if index < len(plan.output_shapes) else None
            value = ir.Value(
                name=self._new_value_name(schema.name.lower()),
                type=ir.TensorType(dtype),
                shape=ir.Shape(shape) if shape is not None else None,
            )
            outputs.append(value)
            port = _Port(value, dtype, shape, data_dependent)
            ports.append(port)
            if data_dependent and value.name is not None:
                self.data_dependent_ports.add(value.name)
        node = ir.Node(
            op_key[0],
            op_key[1],
            inputs=[port.value if port is not None else None for port in plan.inputs],
            outputs=outputs,
            attributes=plan.attributes,
            name=self._new_value_name("node"),
        )
        self.nodes.append(node)
        self.ports.extend(ports)
        boundary = max(
            version
            for version in _registry.registry.version_boundaries(*op_key)
            if version <= self.opset
        )
        self.selected_ops.append((*op_key, boundary))
        self.op_counts[op_key] = self.op_counts.get(op_key, 0) + 1
        return ports

    def _cf_value(self, prefix: str, dtype: ir.DataType, shape: _Shape) -> ir.Value:
        return ir.Value(
            name=self._new_value_name(prefix),
            type=ir.TensorType(dtype),
            shape=ir.Shape(shape),
        )

    def _cf_constant_node(
        self, array: np.ndarray, dtype: ir.DataType
    ) -> tuple[ir.Node, ir.Value]:
        """Build a self-contained ``Constant`` node for use inside a subgraph."""
        out = self._cf_value("cf_const", dtype, array.shape)
        node = ir.Node(
            "",
            "Constant",
            inputs=[],
            outputs=[out],
            attributes={
                "value": ir.Attr(
                    "value",
                    ir.AttributeType.TENSOR,
                    ir.Tensor(array, name=self._new_value_name("cf_value")),
                )
            },
            name=self._new_value_name("node"),
        )
        return node, out

    def _register_control_flow(
        self,
        op_type: str,
        node: ir.Node,
        output: ir.Value,
        shape: _Shape,
        dtype: ir.DataType,
    ) -> None:
        self.nodes.append(node)
        self.ports.append(_Port(output, dtype, shape))
        boundaries = [
            version
            for version in _registry.registry.version_boundaries("", op_type)
            if version <= self.opset
        ]
        if boundaries:
            self.selected_ops.append(("", op_type, max(boundaries)))
        self.op_counts[("", op_type)] = self.op_counts.get(("", op_type), 0) + 1

    def _add_if(self) -> None:
        dtype = ir.DataType.FLOAT
        shape: _Shape = (2, 3)
        cond = self._constant_port("if_cond", True, ir.DataType.BOOL)
        then_node, then_out = self._cf_constant_node(np.ones(shape, np.float32), dtype)
        then_branch = ir.Graph(
            inputs=[],
            outputs=[then_out],
            nodes=[then_node],
            name=self._new_value_name("then_branch"),
        )
        else_node, else_out = self._cf_constant_node(np.full(shape, 2.0, np.float32), dtype)
        else_branch = ir.Graph(
            inputs=[],
            outputs=[else_out],
            nodes=[else_node],
            name=self._new_value_name("else_branch"),
        )
        out = self._cf_value("if_out", dtype, shape)
        node = ir.Node(
            "",
            "If",
            inputs=[cond.value],
            outputs=[out],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            },
            name=self._new_value_name("node"),
        )
        self._register_control_flow("If", node, out, shape, dtype)

    def _add_loop(self) -> None:
        dtype = ir.DataType.FLOAT
        shape: _Shape = (2, 3)
        trip = self._constant_port("loop_trip", 3, ir.DataType.INT64)
        cond = self._constant_port("loop_cond", True, ir.DataType.BOOL)
        init = self._constant_port("loop_init", np.ones(shape, np.float32), dtype)
        iter_num = self._cf_value("loop_iter", ir.DataType.INT64, ())
        cond_in = self._cf_value("loop_cond_in", ir.DataType.BOOL, ())
        carried_in = self._cf_value("loop_carried_in", dtype, shape)
        cond_out = self._cf_value("loop_cond_out", ir.DataType.BOOL, ())
        carried_out = self._cf_value("loop_carried_out", dtype, shape)
        body = ir.Graph(
            inputs=[iter_num, cond_in, carried_in],
            outputs=[cond_out, carried_out],
            nodes=[
                ir.Node(
                    "",
                    "Identity",
                    inputs=[cond_in],
                    outputs=[cond_out],
                    name=self._new_value_name("node"),
                ),
                ir.Node(
                    "",
                    "Identity",
                    inputs=[carried_in],
                    outputs=[carried_out],
                    name=self._new_value_name("node"),
                ),
            ],
            name=self._new_value_name("loop_body"),
        )
        out = self._cf_value("loop_out", dtype, shape)
        node = ir.Node(
            "",
            "Loop",
            inputs=[trip.value, cond.value, init.value],
            outputs=[out],
            attributes={"body": ir.Attr("body", ir.AttributeType.GRAPH, body)},
            name=self._new_value_name("node"),
        )
        self._register_control_flow("Loop", node, out, shape, dtype)

    def _add_scan(self) -> None:
        dtype = ir.DataType.FLOAT
        slice_shape: _Shape = (2, 3)
        seq_len = 4
        seq = self._constant_port(
            "scan_seq", np.ones((seq_len, *slice_shape), np.float32), dtype
        )
        x_in = self._cf_value("scan_x", dtype, slice_shape)
        y_out = self._cf_value("scan_y", dtype, slice_shape)
        body = ir.Graph(
            inputs=[x_in],
            outputs=[y_out],
            nodes=[
                ir.Node(
                    "",
                    "Identity",
                    inputs=[x_in],
                    outputs=[y_out],
                    name=self._new_value_name("node"),
                )
            ],
            name=self._new_value_name("scan_body"),
        )
        output_shape = (seq_len, *slice_shape)
        out = self._cf_value("scan_out", dtype, output_shape)
        node = ir.Node(
            "",
            "Scan",
            inputs=[seq.value],
            outputs=[out],
            attributes={
                "body": ir.Attr("body", ir.AttributeType.GRAPH, body),
                "num_scan_inputs": ir.Attr("num_scan_inputs", ir.AttributeType.INT, 1),
            },
            name=self._new_value_name("node"),
        )
        self._register_control_flow("Scan", node, out, output_shape, dtype)

    def _maybe_add_control_flow(self) -> None:
        """Occasionally append a control-flow op with valid, small subgraphs.

        If and Loop require GRAPH-typed subgraph attributes that the generic
        schema-driven planner cannot synthesise, so they are constructed here.
        Scan is only emitted at opset 9+, where its modern (no ``sequence_lens``)
        form is available.
        """
        if self.seed % 3 != 0:
            return
        builders = [self._add_if, self._add_loop]
        if self.opset >= 9:
            builders.append(self._add_scan)
        builders[(self.seed // 3) % len(builders)]()

    def _template_node(
        self,
        op_type: str,
        inputs: Sequence[_Port | None],
        dtype: ir.DataType,
        shape: _Shape,
        *,
        attributes: dict[str, ir.Attr] | None = None,
    ) -> _Port:
        op_key = ("", op_type)
        output = ir.Value(
            name=self._new_value_name(op_type.lower()),
            type=ir.TensorType(dtype),
            shape=ir.Shape(shape),
        )
        node = ir.Node(
            "",
            op_type,
            inputs=[port.value if port is not None else None for port in inputs],
            outputs=[output],
            attributes=attributes or {},
            name=self._new_value_name("node"),
        )
        port = _Port(output, dtype, shape, op_key in DATA_DEPENDENT_OPS)
        self.nodes.append(node)
        self.ports.append(port)
        boundary = max(
            version
            for version in _registry.registry.version_boundaries(*op_key)
            if version <= self.opset
        )
        self.selected_ops.append((*op_key, boundary))
        self.op_counts[op_key] = self.op_counts.get(op_key, 0) + 1
        if port.data_dependent and output.name is not None:
            self.data_dependent_ports.add(output.name)
        return port

    def _template_input(self) -> _Port:
        symbol = self._new_symbol()
        return self._graph_input(ir.DataType.FLOAT, (symbol, 2, 3))

    def _slice_first_shape_dim(self, shape_port: _Port) -> _Port:
        starts = self._constant_port("template_starts", [0], ir.DataType.INT64)
        ends = self._constant_port("template_ends", [1], ir.DataType.INT64)
        axes = self._constant_port("template_axes", [0], ir.DataType.INT64)
        steps = self._constant_port("template_steps", [1], ir.DataType.INT64)
        return self._template_node(
            "Slice",
            [shape_port, starts, ends, axes, steps],
            ir.DataType.INT64,
            (1,),
        )

    def _add_template(self) -> _Port | None:
        if self.template_name is None:
            return None
        if self.template_name == "constant_of_shape":
            shape = self._constant_port("template_shape", [2, 3], ir.DataType.INT64)
            return self._template_node(
                "ConstantOfShape",
                [shape],
                ir.DataType.FLOAT,
                (2, 3),
            )

        data = self._template_input()
        assert data.shape is not None
        shape = self._template_node("Shape", [data], ir.DataType.INT64, (len(data.shape),))
        symbol = data.shape[0]
        if self.template_name == "shape_slice_concat_reshape":
            first_dim = self._slice_first_shape_dim(shape)
            tail = self._constant_port("template_tail", [6], ir.DataType.INT64)
            target = self._template_node(
                "Concat",
                [first_dim, tail],
                ir.DataType.INT64,
                (2,),
                attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            )
            return self._template_node(
                "Reshape",
                [data, target],
                ir.DataType.FLOAT,
                (symbol, 6),
            )
        if self.template_name == "shape_gather_unsqueeze_concat_reshape":
            index = self._constant_port("template_index", 0, ir.DataType.INT64)
            gathered = self._template_node(
                "Gather",
                [shape, index],
                ir.DataType.INT64,
                (),
                attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            )
            if self.opset >= 13:
                axes = self._constant_port("template_unsqueeze_axes", [0], ir.DataType.INT64)
                unsqueezed = self._template_node(
                    "Unsqueeze",
                    [gathered, axes],
                    ir.DataType.INT64,
                    (1,),
                )
            else:
                unsqueezed = self._template_node(
                    "Unsqueeze",
                    [gathered],
                    ir.DataType.INT64,
                    (1,),
                    attributes={"axes": ir.Attr("axes", ir.AttributeType.INTS, (0,))},
                )
            tail = self._constant_port("template_tail", [6], ir.DataType.INT64)
            target = self._template_node(
                "Concat",
                [unsqueezed, tail],
                ir.DataType.INT64,
                (2,),
                attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
            )
            return self._template_node(
                "Reshape",
                [data, target],
                ir.DataType.FLOAT,
                (symbol, 6),
            )
        if self.template_name == "shape_slice_range":
            first_dim = self._slice_first_shape_dim(shape)
            if self.opset >= 13:
                axes = self._constant_port("template_squeeze_axes", [0], ir.DataType.INT64)
                limit = self._template_node(
                    "Squeeze",
                    [first_dim, axes],
                    ir.DataType.INT64,
                    (),
                )
            else:
                limit = self._template_node(
                    "Squeeze",
                    [first_dim],
                    ir.DataType.INT64,
                    (),
                    attributes={"axes": ir.Attr("axes", ir.AttributeType.INTS, (0,))},
                )
            start = self._constant_port("template_range_start", 0, ir.DataType.INT64)
            delta = self._constant_port("template_range_delta", 1, ir.DataType.INT64)
            return self._template_node(
                "Range",
                [start, limit, delta],
                ir.DataType.INT64,
                (None,),
            )
        raise AssertionError(f"Unknown template: {self.template_name}")


def generate(seed: int) -> FuzzCase:
    """Generate a deterministic structurally-valid ONNX shape-fuzzing case.

    Args:
        seed: Integer seed controlling every generation decision.

    Returns:
        A generated :class:`FuzzCase`. Reusing a seed produces byte-identical
        serialized graphs.

    """
    return _Generator(seed).build()
