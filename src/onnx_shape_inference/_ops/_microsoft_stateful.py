# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for packed recurrent and Engram contrib operators."""

from __future__ import annotations

__all__: list[str] = []

import onnx_ir as ir

from onnx_shape_inference import _context, _registry

_reg = _registry.registry.register


def _input(node: ir.Node, index: int) -> ir.Value | None:
    return node.inputs[index] if index < len(node.inputs) else None


def _required(node: ir.Node, index: int, name: str) -> ir.Value:
    value = _input(node, index)
    if value is None:
        raise _context.OpUsageError(node, f"Required input '{name}' (#{index}) is missing")
    return value


def _int_attr(node: ir.Node, name: str, default: int) -> int:
    attr = node.attributes.get(name)
    return attr.as_int() if attr is not None else default


def _capture_capacity(
    ctx: _context.ShapeInferenceContext, node: ir.Node, count_index: int
) -> int | None:
    capacity = _int_attr(node, "state_update_capacity", 0)
    if not 0 <= capacity <= 8:
        ctx.record_error(node, "state_update_capacity must be in [0, 8]")
        return None
    if (_input(node, count_index) is not None) != (capacity > 0):
        raise _context.OpUsageError(
            node, "capture_count is required exactly when state_update_capacity is positive"
        )
    return capacity


@_reg("com.microsoft", "HyperConnectionPreMix", since_version=1)
def infer_hyper_connection_pre_mix(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    streams, _pre_mix = _context.check_inputs(node, "streams", "pre_mix")
    branches = _int_attr(node, "num_branches", 0)
    if branches < 0:
        ctx.record_error(node, "num_branches must be nonnegative")
        return
    shape = streams.shape
    if shape is not None:
        if shape.rank() < (1 if branches else 2):
            ctx.record_error(node, "streams rank is incompatible with num_branches")
            return
        if branches:
            width = shape[-1]
            if isinstance(width, int) and width % branches:
                ctx.record_error(node, "flattened stream width must divide num_branches")
                return
            shape = ir.Shape([*shape.dims[:-1], width // branches])
        else:
            shape = ir.Shape([*shape.dims[:-2], shape[-1]])
    ctx.set_shape_and_dtype(node.outputs[0], shape, streams.dtype)


@_reg("com.microsoft", "HyperConnectionPostMix", since_version=1)
def infer_hyper_connection_post_mix(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    streams, _block, _post_mix = _context.check_inputs(
        node, "streams", "block_output", "post_mix"
    )
    ctx.set_shape_and_dtype(node.outputs[0], streams.shape, streams.dtype)


@_reg("com.microsoft", "EngramGate", since_version=1)
def infer_engram_gate(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    key, _query, _value, _key_scale, _query_scale = _context.check_inputs(
        node, "key", "query", "value", "key_norm_scale", "query_norm_scale"
    )
    if len(node.outputs) > 1 and node.outputs[1].name != "":
        _required(node, 5, "conv_norm_scale")
    for output in node.outputs:
        if output.name != "":
            ctx.set_shape_and_dtype(output, key.shape, key.dtype)


@_reg("com.microsoft", "NGramHashMapping", since_version=1)
@_reg("com.microsoft", "VarlenNGramHashMapping", since_version=1)
def infer_ngram_hash_mapping(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    ids, _multipliers, _vocab = _context.check_inputs(
        node, "input_ids", "multipliers", "vocab_sizes"
    )
    packed = node.op_type == "VarlenNGramHashMapping"
    offsets = _required(node, 3, "cumulative_sequence_length") if packed else None
    order = _context.require_attr(node, "max_ngram_size").as_int()
    heads = _context.require_attr(node, "n_head_per_ngram").as_int()
    if order < 2 or heads < 1 or (order - 1) * heads > 2**63 - 1:
        ctx.record_error(node, "invalid n-gram order or head count")
        return
    capacity = _capture_capacity(ctx, node, 9) if packed else 0
    if capacity is None:
        return
    shape = ids.shape
    if shape is not None and shape.rank() != (1 if packed else 2):
        ctx.record_error(node, "input_ids must have rank 1 for packed ids, otherwise rank 2")
        return
    batch: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    if offsets is not None and offsets.shape is not None:
        if offsets.shape.rank() != 1:
            ctx.record_error(node, "cumulative_sequence_length must have rank 1")
            return
        if isinstance(offsets.shape[0], int) and offsets.shape[0] < 2:
            ctx.record_error(node, "cumulative_sequence_length must have at least 2 elements")
            return
        batch = offsets.shape[0] - 1
    elif not packed and shape is not None:
        batch = shape[0]
    if packed:
        counts = _input(node, 9)
        if counts is not None and counts.shape is not None:
            if counts.shape.rank() != 1 or (
                isinstance(batch, int)
                and isinstance(counts.shape[0], int)
                and counts.shape[0] != batch
            ):
                ctx.record_error(node, "capture_count must have shape (batch_size)")
                return
    output_shape = ir.Shape([*shape.dims, (order - 1) * heads]) if shape is not None else None
    ctx.set_shape_and_dtype(node.outputs[0], output_shape, ids.dtype)
    for i, output in enumerate(node.outputs[1:], 1):
        if output.name == "":
            continue
        dims = [batch, order - 1] if i != 3 else [batch, capacity, order - 1]
        dtype = ir.DataType.INT32 if packed and i == 2 else ids.dtype
        ctx.set_shape_and_dtype(output, ir.Shape(dims), dtype)


@_reg("com.microsoft", "VarlenCausalConvWithState", since_version=1)
def infer_varlen_causal_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    x, weight, offsets = _context.check_inputs(
        node, "input", "weight", "cumulative_sequence_length"
    )
    initial = _required(node, 4, "initial_state")
    capacity = _capture_capacity(ctx, node, 5)
    dilation = _int_attr(node, "dilation", 1)
    if capacity is None:
        return
    if dilation < 1:
        ctx.record_error(node, "dilation must be positive")
        return
    for value, rank in ((x, 2), (weight, 3), (offsets, 1), (initial, 3)):
        if value.shape is not None and value.shape.rank() != rank:
            ctx.record_error(node, f"input {value.name!r} must have rank {rank}")
            return
    batch: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    if offsets.shape is not None:
        if isinstance(offsets.shape[0], int) and offsets.shape[0] < 2:
            ctx.record_error(node, "cumulative_sequence_length must have at least 2 elements")
            return
        batch = offsets.shape[0] - 1
    elif initial.shape is not None:
        batch = initial.shape[0]
    state_shape = initial.shape
    channels = x.shape[1] if x.shape is not None else ctx.new_symbolic_dim()
    if x.shape is not None and weight.shape is not None:
        state_shape = ir.Shape([batch, channels, (weight.shape[2] - 1) * dilation])
    if x.shape is not None and isinstance(batch, int) and isinstance(x.shape[0], int):
        if x.shape[0] < batch:
            ctx.record_error(node, "total_tokens must be at least batch_size")
            return
    if state_shape is not None and initial.shape is not None:
        if any(
            isinstance(a, int) and isinstance(b, int) and a != b
            for a, b in zip(state_shape, initial.shape)
        ):
            ctx.record_error(
                node, "initial_state has incompatible batch, channels, or state length"
            )
            return
    counts = _input(node, 5)
    if counts is not None and counts.shape is not None:
        if counts.shape.rank() != 1 or (
            isinstance(batch, int)
            and isinstance(counts.shape[0], int)
            and counts.shape[0] != batch
        ):
            ctx.record_error(node, "capture_count must have shape (batch_size)")
            return
    ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)
    ctx.set_shape_and_dtype(node.outputs[1], state_shape, x.dtype)
    if len(node.outputs) > 2 and node.outputs[2].name != "":
        ctx.set_shape_and_dtype(
            node.outputs[2], ir.Shape([batch, capacity, channels]), x.dtype
        )


@_reg("com.microsoft", "GatedDeltaNet", since_version=1)
def infer_gated_delta_net(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (query,) = _context.check_inputs(node, "query")
    key, value, initial = _input(node, 1), _input(node, 2), _input(node, 6)
    if (key is None) != (value is None):
        raise _context.OpUsageError(
            node, "key and value must both be present or both be omitted"
        )
    packed = value is None
    if packed and initial is None:
        raise _context.OpUsageError(node, "initial_state is required for packed QKV")
    capacity = _capture_capacity(ctx, node, 9)
    if capacity is None:
        return
    for i, output in enumerate(node.outputs):
        if output.name != "":
            ctx.set_shape_and_dtype(output, None, query.dtype if i == 0 else ir.DataType.FLOAT)
    if query.shape is None:
        return
    rank = query.shape.rank()
    token_rank = rank - (1 if packed else 2)
    if rank not in ((2, 3) if packed else (3, 4)):
        ctx.record_error(node, "packed QKV must have rank 2 or 3; separate QKV rank 3 or 4")
        return
    state = initial.shape if initial is not None else None
    if state is not None and state.rank() != 4:
        ctx.record_error(node, "initial_state must have rank 4")
        return
    if packed:
        if state is None:
            return
        value_heads, value_size, key_size = state[1], state[2], state[3]
        if any(isinstance(d, int) and d <= 0 for d in (value_heads, value_size, key_size)):
            ctx.record_error(node, "head counts and head sizes must be positive")
            return
        packed_width = query.shape[-1]
        if all(isinstance(d, int) for d in (packed_width, value_heads, value_size, key_size)):
            if packed_width <= value_heads * value_size or (
                packed_width - value_heads * value_size
            ) % (2 * key_size):
                ctx.record_error(node, "invalid packed QKV head geometry")
                return
        key_heads = (packed_width - value_heads * value_size) // (2 * key_size)
        if isinstance(key_heads, int) and (
            key_heads <= 0 or (isinstance(value_heads, int) and value_heads % key_heads)
        ):
            ctx.record_error(node, "value heads must be a positive multiple of query heads")
            return
        if (
            token_rank == 1
            and _input(node, 3) is None
            and isinstance(query.shape[0], int)
            and isinstance(state[0], int)
            and (state[0] <= 0 or query.shape[0] <= 0 or query.shape[0] % state[0])
        ):
            ctx.record_error(node, "total_tokens must be positive and divisible by batch_size")
            return
        output_heads = value_heads
    else:
        if value.shape is None:
            return
        if value.shape.rank() != rank:
            ctx.record_error(node, "query and value must have the same rank")
            return
        key_heads, key_size = query.shape[-2], query.shape[-1]
        value_heads, value_size = value.shape[-2], value.shape[-1]
        if any(
            isinstance(d, int) and d <= 0
            for d in (key_heads, key_size, value_heads, value_size)
        ):
            ctx.record_error(node, "head counts and head sizes must be positive")
            return
        if key.shape is not None and key.shape.rank() != rank:
            ctx.record_error(node, "query and key must have the same rank")
            return
        if isinstance(key_heads, int) and isinstance(value_heads, int):
            output_heads = max(key_heads, value_heads)
        elif key_heads == value_heads:
            output_heads = value_heads
        else:
            output_heads = ctx.new_symbolic_dim()
    ctx.set_shape_and_dtype(
        node.outputs[0],
        ir.Shape([*query.shape.dims[:token_rank], output_heads, value_size]),
        query.dtype,
    )
    offsets, counts = _input(node, 3), _input(node, 9)
    if counts is not None and counts.shape is not None and counts.shape.rank() == 1:
        batch: int | ir.SymbolicDim = counts.shape[0]
    elif token_rank == 2:
        batch = query.shape[0]
    elif state is not None:
        batch = state[0]
    elif offsets is not None and offsets.shape is not None and offsets.shape.rank() == 1:
        batch = offsets.shape[0] - 1
    else:
        batch = ctx.new_symbolic_dim()
    if len(node.outputs) > 1 and node.outputs[1].name != "":
        ctx.set_shape_and_dtype(
            node.outputs[1],
            state
            if state is not None
            else ir.Shape([batch, value_heads, value_size, key_size]),
            ir.DataType.FLOAT,
        )
    if len(node.outputs) > 2 and node.outputs[2].name != "":
        width = capacity * (value_heads + key_heads * key_size + value_heads * value_size)
        if isinstance(width, int) and width > 2**63 - 1:
            ctx.record_error(node, "state_update width overflows int64")
            return
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([batch, width]), ir.DataType.FLOAT)
