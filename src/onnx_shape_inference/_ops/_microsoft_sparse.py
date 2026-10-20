# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for selected-index sparse attention contrib operators."""

from __future__ import annotations

__all__: list[str] = []

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _microsoft_stateful as _stateful

_reg = _registry.registry.register


@_reg("com.microsoft", "DynamicSparseAttention", since_version=1)
def infer_dynamic_sparse_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (query,) = _context.check_inputs(node, "query")
    for i, name in (
        (7, "selected_indices"),
        (8, "selected_counts"),
        (9, "seqlens_k"),
        (10, "total_sequence_length"),
    ):
        _stateful._required(node, i, name)
    heads = _context.require_attr(node, "num_heads").as_int()
    kv_heads = _context.require_attr(node, "kv_num_heads").as_int()
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        ctx.record_error(node, "kv_num_heads must be positive and divide num_heads")
        return
    key, value = _stateful._input(node, 1), _stateful._input(node, 2)
    if (key is None) != (value is None):
        raise _context.OpUsageError(
            node, "key and value must both be present or both be omitted"
        )
    shape = query.shape
    head_size: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    if shape is not None:
        if shape.rank() != 3:
            ctx.record_error(node, "query must have rank 3")
            return
        divisor = heads if value is not None else heads + 2 * kv_heads
        if isinstance(shape[2], int) and shape[2] % divisor:
            ctx.record_error(node, "query width must be divisible by the head count")
            return
        head_size = shape[2] // divisor
        shape = ir.Shape([shape[0], shape[1], heads * head_size])
    ctx.set_shape_and_dtype(node.outputs[0], shape, query.dtype)
    past_key, past_value = _stateful._input(node, 3), _stateful._input(node, 4)
    total: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    if past_key is None and past_value is None:
        length = ctx.get_symbolic_value(node.inputs[10])
        if length is not None:
            if len(length) != 1:
                ctx.record_error(
                    node, "total_sequence_length must contain exactly one element"
                )
                return
            total = length[0]
    for i, past in ((1, past_key), (2, past_value)):
        if len(node.outputs) <= i or node.outputs[i].name == "":
            continue
        cache_shape = past.shape if past is not None else None
        if cache_shape is not None and cache_shape.rank() != 4:
            ctx.record_error(node, "past cache must have rank 4")
            return
        if cache_shape is None and query.shape is not None:
            cache_shape = ir.Shape([query.shape[0], kv_heads, total, head_size])
        ctx.set_shape_and_dtype(node.outputs[i], cache_shape, query.dtype)


def _indexer_policy(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> tuple[bool, int, int] | None:
    policy = _context.require_attr(node, "policy_mode").as_string()
    ratio = _context.require_attr(node, "compress_ratio").as_int()
    if policy not in ("qsa", "csa") or not 0 < ratio <= 2**31 - 1:
        ctx.record_error(node, "policy_mode must be qsa or csa and compress_ratio positive")
        return None
    qsa = policy == "qsa"
    name = "token_budget" if qsa else "index_topk"
    count = _context.require_attr(node, name).as_int()
    capacity = count + ratio - 1 if qsa else count
    if count <= 0 or capacity > 2**31 - 1 or (qsa and count % ratio):
        ctx.record_error(
            node,
            "selection count must fit int32; token_budget must be divisible by compress_ratio",
        )
        return None
    forbidden = ("index_topk", "head_weight_scale") if qsa else ("token_budget",)
    if any(name in node.attributes for name in forbidden):
        ctx.record_error(node, "selection attributes are incompatible with policy_mode")
        return None
    return qsa, ratio, capacity


@_reg("com.microsoft", "SparseAttentionIndexer", since_version=1)
@_reg("com.microsoft", "PackedSparseAttentionIndexer", since_version=1)
def infer_sparse_attention_indexer(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    policy = _indexer_policy(ctx, node)
    if policy is None:
        return
    qsa, ratio, capacity = policy
    packed = node.op_type == "PackedSparseAttentionIndexer"
    required = {
        0: "query",
        2: "query_norm_weight",
        3: "key_norm_weight",
        4: "cos_cache",
        5: "sin_cache",
    }
    required.update(
        {
            6: "cumulative_sequence_lengths",
            7: "past_sequence_lengths",
            12: "past_key_state",
            13: "past_kv_buffer",
            15: "past_state_lengths",
        }
        if packed
        else {7: "past_key"}
    )
    if not qsa:
        required.update(
            {1: "key", 8: "gate", 9: "position_bias", 10: "head_weights", 11: "position_ids"}
        )
        required[14 if packed else 13] = "past_gate_buffer" if packed else "past_proj_buffer"
    for i, name in required.items():
        _stateful._required(node, i, name)
    forbidden_inputs = (8, 9, 10, 11, 14) if packed else (8, 9, 10, 11, 13)
    if qsa and any(_stateful._input(node, i) is not None for i in forbidden_inputs):
        raise _context.OpUsageError(node, "CSA-only inputs must be omitted for qsa")
    if not qsa and not packed and _stateful._input(node, 6) is not None:
        raise _context.OpUsageError(node, "mask is only supported for qsa")
    query = node.inputs[0]
    rank = 2 if packed else 3
    if query.shape is not None and query.shape.rank() != rank:
        ctx.record_error(node, f"query must have rank {rank}")
        return
    norm = node.inputs[2]
    if norm.shape is not None and norm.shape.rank() != 1:
        ctx.record_error(node, "query_norm_weight must have rank 1")
        return
    for i in (4, 5):
        if node.inputs[i].shape is not None and node.inputs[i].shape.rank() not in (2, 3):
            ctx.record_error(node, "rotary caches must have rank 2 or 3")
            return
    head_size = norm.shape[0] if norm.shape is not None else ctx.new_symbolic_dim()
    if isinstance(head_size, int) and head_size <= 0:
        ctx.record_error(node, "head_size must be positive")
        return
    if query.shape is not None and isinstance(head_size, int):
        width = query.shape[-1]
        if isinstance(width, int) and (
            width <= 0
            or width % head_size
            or (qsa and _stateful._input(node, 1) is None and width // head_size < 2)
        ):
            ctx.record_error(node, "query width is incompatible with head_size")
            return
    selected_shape = (
        ir.Shape([*query.shape.dims[:-1], capacity]) if query.shape is not None else None
    )
    ctx.set_shape_and_dtype(node.outputs[0], selected_shape, ir.DataType.INT32)
    if packed:
        _infer_packed_indexer_state(ctx, node, qsa)
        return
    if len(node.outputs) != (2 if qsa else 3):
        raise _context.OpUsageError(node, "qsa requires 2 outputs; csa requires 3")
    past = node.inputs[7]
    if past.shape is not None and past.shape.rank() != 3:
        ctx.record_error(node, "past_key must have rank 3")
        return
    batch = (
        query.shape[0]
        if query.shape is not None
        else past.shape[0]
        if past.shape is not None
        else ctx.new_symbolic_dim()
    )
    sequence = query.shape[1] if query.shape is not None else None
    present_shape: ir.Shape | None = None
    fixed_cache = _stateful._input(node, 12) is not None
    if qsa:
        if past.shape is not None:
            length = (
                past.shape[1]
                if fixed_cache
                else past.shape[1] + sequence
                if sequence is not None
                else ctx.new_symbolic_dim()
            )
            present_shape = ir.Shape([batch, length, head_size])
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, query.dtype)
        return
    buffer = node.inputs[13]
    buffer_shape = buffer.shape
    if buffer_shape is not None and buffer_shape.rank() != 4:
        ctx.record_error(node, "past_proj_buffer must have rank 4")
        return
    new_entries: int | None = None
    buffer_length: int | ir.SymbolicDim = ctx.new_symbolic_dim()
    if buffer_shape is not None:
        if (
            (isinstance(buffer_shape[0], int) and buffer_shape[0] != 2)
            or (
                isinstance(batch, int)
                and isinstance(buffer_shape[1], int)
                and buffer_shape[1] != batch
            )
            or (
                isinstance(head_size, int)
                and isinstance(buffer_shape[3], int)
                and buffer_shape[3] != 2 * head_size
            )
        ):
            ctx.record_error(
                node, "past_proj_buffer must have shape (2, batch, length, 2 * head_size)"
            )
            return
        previous = buffer_shape[2]
        if isinstance(previous, int) and not 0 <= previous < 2 * ratio:
            ctx.record_error(
                node, "past_proj_buffer length must be in [0, 2 * compress_ratio)"
            )
            return
        if isinstance(previous, int) and isinstance(sequence, int):
            pending = previous % ratio + sequence
            new_entries = pending // ratio
            buffer_length = ratio + pending % ratio if new_entries else previous + sequence
    if past.shape is not None:
        length = (
            past.shape[1]
            if fixed_cache
            else past.shape[1] + new_entries
            if new_entries is not None
            else ctx.new_symbolic_dim()
        )
        present_shape = ir.Shape([batch, length, head_size])
    ctx.set_shape_and_dtype(node.outputs[1], present_shape, query.dtype)
    output_buffer = (
        ir.Shape([buffer_shape[0], batch, buffer_length, buffer_shape[3]])
        if buffer_shape is not None
        else None
    )
    ctx.set_shape_and_dtype(node.outputs[2], output_buffer, query.dtype)


def _infer_packed_indexer_state(
    ctx: _context.ShapeInferenceContext, node: ir.Node, qsa: bool
) -> None:
    if len(node.outputs) not in (6, 7):
        raise _context.OpUsageError(node, "packed indexer requires 6 or 7 output slots")
    if (node.outputs[4].name != "") == qsa:
        raise _context.OpUsageError(
            node, "present_gate_buffer must be present exactly for csa"
        )
    state_capacity = _context.require_attr(node, "state_capacity").as_int()
    update_capacity = _stateful._int_attr(node, "state_update_capacity", 0)
    if state_capacity <= 0 or not 0 <= update_capacity <= 8:
        ctx.record_error(node, "invalid state_capacity or state_update_capacity")
        return
    if not qsa and (
        update_capacity
        or _stateful._input(node, 16) is not None
        or _stateful._input(node, 17) is not None
        or (len(node.outputs) > 6 and node.outputs[6].name != "")
    ):
        raise _context.OpUsageError(node, "state update capture is only supported for qsa")
    if update_capacity > 0:
        _stateful._required(node, 16, "state_update_capture_count")
    query = node.inputs[0]
    counts_shape = ir.Shape([query.shape[0]]) if query.shape is not None else None
    ctx.set_shape_and_dtype(node.outputs[1], counts_shape, ir.DataType.INT32)
    for output_index, input_index, rank in ((2, 12, 3), (3, 13, 3), (4, 14, 3), (5, 15, 2)):
        source = _stateful._input(node, input_index)
        if source is None or node.outputs[output_index].name == "":
            continue
        if source.shape is not None and source.shape.rank() != rank:
            ctx.record_error(node, f"state input {input_index} must have rank {rank}")
            return
        dtype = ir.DataType.INT32 if output_index == 5 else source.dtype
        ctx.set_shape_and_dtype(node.outputs[output_index], source.shape, dtype)
    ratio = _stateful._int_attr(node, "compress_ratio", 0)
    for source, expected in (
        (node.inputs[12], state_capacity),
        (node.inputs[13], 2 * ratio - 1),
        (node.inputs[15], 2),
    ):
        if source.shape is not None and isinstance(source.shape[1], int):
            if source.shape[1] != expected:
                ctx.record_error(
                    node, "fixed state dimensions disagree with indexer attributes"
                )
                return
    if len(node.outputs) > 6 and node.outputs[6].name != "":
        state = node.inputs[12]
        shape = (
            ir.Shape([state.shape[0], update_capacity, state.shape[2]])
            if state.shape is not None
            else None
        )
        ctx.set_shape_and_dtype(node.outputs[6], shape, state.dtype)
