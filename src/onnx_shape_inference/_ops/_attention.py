# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for attention operators."""

from __future__ import annotations

__all__ = [
    "infer_attention",
    "infer_flex_attention",
    "infer_linear_attention",
    "infer_rotary_embedding",
]

import onnx_ir as ir

from onnx_shape_inference import _context, _registry

_reg = _registry.registry.register


def _attr_int(node: ir.Node, name: str) -> int | None:
    """Read an INT attribute, or ``None`` when absent."""
    attr = node.attributes.get(name)
    return attr.as_int() if attr is not None else None


def _head_size(
    ctx: _context.ShapeInferenceContext,
    hidden: int | ir.SymbolicDim | None,
    num_heads: int | None,
) -> int | ir.SymbolicDim:
    """Split a packed hidden size into per-head size.

    Returns a concrete ``hidden // num_heads`` when both are known integers and
    divide evenly; otherwise a fresh symbolic dim (the value is data-dependent
    or not statically resolvable).
    """
    if (
        isinstance(hidden, int)
        and isinstance(num_heads, int)
        and num_heads > 0
        and hidden % num_heads == 0
    ):
        return hidden // num_heads
    return ctx.new_symbolic_dim()


@_reg("", "Attention", since_version=23)
def infer_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Attention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Attention.html

    3D: Q=[B,Sq,Hq], K=[B,Sk,Hk], V=[B,Sk,Hv] -> Y=[B,Sq,Hq]
        (Hq = q_num_heads * head_size, output preserves Q's hidden dim)
    4D: Q=[B,Nh,Sq,Hd], K=[B,Nkv,Sk,Hd], V=[B,Nkv,Sk,Hdv] -> Y=[B,Nh,Sq,Hdv]
    present_key/value: if past is provided, present is past shape with seq extended
    """
    (q, k, v) = _context.check_inputs(node, "Q", "K", "V")

    output_dtype = q.dtype

    # Output[0]: Y shape
    # 3D: Y=[B, Sq, q_num_heads * v_head_size]
    # 4D: Y=[B, q_num_heads, Sq, v_head_size]
    if len(node.outputs) > 0:
        output_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            if rank == 3 and v.shape is not None:
                # Output hidden = q_num_heads * v_head_size
                q_nh_attr = node.attributes.get("q_num_heads")
                kv_nh_attr = node.attributes.get("kv_num_heads")
                q_nh = q_nh_attr.as_int() if q_nh_attr is not None else None
                kv_nh = kv_nh_attr.as_int() if kv_nh_attr is not None else None
                v_hidden = v.shape[2]
                if q_nh is not None and kv_nh is not None and isinstance(v_hidden, int):
                    v_head_size = v_hidden // kv_nh
                    out_hidden: int | ir.SymbolicDim = q_nh * v_head_size
                else:
                    out_hidden = v_hidden
                output_shape = ir.Shape([q.shape[0], q.shape[1], out_hidden])
            elif rank == 4 and v.shape is not None:
                output_shape = ir.Shape([q.shape[0], q.shape[1], q.shape[2], v.shape[3]])
            else:
                output_shape = q.shape
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # present_key and present_value
    past_key = node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None
    past_value = (
        node.inputs[5] if len(node.inputs) > 5 and node.inputs[5] is not None else None
    )

    if len(node.outputs) > 1 and node.outputs[1] is not None:
        pk_shape = _attention_present_shape(k, past_key)
        ctx.set_shape_and_dtype(node.outputs[1], pk_shape, output_dtype)

    if len(node.outputs) > 2 and node.outputs[2] is not None:
        pv_shape = _attention_present_shape(v, past_value)
        ctx.set_shape_and_dtype(node.outputs[2], pv_shape, output_dtype)

    # Output[3]: qk_matmul_output = [B, q_num_heads, Sq, total_seq_length]
    if len(node.outputs) > 3 and node.outputs[3] is not None:
        qk_shape: ir.Shape | None = None
        if q.shape is not None:
            rank = q.shape.rank()
            batch = q.shape[0]

            # Get q_num_heads: from attribute (3D) or Q shape dim 1 (4D)
            qk_nh_attr = node.attributes.get("q_num_heads")
            qk_q_nh: int | ir.SymbolicDim | None = None
            if qk_nh_attr is not None:
                qk_q_nh = qk_nh_attr.as_int()
            elif rank == 4:
                qk_q_nh = q.shape[1]

            sq = q.shape[1] if rank == 3 else q.shape[2]

            # total_seq_length: past_seq + current_seq (from K)
            total_seq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
            pk_shape = _attention_present_shape(k, past_key)
            if pk_shape is not None:
                total_seq = pk_shape[pk_shape.rank() - 2]
            elif k.shape is not None:
                k_seq = k.shape[k.shape.rank() - 2]
                total_seq = k_seq

            if qk_q_nh is not None:
                qk_shape = ir.Shape([batch, qk_q_nh, sq, total_seq])
        ctx.set_shape_and_dtype(node.outputs[3], qk_shape, output_dtype)


def _attention_present_shape(current: ir.Value, past: ir.Value | None) -> ir.Shape | None:
    """Compute present_key/present_value shape.

    When past is provided, present matches past shape with seq dim extended.
    When past is None, present = current shape.
    """
    if past is not None and past.shape is not None:
        past_rank = past.shape.rank()
        seq_idx = past_rank - 2
        past_seq = past.shape[seq_idx]
        if current.shape is not None:
            cur_rank = current.shape.rank()
            cur_seq = current.shape[cur_rank - 2]
        else:
            return None
        if isinstance(past_seq, int) and isinstance(cur_seq, int):
            total_seq: int | ir.SymbolicDim = past_seq + cur_seq
        else:
            total_seq = past_seq + cur_seq  # type: ignore[assignment]
        dims = list(past.shape.dims)
        dims[seq_idx] = total_seq
        return ir.Shape(dims)
    if current.shape is None:
        return None
    return current.shape


@_reg("", "RotaryEmbedding", since_version=23)
def infer_rotary_embedding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RotaryEmbedding operator.

    Spec: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
    """
    (input_val,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)

    # Optional second output (output_position_ids)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        position_ids_shape: ir.Shape | None = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            position_ids_shape = node.inputs[1].shape
        ctx.set_shape_and_dtype(node.outputs[1], position_ids_shape, ir.DataType.INT64)


@_reg("", "LinearAttention", since_version=27)
def infer_linear_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for LinearAttention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__LinearAttention.html

    query/key/value are 3-D packed ``[B, T, H*D]``; ``q_num_heads`` and
    ``kv_num_heads`` are required.  Outputs:

    * ``output``        ``[B, T, q_num_heads * d_v]`` where
      ``d_v = value_hidden // kv_num_heads`` (T type = query dtype).
    * ``present_state`` ``[B, kv_num_heads, d_k, d_v]`` with
      ``d_k = key_hidden // kv_num_heads``; equal to ``past_state`` when that
      optional input is provided.
    """
    (q, k, v) = _context.check_inputs(node, "query", "key", "value")
    past_state = (
        node.inputs[3] if len(node.inputs) > 3 and node.inputs[3] is not None else None
    )

    q_num_heads = _attr_int(node, "q_num_heads")
    kv_num_heads = _attr_int(node, "kv_num_heads")
    output_dtype = q.dtype

    # Output 0: [B, T, q_num_heads * d_v]
    if len(node.outputs) > 0:
        output_shape: ir.Shape | None = None
        if q.shape is not None and q.shape.rank() == 3:
            v_hidden = v.shape[2] if v.shape is not None else None
            d_v = _head_size(ctx, v_hidden, kv_num_heads)
            if isinstance(d_v, int) and q_num_heads is not None:
                out_hidden: int | ir.SymbolicDim = q_num_heads * d_v
            else:
                out_hidden = ctx.new_symbolic_dim()
            output_shape = ir.Shape([q.shape[0], q.shape[1], out_hidden])
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # Output 1: present_state [B, kv_num_heads, d_k, d_v]
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        present_dtype = (
            past_state.dtype
            if past_state is not None and past_state.dtype is not None
            else output_dtype
        )
        present_shape: ir.Shape | None = None
        if (
            past_state is not None
            and past_state.shape is not None
            and past_state.shape.rank() == 4
        ):
            # past_state is ground truth for the recurrent state buffer.
            present_shape = past_state.shape
        elif q.shape is not None and q.shape.rank() == 3 and kv_num_heads is not None:
            k_hidden = k.shape[2] if k.shape is not None else None
            v_hidden = v.shape[2] if v.shape is not None else None
            d_k = _head_size(ctx, k_hidden, kv_num_heads)
            d_v = _head_size(ctx, v_hidden, kv_num_heads)
            present_shape = ir.Shape([q.shape[0], kv_num_heads, d_k, d_v])
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, present_dtype)


@_reg("ai.onnx.preview", "FlexAttention", since_version=1)
def infer_flex_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for ai.onnx.preview FlexAttention operator.

    Inputs are rank-4:

    * Q ``[B, q_num_heads, q_seq, head_size]``
    * K ``[B, kv_num_heads, kv_seq, head_size]``
    * V ``[B, kv_num_heads, kv_seq, v_head_size]``

    Output ``Y = [B, q_num_heads, q_seq, v_head_size]`` (same element type as Q).
    """
    (q, _k, v) = _context.check_inputs(node, "Q", "K", "V")

    if len(node.outputs) > 0:
        output_shape: ir.Shape | None
        if (
            q.shape is not None
            and q.shape.rank() == 4
            and v.shape is not None
            and v.shape.rank() == 4
        ):
            output_shape = ir.Shape([q.shape[0], q.shape[1], q.shape[2], v.shape[3]])
        else:
            output_shape = q.shape
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, q.dtype)
