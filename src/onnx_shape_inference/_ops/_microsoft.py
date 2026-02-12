# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for com.microsoft custom operators.

These operators are defined by ONNX Runtime and commonly appear in
optimized transformer models.
"""

from __future__ import annotations

__all__: list[str] = []

import onnx_ir as ir

from onnx_shape_inference import _context, _registry
from onnx_shape_inference._ops import _matmul

_MSFT = "com.microsoft"
_reg = _registry.registry.register

# ---------------------------------------------------------------------------
# Passthrough ops: output[0] shape/dtype = input[0] shape/dtype
# ---------------------------------------------------------------------------


@_reg(_MSFT, "BiasGelu", since_version=1)
@_reg(_MSFT, "FastGelu", since_version=1)
@_reg(_MSFT, "Gelu", since_version=1)
@_reg(_MSFT, "QuickGelu", since_version=1)
@_reg(_MSFT, "BiasAdd", since_version=1)
@_reg(_MSFT, "LongformerAttention", since_version=1)
@_reg(_MSFT, "GroupNorm", since_version=1)
def _infer_passthrough(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Output shape and dtype are identical to the first input."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


# ---------------------------------------------------------------------------
# SkipGroupNorm: output[0] = input[0], optional output[1] = input[0]
# ---------------------------------------------------------------------------


@_reg(_MSFT, "SkipGroupNorm", since_version=1)
def infer_skip_group_norm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        ctx.set_shape_and_dtype(node.outputs[1], x.shape, x.dtype)


# ---------------------------------------------------------------------------
# BiasSplitGelu: [B, S, H] -> [B, S, H/2] (bias determines hidden size)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "BiasSplitGelu", since_version=1)
def infer_bias_split_gelu(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (x, bias) = _context.check_inputs(node, "input", "bias")

    output_shape: ir.Shape | None = None
    if x.shape is not None and bias.shape is not None and bias.shape.rank() == 1:
        bias_len = bias.shape[0]
        if isinstance(bias_len, int):
            dims: list[int | ir.SymbolicDim] = list(x.shape.dims)
            dims[-1] = bias_len // 2
            output_shape = ir.Shape(dims)
    if output_shape is None:
        output_shape = x.shape

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


# ---------------------------------------------------------------------------
# MatMul-like ops: GemmFastGelu, GemmFloat8
# ---------------------------------------------------------------------------


@_reg(_MSFT, "GemmFastGelu", since_version=1)
@_reg(_MSFT, "GemmFloat8", since_version=1)
def _infer_gemm_like(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    (a, b) = _context.check_inputs(node, "X", "W")
    _matmul._matmul_shape(ctx, node, a.shape, b.shape, a.dtype)


# ---------------------------------------------------------------------------
# RotaryEmbedding: 1-3 outputs depending on export mode
# ---------------------------------------------------------------------------


@_reg(_MSFT, "RotaryEmbedding", since_version=1)
def infer_rotary_embedding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """com.microsoft RotaryEmbedding (ONNX Runtime contrib op).

    May produce 1-3 outputs. Extra outputs are artefacts from
    ``export_modules_as_functions`` and copy input shapes.
    """
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    inp = node.inputs[0]
    cos_cache = node.inputs[1] if len(node.inputs) > 1 else None

    num_out = len(node.outputs)
    if num_out == 1:
        ctx.set_shape_and_dtype(node.outputs[0], inp.shape, inp.dtype)
    elif num_out == 2:
        if cos_cache is not None:
            ctx.set_shape_and_dtype(node.outputs[0], cos_cache.shape, cos_cache.dtype)
        ctx.set_shape_and_dtype(node.outputs[1], inp.shape, inp.dtype)
    elif num_out >= 3:
        if cos_cache is not None:
            ctx.set_shape_and_dtype(node.outputs[0], cos_cache.shape, cos_cache.dtype)
            ctx.set_shape_and_dtype(node.outputs[1], cos_cache.shape, cos_cache.dtype)
        ctx.set_shape_and_dtype(node.outputs[2], inp.shape, inp.dtype)


# ---------------------------------------------------------------------------
# LayerNormalization (com.microsoft version, pre-standard)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "LayerNormalization", since_version=1)
@_reg(_MSFT, "SimplifiedLayerNormalization", since_version=1)
def _infer_ms_layer_norm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """LayerNormalization / SimplifiedLayerNormalization.

    output[0] = input[0] shape
    output[1,2] = mean/inv_std_dev: input[:axis] with trailing 1s
    """
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)

    if (len(node.outputs) > 1 or len(node.outputs) > 2) and x.shape is not None:
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr is not None else -1
        rank = x.shape.rank()
        if axis < 0:
            axis += rank

        reduced_dims: list[int | ir.SymbolicDim] = list(x.shape.dims[:axis]) + [1] * (
            rank - axis
        )
        reduced_shape = ir.Shape(reduced_dims)

        # Mean/inv_std_dev use float32 when input is float16/bfloat16
        mean_dtype = x.dtype
        if mean_dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
            mean_dtype = ir.DataType.FLOAT

        if len(node.outputs) > 1 and node.outputs[1] is not None:
            ctx.set_shape_and_dtype(node.outputs[1], reduced_shape, mean_dtype)
        if len(node.outputs) > 2 and node.outputs[2] is not None:
            ctx.set_shape_and_dtype(node.outputs[2], reduced_shape, mean_dtype)


# ---------------------------------------------------------------------------
# SkipLayerNormalization / SkipSimplifiedLayerNormalization
# ---------------------------------------------------------------------------


@_reg(_MSFT, "SkipLayerNormalization", since_version=1)
@_reg(_MSFT, "SkipSimplifiedLayerNormalization", since_version=1)
def _infer_skip_layer_norm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """output[0] = input[0] shape, optional output[3] = input[0] shape."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)
    # output[3] is optional skip connection passthrough
    if len(node.outputs) > 3 and node.outputs[3] is not None:
        ctx.set_shape_and_dtype(node.outputs[3], x.shape, x.dtype)


# ---------------------------------------------------------------------------
# EmbedLayerNormalization
# ---------------------------------------------------------------------------


@_reg(_MSFT, "EmbedLayerNormalization", since_version=1)
def infer_embed_layer_norm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """[B, S] + [vocab, H] -> [B, S, H]."""
    if len(node.inputs) < 3 or node.inputs[0] is None or node.inputs[2] is None:
        raise _context.OpUsageError(node, "Expected input_ids and word_embedding inputs")
    input_ids = node.inputs[0]
    word_emb = node.inputs[2]

    output_shape: ir.Shape | None = None
    output_dtype = word_emb.dtype
    if input_ids.shape is not None and word_emb.shape is not None:
        if input_ids.shape.rank() == 2 and word_emb.shape.rank() == 2:
            output_shape = ir.Shape(
                [input_ids.shape[0], input_ids.shape[1], word_emb.shape[1]]
            )

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
    # output[1]: mask_index = [batch_size]
    if len(node.outputs) > 1 and node.outputs[1] is not None and input_ids.shape is not None:
        ctx.set_shape_and_dtype(
            node.outputs[1], ir.Shape([input_ids.shape[0]]), ir.DataType.INT32
        )
    # output[2]: optional embedding_sum = same as output[0]
    if len(node.outputs) > 2 and node.outputs[2] is not None:
        ctx.set_shape_and_dtype(node.outputs[2], output_shape, output_dtype)


# ---------------------------------------------------------------------------
# Attention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "Attention", since_version=1)
def infer_ms_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """com.microsoft Attention: [B, S, H] -> [B, S, H/3] or qkv_hidden_sizes[2]."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    weights = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1] is not None else None
    bias = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None

    if x.shape is None or x.shape.rank() != 3:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)
        return

    # Determine output hidden size
    tripled_hidden: int | ir.SymbolicDim | None = None
    if bias is not None and bias.shape is not None and bias.shape.rank() == 1:
        tripled_hidden = bias.shape[0]
    elif weights is not None and weights.shape is not None and weights.shape.rank() >= 2:
        tripled_hidden = weights.shape[1]

    qkv_attr = node.attributes.get("qkv_hidden_sizes")
    out_dims: list[int | ir.SymbolicDim] = list(x.shape.dims)
    if qkv_attr is not None:
        sizes = qkv_attr.as_ints()
        if sizes is not None and len(sizes) == 3:
            out_dims[2] = int(sizes[2])
    elif isinstance(tripled_hidden, int):
        out_dims[2] = tripled_hidden // 3

    output_shape = ir.Shape(out_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)

    # output[1]: present state
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        num_heads_attr = node.attributes.get("num_heads")
        if num_heads_attr is not None:
            num_heads = num_heads_attr.as_int()
            past = (
                node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None
            )
            if past is not None and past.shape is not None and past.shape.rank() == 5:
                present_dims: list[int | ir.SymbolicDim] = list(past.shape.dims)
                ctx.set_shape_and_dtype(node.outputs[1], ir.Shape(present_dims), x.dtype)
            else:
                head_size = (
                    x.shape[2] // num_heads
                    if isinstance(x.shape[2], int)
                    else ctx.new_symbolic_dim()
                )
                present_shape = ir.Shape([2, x.shape[0], num_heads, x.shape[1], head_size])
                ctx.set_shape_and_dtype(node.outputs[1], present_shape, x.dtype)


# ---------------------------------------------------------------------------
# MultiHeadAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "MultiHeadAttention", since_version=1)
def infer_ms_multi_head_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """com.microsoft MultiHeadAttention."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input (query)")
    query = node.inputs[0]

    if query.shape is None:
        return

    q_rank = query.shape.rank()

    # 3D query: [B, Sq, Hq]
    if q_rank == 3:
        out_dims: list[int | ir.SymbolicDim] = list(query.shape.dims)
        value = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None
        if value is not None and value.shape is not None and value.shape.rank() == 3:
            out_dims[2] = value.shape[2]
        output_shape = ir.Shape(out_dims)
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], output_shape, query.dtype)

        # Present key/value
        key = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1] is not None else None
        total_seq = (
            key.shape[1]
            if key is not None and key.shape is not None and key.shape.rank() == 3
            else None
        )
        _set_mha_present(ctx, node, query, total_seq)

    # 5D query: [B, Nh, Sq, 3, Hd] packed format
    elif q_rank == 5:
        nh = query.shape[2]
        hd = query.shape[4]
        if isinstance(nh, int) and isinstance(hd, int):
            out_hidden: int | ir.SymbolicDim = nh * hd
        else:
            out_hidden = ctx.new_symbolic_dim()
        output_shape = ir.Shape([query.shape[0], query.shape[1], out_hidden])
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], output_shape, query.dtype)

        _set_mha_present(ctx, node, query, query.shape[1])

    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], query.shape, query.dtype)


def _set_mha_present(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    query: ir.Value,
    total_seq: int | ir.SymbolicDim | None,
) -> None:
    """Set present_key and present_value shapes for MultiHeadAttention."""
    if len(node.outputs) <= 1:
        return
    num_heads_attr = node.attributes.get("num_heads")
    if num_heads_attr is None or query.shape is None:
        return
    num_heads = num_heads_attr.as_int()
    batch = query.shape[0]

    q_rank = query.shape.rank()
    if q_rank == 3 and isinstance(query.shape[2], int):
        head_size: int | ir.SymbolicDim = query.shape[2] // num_heads
    elif q_rank == 5:
        head_size = query.shape[4]
    else:
        head_size = ctx.new_symbolic_dim()

    # Extend total_seq with past
    past = node.inputs[6] if len(node.inputs) > 6 and node.inputs[6] is not None else None
    if past is not None and past.shape is not None and total_seq is not None:
        past_seq = past.shape[2]
        if isinstance(past_seq, int) and isinstance(total_seq, int):
            total_seq = past_seq + total_seq
        else:
            total_seq = ctx.new_symbolic_dim()

    if total_seq is None:
        total_seq = ctx.new_symbolic_dim()

    present_shape = ir.Shape([batch, num_heads, total_seq, head_size])
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, query.dtype)
    if len(node.outputs) > 2 and node.outputs[2] is not None:
        ctx.set_shape_and_dtype(node.outputs[2], present_shape, query.dtype)


# ---------------------------------------------------------------------------
# PackedAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "PackedAttention", since_version=1)
def infer_packed_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """[token_count, H] -> [token_count, H/3] or qkv_hidden_sizes[2]."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    weights = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1] is not None else None
    bias = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None

    if x.shape is None or x.shape.rank() != 2:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)
        return

    tripled_hidden: int | ir.SymbolicDim | None = None
    if bias is not None and bias.shape is not None and bias.shape.rank() == 1:
        tripled_hidden = bias.shape[0]
    elif weights is not None and weights.shape is not None and weights.shape.rank() >= 2:
        tripled_hidden = weights.shape[1]

    out_dims: list[int | ir.SymbolicDim] = list(x.shape.dims)
    qkv_attr = node.attributes.get("qkv_hidden_sizes")
    if qkv_attr is not None:
        sizes = qkv_attr.as_ints()
        if sizes is not None and len(sizes) == 3:
            out_dims[1] = int(sizes[2])
    elif isinstance(tripled_hidden, int):
        out_dims[1] = tripled_hidden // 3

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(out_dims), x.dtype)


# ---------------------------------------------------------------------------
# PackedMultiHeadAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "PackedMultiHeadAttention", since_version=1)
def infer_packed_mha(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Value input [token_count, Hv] determines output, or query [B,Nh,S,Hd] -> [B*S, Nh*Hd]."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    value = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None
    if value is not None and value.shape is not None and value.shape.rank() == 2:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], value.shape, node.inputs[0].dtype)
        return

    query = node.inputs[0]
    if query.shape is not None and query.shape.rank() == 4:
        token_dim = query.shape[0]
        hidden = query.shape[1]
        head_dim = query.shape[3]
        if isinstance(hidden, int) and isinstance(head_dim, int):
            out_hidden: int | ir.SymbolicDim = hidden * head_dim
        else:
            out_hidden = ctx.new_symbolic_dim()
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(
                node.outputs[0], ir.Shape([token_dim, out_hidden]), query.dtype
            )
    else:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], query.shape, query.dtype)


# ---------------------------------------------------------------------------
# DecoderMaskedMultiHeadAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "DecoderMaskedMultiHeadAttention", since_version=1)
def infer_decoder_masked_mha(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input (query)")
    query = node.inputs[0]

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], query.shape, query.dtype)

    # Present key/value from past
    past = node.inputs[5] if len(node.inputs) > 5 and node.inputs[5] is not None else None
    if past is not None and past.shape is not None:
        if len(node.outputs) > 1 and node.outputs[1] is not None:
            ctx.set_shape_and_dtype(node.outputs[1], past.shape, query.dtype)
        if len(node.outputs) > 2 and node.outputs[2] is not None:
            ctx.set_shape_and_dtype(node.outputs[2], past.shape, query.dtype)


# ---------------------------------------------------------------------------
# GatedRelativePositionBias (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "GatedRelativePositionBias", since_version=1)
def infer_gated_relative_position_bias(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    """Output: [B, num_heads, S, S]."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    num_heads_attr = node.attributes.get("num_heads")
    if num_heads_attr is None:
        return
    num_heads = num_heads_attr.as_int()

    # Try token_offset (input 6) first, then query_layer (input 0)
    token_offset = (
        node.inputs[6] if len(node.inputs) > 6 and node.inputs[6] is not None else None
    )
    if (
        token_offset is not None
        and token_offset.shape is not None
        and token_offset.shape.rank() == 2
    ):
        batch = token_offset.shape[0]
        seq_len = token_offset.shape[1]
    elif node.inputs[0].shape is not None and node.inputs[0].shape.rank() == 3:
        batch = node.inputs[0].shape[0]
        seq_len = node.inputs[0].shape[1]
    else:
        return

    output_shape = ir.Shape([batch, num_heads, seq_len, seq_len])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, node.inputs[0].dtype)


# ---------------------------------------------------------------------------
# RemovePadding (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "RemovePadding", since_version=1)
def infer_remove_padding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """[B, S, H] -> [token_count, H] + metadata outputs."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]

    if x.shape is None or x.shape.rank() != 3:
        return

    token_count = ctx.new_symbolic_dim()
    hidden = x.shape[2]
    batch = x.shape[0]
    seq = x.shape[1]

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape([token_count, hidden]), x.dtype)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        ctx.set_shape_and_dtype(node.outputs[1], ir.Shape([batch, seq]), ir.DataType.INT32)
    if len(node.outputs) > 2 and node.outputs[2] is not None:
        batch_plus_1 = ctx.new_symbolic_dim()
        ctx.set_shape_and_dtype(node.outputs[2], ir.Shape([batch_plus_1]), ir.DataType.INT32)
    if len(node.outputs) > 3 and node.outputs[3] is not None:
        ctx.set_shape_and_dtype(node.outputs[3], ir.Shape([1]), ir.DataType.INT32)


# ---------------------------------------------------------------------------
# RestorePadding (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "RestorePadding", since_version=1)
def infer_restore_padding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """[token_count, H] + [B, S] -> [B, S, H]."""
    (x, token_offset) = _context.check_inputs(node, "input", "token_offset")

    if (
        x.shape is not None
        and x.shape.rank() == 2
        and token_offset.shape is not None
        and token_offset.shape.rank() == 2
    ):
        output_shape = ir.Shape([token_offset.shape[0], token_offset.shape[1], x.shape[1]])
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, x.dtype)


# ---------------------------------------------------------------------------
# GroupQueryAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "GroupQueryAttention", since_version=1)
def infer_group_query_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """com.microsoft GroupQueryAttention.

    Inputs:
      0: query  [B, S, hidden_size] or packed QKV [B, S, d]
      1: key    (optional) [B, kv_S, kv_hidden_size]
      2: value  (optional) [B, kv_S, kv_hidden_size]
      3: past_key   (optional) [B, kv_num_heads, past_S, head_size]
      4: past_value (optional) [B, kv_num_heads, past_S, head_size]
      5: seqlens_k  [B]
      6: total_sequence_length  scalar
      ...

    Outputs:
      0: output       [B, S, hidden_size]
      1: present_key  [B, kv_num_heads, total_S, head_size]
      2: present_value [B, kv_num_heads, total_S, head_size]
      3: output_qk    (optional)
    """
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input (query)")
    query = node.inputs[0]

    num_heads_attr = node.attributes.get("num_heads")
    kv_num_heads_attr = node.attributes.get("kv_num_heads")
    num_heads = num_heads_attr.as_int() if num_heads_attr is not None else None
    kv_num_heads = kv_num_heads_attr.as_int() if kv_num_heads_attr is not None else None

    # Output[0]: [B, S, num_heads * head_size]
    if query.shape is not None and query.shape.rank() == 3:
        hidden = query.shape[2]
        if num_heads is not None and kv_num_heads is not None and isinstance(hidden, int):
            # Check if packed QKV: d = (num_heads + 2*kv_num_heads) * head_size
            packed_total = num_heads + 2 * kv_num_heads
            head_size_check = hidden // packed_total
            if hidden == head_size_check * packed_total:
                out_hidden: int | ir.SymbolicDim = num_heads * head_size_check
            else:
                out_hidden = hidden
        else:
            out_hidden = hidden
        output_shape: ir.Shape | None = ir.Shape([query.shape[0], query.shape[1], out_hidden])
    else:
        output_shape = query.shape

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, query.dtype)

    # Present key/value: [B, kv_num_heads, total_seq, head_size]
    if kv_num_heads is not None and query.shape is not None and query.shape.rank() == 3:
        batch = query.shape[0]

        head_size: int | ir.SymbolicDim = ctx.new_symbolic_dim()
        hidden = query.shape[2]
        if num_heads is not None and isinstance(hidden, int):
            packed_total = num_heads + 2 * kv_num_heads
            head_size_check = hidden // packed_total
            if hidden == head_size_check * packed_total:
                head_size = head_size_check
            else:
                head_size = hidden // num_heads

        total_seq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
        past_key = (
            node.inputs[3] if len(node.inputs) > 3 and node.inputs[3] is not None else None
        )
        if past_key is not None and past_key.shape is not None and past_key.shape.rank() == 4:
            past_seq = past_key.shape[2]
            cur_seq = query.shape[1]
            if isinstance(past_seq, int) and isinstance(cur_seq, int):
                total_seq = past_seq + cur_seq

        present_shape = ir.Shape([batch, kv_num_heads, total_seq, head_size])
        if len(node.outputs) > 1 and node.outputs[1] is not None:
            ctx.set_shape_and_dtype(node.outputs[1], present_shape, query.dtype)
        if len(node.outputs) > 2 and node.outputs[2] is not None:
            ctx.set_shape_and_dtype(node.outputs[2], present_shape, query.dtype)


# ---------------------------------------------------------------------------
# MatMulNBits (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "MatMulNBits", since_version=1)
def infer_matmul_nbits(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """com.microsoft MatMulNBits: A @ dequantize(B).

    Inputs:
      0: A      [..., K]  (not quantized)
      1: B      [N, k_blocks, blob_size]  (packed uint8)
      2: scales [N, k_blocks]
      3: zero_points (optional)
      4: g_idx       (optional, deprecated)
      5: bias        (optional) [N]

    Attributes (required):
      K: int - input feature dimension of the weight matrix
      N: int - output feature dimension of the weight matrix
      block_size: int - quantization block size along K

    Output:
      Y: same rank as A, last dim replaced by N.  dtype = A's dtype.
    """
    (a, _b, _scales) = _context.check_inputs(node, "A", "B", "scales")

    n_dim = _context.require_attr(node, "N").as_int()

    output_shape: ir.Shape | None = None
    if a.shape is not None and a.shape.rank() >= 1:
        out_dims: list[int | ir.SymbolicDim] = [*list(a.shape.dims[:-1]), n_dim]
        output_shape = ir.Shape(out_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, a.dtype)


# ---------------------------------------------------------------------------
# SparseAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "SparseAttention", since_version=1)
def infer_sparse_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Sparse variant of GroupQueryAttention. Same output semantics."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input (query)")
    query = node.inputs[0]

    num_heads_attr = node.attributes.get("num_heads")
    kv_num_heads_attr = node.attributes.get("kv_num_heads")
    kv_num_heads = kv_num_heads_attr.as_int() if kv_num_heads_attr is not None else None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], query.shape, query.dtype)

    # Present key/value: [B, kv_num_heads, total_seq, head_size]
    if kv_num_heads is not None and query.shape is not None and query.shape.rank() == 3:
        batch = query.shape[0]
        num_heads = num_heads_attr.as_int() if num_heads_attr is not None else None
        head_size: int | ir.SymbolicDim = ctx.new_symbolic_dim()
        if num_heads is not None and isinstance(query.shape[2], int):
            head_size = query.shape[2] // num_heads
        total_seq: int | ir.SymbolicDim = ctx.new_symbolic_dim()
        present_shape = ir.Shape([batch, kv_num_heads, total_seq, head_size])
        if len(node.outputs) > 1 and node.outputs[1] is not None:
            ctx.set_shape_and_dtype(node.outputs[1], present_shape, query.dtype)
        if len(node.outputs) > 2 and node.outputs[2] is not None:
            ctx.set_shape_and_dtype(node.outputs[2], present_shape, query.dtype)


# ---------------------------------------------------------------------------
# FusedMatMul / TransposeMatMul (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "FusedMatMul", since_version=1)
@_reg(_MSFT, "TransposeMatMul", since_version=1)
def infer_fused_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """FusedMatMul: MatMul with optional transpose on inputs.

    Attributes: transA, transB (default 0).
    """
    (a, b) = _context.check_inputs(node, "A", "B")

    shape_a = a.shape
    shape_b = b.shape

    trans_a_attr = node.attributes.get("transA")
    trans_b_attr = node.attributes.get("transB")
    trans_a = trans_a_attr.as_int() if trans_a_attr is not None else 0
    trans_b = trans_b_attr.as_int() if trans_b_attr is not None else 0

    if trans_a and shape_a is not None and shape_a.rank() >= 2:
        dims = list(shape_a.dims)
        dims[-2], dims[-1] = dims[-1], dims[-2]
        shape_a = ir.Shape(dims)

    if trans_b and shape_b is not None and shape_b.rank() >= 2:
        dims = list(shape_b.dims)
        dims[-2], dims[-1] = dims[-1], dims[-2]
        shape_b = ir.Shape(dims)

    _matmul._matmul_shape(ctx, node, shape_a, shape_b, a.dtype)


# ---------------------------------------------------------------------------
# MoE / QMoE (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "MoE", since_version=1)
@_reg(_MSFT, "QMoE", since_version=1)
def infer_moe(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Mixture of Experts. Output shape = input shape."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


# ---------------------------------------------------------------------------
# QAttention (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QAttention", since_version=1)
def infer_q_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Quantized Attention.

    Inputs:
      0: input   [B, S, input_hidden_size] (int8/uint8)
      1: weight  [input_hidden_size, 3*hidden_size] (int8/uint8)
      2: bias    [3*hidden_size] (float)
      ...
      8: past (optional) [2, B, num_heads, past_S, head_size]

    Outputs:
      0: output  [B, S, hidden_size] (float)
      1: present (optional) [2, B, num_heads, total_S, head_size]
    """
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    bias = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None

    # Output dtype is float (T3 type from bias/scales)
    output_dtype = bias.dtype if bias is not None else ir.DataType.FLOAT

    if x.shape is None or x.shape.rank() != 3:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], x.shape, output_dtype)
        return

    # Determine hidden_size from bias (3*hidden_size)
    hidden: int | ir.SymbolicDim = x.shape[2]
    if bias is not None and bias.shape is not None and bias.shape.rank() == 1:
        tripled = bias.shape[0]
        if isinstance(tripled, int):
            hidden = tripled // 3

    output_shape = ir.Shape([x.shape[0], x.shape[1], hidden])
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # Present state
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        num_heads = _context.require_attr(node, "num_heads").as_int()
        head_size: int | ir.SymbolicDim = (
            hidden // num_heads if isinstance(hidden, int) else ctx.new_symbolic_dim()
        )
        past = node.inputs[8] if len(node.inputs) > 8 and node.inputs[8] is not None else None
        total_seq: int | ir.SymbolicDim
        if past is not None and past.shape is not None and past.shape.rank() == 5:
            past_seq = past.shape[3]
            cur_seq = x.shape[1]
            if isinstance(past_seq, int) and isinstance(cur_seq, int):
                total_seq = past_seq + cur_seq
            else:
                total_seq = ctx.new_symbolic_dim()
        else:
            total_seq = x.shape[1]
        present_shape = ir.Shape([2, x.shape[0], num_heads, total_seq, head_size])
        ctx.set_shape_and_dtype(node.outputs[1], present_shape, output_dtype)


# ---------------------------------------------------------------------------
# QGemm (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QGemm", since_version=1)
def infer_q_gemm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Quantized Gemm: Y = alpha * (A @ B) + C.

    Inputs: A(0), a_scale(1), a_zp(2), B(3), b_scale(4), b_zp(5),
            C(6, opt), y_scale(7, opt), y_zp(8, opt)
    Output: Y = (M, N), dtype from y_scale or float.
    """
    if len(node.inputs) < 4 or node.inputs[0] is None or node.inputs[3] is None:
        raise _context.OpUsageError(node, "Expected inputs A and B")
    a = node.inputs[0]
    b = node.inputs[3]

    trans_a_attr = node.attributes.get("transA")
    trans_b_attr = node.attributes.get("transB")
    trans_a = trans_a_attr.as_int() if trans_a_attr is not None else 0
    trans_b = trans_b_attr.as_int() if trans_b_attr is not None else 0

    shape_a = a.shape
    shape_b = b.shape

    if trans_a and shape_a is not None and shape_a.rank() == 2:
        shape_a = ir.Shape([shape_a[1], shape_a[0]])
    if trans_b and shape_b is not None and shape_b.rank() == 2:
        shape_b = ir.Shape([shape_b[1], shape_b[0]])

    # Output dtype: y_scale dtype if provided, otherwise float
    y_scale = node.inputs[7] if len(node.inputs) > 7 and node.inputs[7] is not None else None
    output_dtype = y_scale.dtype if y_scale is not None else ir.DataType.FLOAT

    output_shape: ir.Shape | None = None
    if (
        shape_a is not None
        and shape_b is not None
        and shape_a.rank() == 2
        and shape_b.rank() == 2
    ):
        output_shape = ir.Shape([shape_a[0], shape_b[1]])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# ---------------------------------------------------------------------------
# QLinear elementwise ops (com.microsoft)
# ---------------------------------------------------------------------------


def _qlinear_unary_shape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """QLinear unary ops: output shape = input[0] shape.

    Pattern: X(0), x_scale(1), x_zp(2), y_scale(3), y_zp(4)
    Output dtype = X's dtype (quantized), fallback to x_zp's dtype.
    """
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    output_dtype = x.dtype
    if output_dtype is None:
        x_zp = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None
        output_dtype = x_zp.dtype if x_zp is not None else None
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, output_dtype)


@_reg(_MSFT, "QLinearLeakyRelu", since_version=1)
@_reg(_MSFT, "QLinearSigmoid", since_version=1)
@_reg(_MSFT, "QLinearSoftmax", since_version=1)
@_reg(_MSFT, "QLinearAveragePool", since_version=1)
@_reg(_MSFT, "QLinearGlobalAveragePool", since_version=1)
@_reg(_MSFT, "QLinearReduceMean", since_version=1)
def _infer_qlinear_unary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    _qlinear_unary_shape(ctx, node)


def _qlinear_binary_shape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """QLinear binary ops (Add, Mul): output shape = broadcast(A, B).

    Pattern: A(0), A_scale(1), A_zp(2), B(3), B_scale(4), B_zp(5),
             C_scale(6), C_zp(7)
    Output dtype = A's dtype (quantized), fallback to A_zp's dtype.
    """
    if len(node.inputs) < 4 or node.inputs[0] is None or node.inputs[3] is None:
        raise _context.OpUsageError(node, "Expected inputs A and B")
    a = node.inputs[0]
    b = node.inputs[3]
    from onnx_shape_inference import _broadcast

    output_shape = _broadcast.broadcast_shapes(a.shape, b.shape)
    output_dtype = a.dtype
    if output_dtype is None:
        a_zp = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None
        output_dtype = a_zp.dtype if a_zp is not None else None
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


@_reg(_MSFT, "QLinearAdd", since_version=1)
@_reg(_MSFT, "QLinearMul", since_version=1)
def _infer_qlinear_binary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    _qlinear_binary_shape(ctx, node)


# ---------------------------------------------------------------------------
# QLinearConv (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QLinearConv", since_version=1)
def infer_qlinear_conv(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Quantized Conv. x(0), x_scale(1), x_zp(2), w(3), w_scale(4), w_zp(5), ..."""
    if len(node.inputs) < 4 or node.inputs[0] is None or node.inputs[3] is None:
        raise _context.OpUsageError(node, "Expected inputs x and w")
    x = node.inputs[0]
    w = node.inputs[3]

    output_dtype = x.dtype
    if output_dtype is None:
        x_zp = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] is not None else None
        output_dtype = x_zp.dtype if x_zp is not None else None

    from onnx_shape_inference._ops import _conv

    output_shape = _conv._compute_conv_shape(ctx, node, x, w)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# ---------------------------------------------------------------------------
# QLinearConcat (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QLinearConcat", since_version=1)
def infer_qlinear_concat(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """QLinear Concat. Inputs: Y_scale(0), Y_zp(1), then (tensor, scale, zp) triples.

    Output shape = concat of input tensors along axis.
    """
    axis_attr = node.attributes.get("axis")
    axis = axis_attr.as_int() if axis_attr is not None else 0

    # Extract data tensors: inputs[2], inputs[5], inputs[8], ...
    data_values: list[ir.Value] = []
    i = 2
    while i < len(node.inputs):
        if node.inputs[i] is not None:
            data_values.append(node.inputs[i])
        i += 3  # Skip scale and zp

    if not data_values:
        return

    output_dtype = data_values[0].dtype
    if output_dtype is None:
        # Fallback: first data tensor's zp (inputs[4]) shares the same type
        zp = node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None
        output_dtype = zp.dtype if zp is not None else None

    # Compute concat output shape
    first_shape = data_values[0].shape
    if first_shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    rank = first_shape.rank()
    if axis < 0:
        axis += rank

    concat_dim: int | ir.SymbolicDim = 0
    out_dims: list[int | ir.SymbolicDim] = list(first_shape.dims)
    all_concrete = True
    for v in data_values:
        if v.shape is None or v.shape.rank() != rank:
            all_concrete = False
            break
        dim = v.shape[axis]
        if isinstance(dim, int) and isinstance(concat_dim, int):
            concat_dim += dim
        else:
            all_concrete = False
            break

    if all_concrete:
        out_dims[axis] = concat_dim
        output_shape: ir.Shape | None = ir.Shape(out_dims)
    else:
        out_dims[axis] = ctx.new_symbolic_dim()
        output_shape = ir.Shape(out_dims)

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# ---------------------------------------------------------------------------
# QLinearWhere (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QLinearWhere", since_version=1)
def infer_qlinear_where(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Quantized Where: Z = condition ? X : Y.

    Inputs: condition(0), X(1), x_scale(2), x_zp(3), Y(4), y_scale(5), y_zp(6),
            z_scale(7), z_zp(8)
    Output: broadcast(condition, X, Y)
    """
    if len(node.inputs) < 5 or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected condition, X, Y inputs")
    condition = node.inputs[0]
    x = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1] is not None else None
    y = node.inputs[4] if len(node.inputs) > 4 and node.inputs[4] is not None else None

    from onnx_shape_inference import _broadcast

    output_shape = condition.shape
    if x is not None:
        output_shape = _broadcast.broadcast_shapes(output_shape, x.shape)
    if y is not None:
        output_shape = _broadcast.broadcast_shapes(output_shape, y.shape)

    output_dtype = x.dtype if x is not None else None
    if output_dtype is None and x is not None:
        x_zp = node.inputs[3] if len(node.inputs) > 3 and node.inputs[3] is not None else None
        output_dtype = x_zp.dtype if x_zp is not None else None
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# ---------------------------------------------------------------------------
# QOrdered* ops (com.microsoft) — passthrough shape semantics
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QOrderedGelu", since_version=1)
@_reg(_MSFT, "QOrderedLayerNormalization", since_version=1)
@_reg(_MSFT, "QOrderedAttention", since_version=1)
@_reg(_MSFT, "QOrderedLongformerAttention", since_version=1)
def _infer_qordered_passthrough(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """QOrdered ops: output shape = input[0] shape."""
    if not node.inputs or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")
    x = node.inputs[0]
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], x.shape, x.dtype)


# ---------------------------------------------------------------------------
# QOrderedMatMul (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QOrderedMatMul", since_version=1)
def infer_qordered_matmul(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """QOrdered MatMul. Inputs: A(0), scaleA(1), B(2), scaleB(3), ...

    Output: matmul(A, B) shape.
    """
    if len(node.inputs) < 3 or node.inputs[0] is None or node.inputs[2] is None:
        raise _context.OpUsageError(node, "Expected inputs A and B")
    a = node.inputs[0]
    b = node.inputs[2]
    _matmul._matmul_shape(ctx, node, a.shape, b.shape, a.dtype)


# ---------------------------------------------------------------------------
# QEmbedLayerNormalization (com.microsoft)
# ---------------------------------------------------------------------------


@_reg(_MSFT, "QEmbedLayerNormalization", since_version=1)
def infer_qembed_layer_norm(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Quantized EmbedLayerNormalization.

    Inputs: input_ids(0), ..., word_embedding(2), ...
    Output[0]: [B, S, H], Output[1]: mask_index [B]
    """
    if len(node.inputs) < 3 or node.inputs[0] is None or node.inputs[2] is None:
        raise _context.OpUsageError(node, "Expected input_ids and word_embedding inputs")
    input_ids = node.inputs[0]
    word_emb = node.inputs[2]

    output_shape: ir.Shape | None = None
    output_dtype = word_emb.dtype
    if input_ids.shape is not None and word_emb.shape is not None:
        if input_ids.shape.rank() == 2 and word_emb.shape.rank() == 2:
            output_shape = ir.Shape(
                [input_ids.shape[0], input_ids.shape[1], word_emb.shape[1]]
            )

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
    if len(node.outputs) > 1 and node.outputs[1] is not None and input_ids.shape is not None:
        ctx.set_shape_and_dtype(
            node.outputs[1], ir.Shape([input_ids.shape[0]]), ir.DataType.INT32
        )
