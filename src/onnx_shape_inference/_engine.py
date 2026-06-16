# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference engine.

This module contains the core graph-traversal logic for symbolic shape
inference, decoupled from the ``ir.passes`` framework so that
``infer_symbolic_shapes`` can be called directly without pass-level
error wrapping.
"""

from __future__ import annotations  # makes all annotations lazy strings (Python 3.9 compat)

__all__ = [
    "infer_symbolic_shapes",
]

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import onnx_ir as ir

from onnx_shape_inference import _context, _functions, _registry

if TYPE_CHECKING:
    from onnx_shape_inference._functions import _FuncOutputCache

logger = logging.getLogger(__name__)


def infer_symbolic_shapes(
    model: ir.Model,
    *,
    policy: _context.ShapeMergePolicy = "refine",
    warn_on_missing: bool = True,
) -> ir.Model:
    """Perform symbolic shape inference on the model.

    Traverses every graph in *model* in topological order, applying
    registered shape-inference functions to each node.  The model is
    modified **in place** and also returned for convenience.

    Args:
        model: The model to perform shape inference on.
        policy: How to merge inferred shapes with existing shapes.
        warn_on_missing: If ``True``, log warnings for ops without
            registered shape inference.

    Returns:
        The same *model* object, with shapes updated in place.

    Example::

        import onnx_ir as ir
        from onnx_shape_inference import infer_symbolic_shapes

        model = ir.load("model.onnx")
        model = infer_symbolic_shapes(model)
    """
    _infer_symbolic_shapes(model, policy=policy, warn_on_missing=warn_on_missing)
    return model


def _infer_symbolic_shapes(
    model: ir.Model,
    *,
    policy: _context.ShapeMergePolicy = "refine",
    warn_on_missing: bool = True,
) -> bool:
    """Core implementation that returns whether the model was modified."""
    _registry.registry.collect()

    ctx = _context.ShapeInferenceContext(model.opset_imports, policy=policy)
    # Per-run cache maps (id(function), input_signature, attr_signature) to output
    # (shape, dtype, sym_data): function identity, call-site input shapes/dtypes/sym_data,
    # and call-site attribute values.  A new dict per call ensures no stale hits across
    # separate infer_symbolic_shapes calls.
    inference_cache: _FuncOutputCache = {}
    # Per-run cache of materialized op-schema function bodies, keyed by
    # (domain, op_type, opset, input-type signature).
    op_function_cache: dict = {}

    return _process_graph(
        ctx,
        model.graph,
        warn_on_missing=warn_on_missing,
        model_functions=model.functions,
        inference_cache=inference_cache,
        op_function_cache=op_function_cache,
    )


def _refine_body_input(
    ctx: _context.ShapeInferenceContext,
    body_inp: ir.Value,
    actual_shape: ir.Shape | None,
    actual_dtype: ir.DataType | None,
) -> None:
    """Push an actual input's dtype/shape onto a subgraph formal input.

    The subgraph's declared ``value_info`` is advisory for *shape*: concrete
    dims from the actual call-site input take precedence over symbolic dims
    declared on the body input, while symbolic body dims are kept where the
    actual dim is unknown.

    The body input's declared **dtype** is treated as authoritative and is only
    filled in when it is missing — it is never overridden here.  A genuine
    dtype conflict between the body declaration and the actual input is left for
    the body's own op inference to surface rather than being silently rewritten.
    """
    if actual_dtype is not None and body_inp.dtype is None:
        ctx.set_dtype(body_inp, actual_dtype)

    if actual_shape is None:
        return
    body_shape = body_inp.shape
    if body_shape is None or body_shape.rank() != actual_shape.rank():
        ctx.set_shape(body_inp, actual_shape)
        return

    merged: list[int | ir.SymbolicDim] = []
    for body_dim, actual_dim in zip(body_shape.dims, actual_shape.dims):
        # Prefer a concrete dim from the actual input; otherwise keep whichever
        # side carries a named/concrete value.
        if isinstance(actual_dim, int):
            merged.append(actual_dim)
        elif isinstance(body_dim, int):
            merged.append(body_dim)
        elif isinstance(body_dim, ir.SymbolicDim) and body_dim.value is not None:
            merged.append(body_dim)
        else:
            merged.append(actual_dim)
    ctx.set_shape(body_inp, ir.Shape(merged))


def _propagate_types_to_subgraph_inputs(
    ctx: _context.ShapeInferenceContext, node: ir.Node
) -> None:
    """Propagate types from node inputs to subgraph inputs before subgraph processing.

    Loop body inputs ``[iteration_num, condition, v_0, ..., v_N]`` get their
    types from the corresponding node inputs ``[max_trip_count, cond,
    v_init_0, ..., v_init_N]``, so that the body graph can be inferred with
    correct type information.

    Scan body inputs ``[state_0, ..., scan_slice_0, ...]`` get the state
    initializer shapes directly and the scan-input shapes with the scanned
    axis removed, so concrete state dims (e.g. from a zeros initializer)
    propagate into the body instead of leaving the body's declared symbolic
    dims in place.
    """
    if (node.domain or "") != "":
        return

    if node.op_type == "Loop":
        body_attr = node.attributes.get("body")
        if body_attr is None:
            return
        body_graph = body_attr.as_graph()
        if body_graph is None:
            return
        # Body inputs: [iteration_num, condition, ...loop_carried]
        # Node inputs: [max_trip_count, cond, ...loop_carried_init]
        for j in range(2, min(len(node.inputs), len(body_graph.inputs))):
            init_val = node.inputs[j]
            body_inp = body_graph.inputs[j]
            if init_val is not None and body_inp.dtype is None and init_val.dtype is not None:
                ctx.set_dtype(body_inp, init_val.dtype)
            if init_val is not None and body_inp.shape is None and init_val.shape is not None:
                ctx.set_shape(body_inp, init_val.shape)
    elif node.op_type == "Scan":
        _propagate_types_to_scan_body(ctx, node)


def _propagate_types_to_scan_body(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Bind Scan node input shapes onto the body's formal inputs."""
    body_attr = node.attributes.get("body")
    if body_attr is None:
        return
    body_graph = body_attr.as_graph()
    if body_graph is None:
        return

    num_scan_attr = node.attributes.get("num_scan_inputs")
    if num_scan_attr is None:
        return
    num_scan_inputs = num_scan_attr.as_int()
    num_state = len(node.inputs) - num_scan_inputs
    if num_state < 0:
        return

    scan_input_axes_attr = node.attributes.get("scan_input_axes")
    input_axes = (
        list(scan_input_axes_attr.as_ints()) if scan_input_axes_attr is not None else []
    )

    for j in range(min(len(node.inputs), len(body_graph.inputs))):
        actual = node.inputs[j]
        if actual is None:
            continue
        body_inp = body_graph.inputs[j]
        if j < num_state:
            _refine_body_input(ctx, body_inp, actual.shape, actual.dtype)
        else:
            # Scan slice: drop the scanned axis from the input shape.
            sliced: ir.Shape | None = None
            if actual.shape is not None and actual.shape.rank() is not None:
                idx = j - num_state
                axis = input_axes[idx] if idx < len(input_axes) else 0
                rank = actual.shape.rank()
                if rank > 0:
                    if axis < 0:
                        axis += rank
                    if 0 <= axis < rank:
                        dims = list(actual.shape.dims)
                        del dims[axis]
                        sliced = ir.Shape(dims)
            _refine_body_input(ctx, body_inp, sliced, actual.dtype)


def _emit_missing_warning(domain: str, op_type: str, warned_ops: set[tuple[str, str]]) -> None:
    """Log a warning for an op with no registered shape inference (once per op)."""
    key = (domain, op_type)
    if key not in warned_ops:
        logger.warning("No shape inference registered for %s::%s", domain, op_type)
        warned_ops.add(key)


def _process_graph(
    ctx: _context.ShapeInferenceContext,
    graph: ir.Graph,
    *,
    warn_on_missing: bool = True,
    model_functions: Mapping[tuple[str, str, str], ir.Function] | None = None,
    active_functions: frozenset[tuple[str, str, str]] | None = None,
    inference_cache: _FuncOutputCache | None = None,
    op_function_cache: dict | None = None,
) -> bool:
    """Process a single graph.

    Args:
        ctx: The shape inference context.
        graph: The graph to process.
        warn_on_missing: If ``True``, log warnings for ops without
            registered shape inference.
        model_functions: All local functions defined in the model, used to
            dispatch function-call nodes that have no registered handler.
        active_functions: Set of function keys currently on the call stack,
            used to detect and break recursive function calls.
        inference_cache: Optional per-inference-run cache for function body
            results, keyed by ``(id(function), input_signature, attr_signature)``
            (function identity, call-site input shapes/dtypes/sym_data, call-site
            attribute values).
        op_function_cache: Optional per-inference-run cache of materialized
            op-schema function bodies, keyed by ``(domain, op_type, opset,
            input-type signature)``.

    Returns:
        ``True`` if any shapes were modified.
    """
    modified = False
    warned_ops: set[tuple[str, str]] = set()

    # Fix shapes for initializers: the actual tensor shape is ground truth
    # and takes precedence over any (possibly incorrect) value_info annotation.
    for value in graph.initializers.values():
        const = ir.convenience.get_const_tensor(value)
        if const is not None:
            true_shape = const.shape
            if value.shape != true_shape:
                logger.debug(
                    "Correcting shape for initializer %s: %s -> %s",
                    value.name,
                    value.shape,
                    true_shape,
                )
                value.shape = true_shape
                modified = True

    # Assign unique names to any anonymous (None) dims on graph inputs
    for value in graph.inputs:
        if ctx.name_anonymous_dims(value):
            modified = True

    # Traverse nodes in topological order
    for node in graph:
        _propagate_types_to_subgraph_inputs(ctx, node)

        # Recursively process any subgraphs (e.g. If/Loop bodies)
        for attr in node.attributes.values():
            if (
                attr is not None
                and isinstance(attr, ir.Attr)
                and attr.type == ir.AttributeType.GRAPH
            ):
                subgraph = attr.as_graph()
                if subgraph is not None:
                    if _process_graph(
                        ctx,
                        subgraph,
                        warn_on_missing=warn_on_missing,
                        model_functions=model_functions,
                        active_functions=active_functions,
                        inference_cache=inference_cache,
                        op_function_cache=op_function_cache,
                    ):
                        modified = True

        domain = node.domain or ""
        op_type = node.op_type
        opset_version = ctx.get_opset_version(domain)

        # Name anonymous dims on ALL nodes (not just those with registered inference),
        # so that function-call inference and subgraph inference also see named dims.
        for inp in node.inputs:
            if inp is not None:
                ctx.name_anonymous_dims(inp)

        # Capture pre-inference output states for change detection.
        # Must be outside the infer_func block so function-call inference
        # also triggers the modified flag.
        old_states: list[tuple[ir.Shape | None, ir.TypeProtocol | None]] = [
            (out.shape, out.type) for out in node.outputs
        ]

        # Look up shape inference function
        infer_func = _registry.registry.get(domain, op_type, version=opset_version)

        if infer_func is not None:
            try:
                # Resolve RefAttr references before calling the inference function.
                # ctx.resolved_attrs is None for top-level graph nodes (zero overhead).
                # The any() check avoids the context manager for body nodes without refs.
                if ctx.resolved_attrs and any(a.is_ref() for a in node.attributes.values()):
                    with _functions._resolve_ref_attrs(node, ctx.resolved_attrs):
                        infer_func(ctx, node)
                else:
                    infer_func(ctx, node)
            except (_context.OpUsageError, _context.ShapeInferenceError):
                raise
            except Exception as e:
                raise _context.ShapeInferenceError(
                    node_name=node.name,
                    op_type=op_type,
                    domain=domain,
                    message=f"Shape inference failed for {domain}::{op_type}",
                ) from e
        elif model_functions is not None and (domain, op_type, node.overload or "") in (
            model_functions
        ):
            _functions.infer_function_call_output_shapes(
                ctx,
                node,
                model_functions,
                process_graph_fn=_process_graph,
                warn_on_missing=warn_on_missing,
                active_functions=active_functions,
                inference_cache=inference_cache,
            )
        elif _functions.infer_via_op_schema_function(
            ctx,
            node,
            process_graph_fn=_process_graph,
            warn_on_missing=warn_on_missing,
            opset_version=opset_version,
            model_functions=model_functions,
            active_functions=active_functions,
            inference_cache=inference_cache,
            op_function_cache=op_function_cache,
        ):
            pass
        elif warn_on_missing:
            _emit_missing_warning(domain, op_type, warned_ops)

        # Check if any output shapes or types changed
        for out, (old_shape, old_type) in zip(node.outputs, old_states):
            if out.shape != old_shape or out.type != old_type:
                modified = True

    return modified
