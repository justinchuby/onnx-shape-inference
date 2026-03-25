# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ONNX local function call nodes."""

from __future__ import annotations

__all__ = ["infer_function_call_output_shapes"]

import contextlib
import logging
from collections.abc import Generator, Mapping

import onnx_ir as ir

from onnx_shape_inference import _context

logger = logging.getLogger(__name__)


def _collect_all_body_values(f: ir.Function) -> list[ir.Value]:
    """Collect every ir.Value owned by the function body.

    Includes formal input values and all outputs produced by body nodes.
    This complete set is needed for the save/restore cycle: every value
    must be cleared before a fresh inference run so that stale shapes from
    a previous call cannot corrupt the current one.

    Args:
        f: The function whose body values to collect.

    Returns:
        A flat list of all :class:`ir.Value` objects in the function body.
    """
    values: list[ir.Value] = list(f.inputs)
    for node in f:
        for v in node.outputs:
            if v is not None:
                values.append(v)
    return values


def _make_input_signature(
    f: ir.Function,
    node: ir.Node,
) -> tuple:
    """Build a hashable cache key from the call-site input shapes and dtypes.

    Used to detect when the same function is called with identical inputs so
    the cached output shapes can be reused.  The key encodes, for each formal
    input of the function, the dimension tuple of the caller's corresponding
    input (or ``None`` when shape is unknown) and its dtype.

    Args:
        f: The function being called.
        node: The call-site node in the parent graph.

    Returns:
        A hashable tuple suitable for use as a dict key.
    """
    parts: list = []
    for _, n_inp in zip(f.inputs, node.inputs):
        if n_inp is None:
            parts.append(None)
        else:
            shape_key = tuple(n_inp.shape.dims) if n_inp.shape is not None else None
            parts.append((shape_key, n_inp.dtype))
    return tuple(parts)


@contextlib.contextmanager
def _function_binding_scope(
    f: ir.Function,
    node: ir.Node,
) -> Generator[None, None, None]:
    """Temporarily bind call-site input shapes to the function's formal inputs.

    Saves the complete state (shape, dtype, type, const_value, metadata_props)
    of ALL values in the function body, then clears them all to a blank slate
    before binding the caller's actual input information to the formal
    parameters.  On exit—whether normal or exceptional—all values are restored
    to their pre-call state unconditionally.

    The clear-before-bind step is critical: without it, internal body values
    left over from a previous inference run would appear as if already inferred,
    causing later callers with different input shapes to see stale (wrong)
    results.

    Args:
        f: The function definition.
        node: The call-site node supplying the actual inputs.

    Yields:
        Nothing; the context establishes the binding for the body of the ``with``.
    """
    all_values = _collect_all_body_values(f)

    # Save complete state for every body value
    saved: list[
        tuple[ir.Value, ir.Shape | None, ir.DataType | None, object, object, dict]
    ] = [
        (v, v.shape, v.dtype, v.type, v.const_value, dict(v.metadata_props))
        for v in all_values
    ]

    # Step 1: Clear everything — fresh slate for this inference run
    for v, _, _, _, _, _ in saved:
        v.shape = None
        v.dtype = None
        v.type = None
        v.const_value = None
        v.metadata_props.clear()

    # Step 2: Bind call-site actual inputs → function formal inputs
    for f_inp, n_inp in zip(f.inputs, node.inputs):
        if n_inp is not None:
            f_inp.shape = n_inp.shape
            f_inp.dtype = n_inp.dtype
            if n_inp.type is not None:
                f_inp.type = n_inp.type
            if n_inp.const_value is not None:
                f_inp.const_value = n_inp.const_value
            sym_data = n_inp.metadata_props.get(_context.SYM_DATA_KEY)
            if sym_data is not None:
                f_inp.metadata_props[_context.SYM_DATA_KEY] = sym_data

    try:
        yield
    finally:
        # Restore everything unconditionally
        for v, shape, dtype, type_, const_value, metadata in saved:
            v.shape = shape
            v.dtype = dtype
            v.type = type_
            v.const_value = const_value
            v.metadata_props.clear()
            v.metadata_props.update(metadata)


@contextlib.contextmanager
def _resolve_ref_attrs(
    node: ir.Node,
    resolved_attrs: dict[str, ir.Attr],
) -> Generator[None, None, None]:
    """Temporarily substitute RefAttr values in a body node's attributes.

    Scans *node*'s attributes for reference attributes (``attr.is_ref() is
    True``).  For each one, looks up the resolved value by
    ``attr.ref_attr_name`` in *resolved_attrs* and replaces the RefAttr
    in-place with a concrete :class:`ir.Attr` carrying the same local name
    but the resolved value.  Restores the original RefAttrs on exit.

    This allows existing inference functions to call
    ``node.attributes.get("axis").as_int()`` without any special-casing of
    reference attributes.

    Args:
        node: The body node whose attributes to patch.
        resolved_attrs: Map from function-level attribute parameter name to
            the actual :class:`ir.Attr` value (call-site or function default).

    Yields:
        Nothing.
    """
    substituted: dict[str, ir.Attr] = {}

    for attr_name, attr in list(node.attributes.items()):
        if attr.is_ref():
            actual = resolved_attrs.get(attr.ref_attr_name)
            if actual is not None:
                resolved = ir.Attr(attr_name, actual.type, actual.value)
                substituted[attr_name] = attr
                node.attributes[attr_name] = resolved
            else:
                logger.debug(
                    "RefAttr '%s' in %s::%s references '%s' but no resolved value found",
                    attr_name,
                    node.domain or "",
                    node.op_type,
                    attr.ref_attr_name,
                )

    try:
        yield
    finally:
        for attr_name, original in substituted.items():
            node.attributes[attr_name] = original


def _warn_ref_attrs_present(f: ir.Function, func_key: tuple[str, str, str]) -> None:
    """Warn once if any body node has an unresolved RefAttr.

    When a body node still carries a RefAttr after resolution (i.e.
    ``resolved_attrs`` didn't cover it), inference for that attribute will
    see ``None`` and may produce wrong shapes.  This warning surfaces that
    situation without crashing.

    Emits at most one warning per function call to avoid log spam.

    Args:
        f: The function to scan.
        func_key: The ``(domain, name, overload)`` triple used in log messages.
    """
    for node in f:
        for attr_name, attr in node.attributes.items():
            if attr.is_ref():
                logger.warning(
                    "Function %s::%s body node %s has RefAttr '%s' -> '%s'; "
                    "shapes for this function may be incorrect.",
                    func_key[0],
                    func_key[1],
                    node.op_type,
                    attr_name,
                    attr.ref_attr_name,
                )
                return  # Warn once per call


def infer_function_call_output_shapes(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    model_functions: Mapping[tuple[str, str, str], ir.Function],
    *,
    warn_on_missing: bool,
    active_functions: frozenset[tuple[str, str, str]] | None = None,
    inference_cache: dict | None = None,
) -> None:
    """Infer output shapes for a node that calls a local ONNX function.

    Looks up the function definition in *model_functions*, temporarily binds
    the call-site input shapes to the function's formal inputs, runs
    ``_process_graph`` on the function body (using a child context with the
    function's own opset imports), then copies the inferred output shapes and
    dtypes back to the calling node's outputs.  All function body values are
    restored to their original state after each call, ensuring calls are
    independent of one another regardless of call order or input shapes.

    Results are cached per ``(id(function), input_signature)`` within a single
    inference run so that repeated calls with identical input shapes incur only
    a single body traversal.

    Args:
        ctx: The current shape inference context (parent graph).
        node: The node in the parent graph that calls the function.
        model_functions: All local functions registered in the model.
        warn_on_missing: Passed through to inner ``_process_graph`` calls.
        active_functions: Functions currently being inferred on the call stack,
            used to detect and break recursive function calls.
        inference_cache: Optional per-inference-run dict for caching body
            inference results.  Pass the same dict across all calls within one
            ``infer_symbolic_shapes`` invocation for maximum reuse.
    """
    # Lazy import to avoid circular dependency: _engine imports _functions,
    # so _functions cannot import _engine at module level.
    from onnx_shape_inference import _engine

    func_key = (node.domain or "", node.op_type, node.overload or "")
    f = model_functions.get(func_key)
    if f is None:
        return

    if active_functions is None:
        active_functions = frozenset()

    if func_key in active_functions:
        logger.warning(
            "Recursive function call detected for %s::%s; skipping to avoid infinite loop",
            node.domain,
            node.op_type,
        )
        return

    # Cache lookup: same function + same input shapes → reuse results
    cache_key: tuple | None = None
    if inference_cache is not None:
        cache_key = (id(f), _make_input_signature(f, node))
        if cache_key in inference_cache:
            cached_outputs = inference_cache[cache_key]
            for n_out, (c_shape, c_dtype, c_sym_data) in zip(node.outputs, cached_outputs):
                if n_out is None:
                    continue
                if c_shape is not None:
                    ctx.set_shape(n_out, c_shape)
                if c_dtype is not None:
                    ctx.set_dtype(n_out, c_dtype)
                if c_sym_data is not None:
                    n_out.metadata_props[_context.SYM_DATA_KEY] = c_sym_data
            return

    # Build resolved_attrs from function defaults overridden by call-site values.
    # This gives body nodes concrete attribute values to substitute for any RefAttrs.
    resolved_attrs: dict[str, ir.Attr] = {}
    for name, attr in f.attributes.items():
        resolved_attrs[name] = attr
    for name, attr in node.attributes.items():
        if not attr.is_ref():
            resolved_attrs[name] = attr

    # Warn if any body node still has an unresolved RefAttr
    _warn_ref_attrs_present(f, func_key)

    # Child context: function's own opset versions so body dispatch is correct
    child_ctx = _context.ShapeInferenceContext(
        opset_imports=f.opset_imports or ctx.opset_imports,
        policy=ctx.policy,
        resolved_attrs=resolved_attrs,
    )
    # Share the dim counter so symbolic dim names are globally unique across
    # parent and child contexts
    child_ctx._dim_counter = ctx._dim_counter

    with _function_binding_scope(f, node):
        _engine._process_graph(
            child_ctx,
            f.graph,
            warn_on_missing=warn_on_missing,
            model_functions=model_functions,
            active_functions=active_functions | {func_key},
            inference_cache=inference_cache,
        )

        # Collect results while still inside the scope (before restore)
        output_results: list[tuple] = []
        for f_out, n_out in zip(f.outputs, node.outputs):
            sym_data = f_out.metadata_props.get(_context.SYM_DATA_KEY)
            output_results.append((f_out.shape, f_out.dtype, sym_data))

            if n_out is None:
                continue
            if f_out.shape is not None:
                ctx.set_shape(n_out, f_out.shape)
            if f_out.dtype is not None:
                ctx.set_dtype(n_out, f_out.dtype)
            if sym_data is not None:
                n_out.metadata_props[_context.SYM_DATA_KEY] = sym_data

        # Store in cache for subsequent identical calls
        if inference_cache is not None and cache_key is not None:
            inference_cache[cache_key] = output_results

    # Propagate dim counter back to the parent context
    ctx._dim_counter = child_ctx._dim_counter
