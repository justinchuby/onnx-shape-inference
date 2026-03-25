# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ONNX local function call nodes."""

from __future__ import annotations

__all__ = ["infer_function_call_output_shapes"]

import contextlib
import logging
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING

import onnx_ir as ir

from onnx_shape_inference import _context

logger = logging.getLogger(__name__)

# Per-inference-run cache type alias — only evaluated by type checkers.
# At runtime the | syntax for union types requires Python 3.10+, so we guard
# this definition to avoid a TypeError on Python 3.9.
if TYPE_CHECKING:
    _FuncOutputCache = dict[
        tuple, list[tuple[ir.Shape | None, ir.DataType | None, str | None]]
    ]


def _collect_all_body_values(f: ir.Function) -> list[ir.Value]:
    """Collect every :class:`ir.Value` owned by the function body, including subgraph values.

    Includes formal input values, all outputs produced by top-level body nodes, and
    all values from nested subgraphs (Scan/Loop/If bodies).  This complete set is
    required for save/restore so that the function body is clean for subsequent calls.

    See Also:
        :func:`_collect_top_level_body_values` — the companion function that returns
        only the top-level subset used for the pre-inference clear step.

    Args:
        f: The function whose body values to collect.

    Returns:
        A flat list of all :class:`ir.Value` objects in the function body.
    """
    values: list[ir.Value] = list(f.inputs)

    def _collect_from_graph(graph: ir.Graph) -> None:
        for node in graph:
            for v in node.outputs:
                if v is not None:
                    values.append(v)
            for attr in node.attributes.values():
                if isinstance(attr, ir.Attr) and attr.type == ir.AttributeType.GRAPH:
                    sg = attr.as_graph()
                    if sg is not None:
                        # Subgraph inputs (e.g. Scan loop-carried state variables)
                        # carry shape/dtype that must also be saved and restored.
                        values.extend(sg.inputs)
                        _collect_from_graph(sg)

    _collect_from_graph(f.graph)
    return values


def _collect_top_level_body_values(f: ir.Function) -> list[ir.Value]:
    """Collect only the top-level (non-subgraph) ir.Value objects.

    Returns function formal inputs plus all direct node outputs in the function
    body.  This is the set that gets *cleared* before binding, so that stale
    shapes from a previous inference run cannot corrupt the current one.

    Subgraph values (e.g. Scan body inputs with pre-annotated symbolic dims) are
    excluded — they must retain their annotated shapes for subgraph inference.

    Args:
        f: The function whose top-level values to collect.

    Returns:
        A flat list of top-level :class:`ir.Value` objects.
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
    of ALL values in the function body—including nested subgraph values—then
    clears them all to a blank slate before binding the caller's actual input
    information to the formal parameters.  On exit—whether normal or
    exceptional—all values are restored to their pre-call state unconditionally.

    The clear-before-bind step is critical: without it, internal body values
    left over from a previous inference run would appear as if already inferred,
    causing later callers with different input shapes to see stale (wrong) results.

    Restore order matters: ``v.type`` must be restored *before* ``v.dtype``
    because the dtype setter mutates the current type object in-place when the
    type is not ``None``.  Restoring type first detaches the shared reference to
    the caller's ``TensorType`` before any dtype write, preventing corruption of
    the caller's input dtype.

    Args:
        f: The function definition.
        node: The call-site node supplying the actual inputs.

    Yields:
        Nothing; the context establishes the binding for the body of the ``with``.
    """
    all_values = _collect_all_body_values(f)
    top_level_values = _collect_top_level_body_values(f)

    # Save complete state for EVERY body value (including subgraph values that
    # inference may mutate as a side-effect of running _process_graph).
    saved: list[
        tuple[
            ir.Value,
            ir.Shape | None,
            ir.DataType | None,
            ir.TypeProtocol | None,
            ir.TensorProtocol | None,
            dict,
        ]
    ] = [
        (v, v.shape, v.dtype, v.type, v.const_value, dict(v.metadata_props))
        for v in all_values
    ]

    # Step 1: Clear TOP-LEVEL values only — fresh slate for this inference run.
    # Subgraph values (e.g. Scan body inputs with pre-annotated symbolic shapes)
    # are intentionally left intact: they are needed by subgraph inference and
    # will be fully restored in the finally block.
    for v in top_level_values:
        v.shape = None
        v.type = None  # clear type first so dtype setter has no object to mutate
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
        # Restore everything unconditionally.
        # Restore v.type FIRST to detach any shared reference to the caller's
        # TensorType before the dtype write (which would otherwise mutate it).
        for v, shape, dtype, type_, const_value, metadata in saved:
            v.shape = shape
            v.type = type_  # FIRST: detach from caller's type object
            if dtype is not None:  # skip to avoid creating a spurious TensorType(None)
                v.dtype = dtype
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


def _warn_unresolved_ref_attrs(
    f: ir.Function,
    func_key: tuple[str, str, str],
    resolved_attrs: dict[str, ir.Attr],
) -> None:
    """Warn once if any body node has a RefAttr that could not be resolved.

    A RefAttr is *unresolved* when its ``ref_attr_name`` is absent from
    *resolved_attrs* (neither the call site nor the function defaults provided
    a value for it).  In that case, the inference function for that body node
    will see ``None`` for the attribute and may produce wrong shapes.

    Emits at most one warning per call to avoid log spam.

    Args:
        f: The function to scan.
        func_key: The ``(domain, name, overload)`` triple used in log messages.
        resolved_attrs: The already-resolved attribute map for this call.
    """
    for node in f:
        for attr_name, attr in node.attributes.items():
            if attr.is_ref() and attr.ref_attr_name not in resolved_attrs:
                logger.warning(
                    "Function %s::%s body node %s has unresolvable RefAttr '%s' -> '%s'; "
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
    process_graph_fn: Callable[..., bool],
    warn_on_missing: bool,
    active_functions: frozenset[tuple[str, str, str]] | None = None,
    inference_cache: _FuncOutputCache | None = None,
) -> None:
    """Infer output shapes for a node that calls a local ONNX function.

    Looks up the function definition in *model_functions*, temporarily binds
    the call-site input shapes to the function's formal inputs, runs
    *process_graph_fn* on the function body (using a child context with the
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
        process_graph_fn: The ``_process_graph`` callable from ``_engine``,
            passed in by the caller to avoid a circular import.
        warn_on_missing: Passed through to inner ``_process_graph`` calls.
        active_functions: Functions currently being inferred on the call stack,
            used to detect and break recursive function calls.
        inference_cache: Optional per-inference-run dict for caching body
            inference results.  Pass the same dict across all calls within one
            ``infer_symbolic_shapes`` invocation for maximum reuse.
    """
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
    resolved_attrs: dict[str, ir.Attr] = dict(f.attributes.items())
    resolved_attrs.update(
        {name: attr for name, attr in node.attributes.items() if not attr.is_ref()}
    )

    # Warn if any body node has a RefAttr that resolved_attrs cannot cover
    _warn_unresolved_ref_attrs(f, func_key, resolved_attrs)

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
        process_graph_fn(
            child_ctx,
            f.graph,
            warn_on_missing=warn_on_missing,
            model_functions=model_functions,
            active_functions=active_functions | {func_key},
            inference_cache=inference_cache,
        )

        # Collect results while still inside the scope (before restore)
        output_results: list[tuple[ir.Shape | None, ir.DataType | None, str | None]] = []
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
