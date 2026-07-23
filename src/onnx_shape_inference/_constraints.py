# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Anchor/constraint propagation for symbolic dimensions.

Per-operator inference names data-dependent dimensions locally with
engine-anonymous symbols (``_d0``, ``_d1`` …).  The model author, however,
declares user-visible names on graph outputs and ``value_info`` (for example
``Y: [N, TopK_k]`` or ``present_key: [batch, 4, total_seq, 4]``).

When the inferred shape of an anchored value differs from its declared shape,
the context records an *equality constraint* relating the two expressions (see
:meth:`ShapeInferenceContext.add_symbolic_equality`).  This module turns those
constraints into a **renaming** of the engine-anonymous symbols so they adopt
the declared names, then rewrites every value's shape in the graph — including
compound occurrences such as ``2*_d0 -> 2*dnz`` and ``past_seq + seq ->
total_seq``.

Only engine-anonymous ``_dN`` symbols are ever renamed; user-declared names are
treated as authoritative and preserved.
"""

from __future__ import annotations

__all__ = [
    "collect_symbol_names",
    "propagate_symbolic_constraints",
    "record_reshape_numel_equalities",
]

from collections.abc import Callable

import onnx_ir as ir
import sympy

from onnx_shape_inference import _context, _symbolic_shapes

_MAX_REWRITE_ITERATIONS = 8


class _UnionFind:
    """Minimal union-find over symbol names."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, name: str) -> str:
        self._parent.setdefault(name, name)
        root = name
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[name] != root:
            self._parent[name], name = root, self._parent[name]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def members(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in list(self._parent):
            groups.setdefault(self.find(name), []).append(name)
        return groups


def _leaf_equality(a: sympy.Expr, b: sympy.Expr) -> tuple[str, str] | None:
    """Return a leaf symbol equality implied by ``a == b``, if any.

    ``a == b`` implies ``x == y`` when ``a - b`` reduces to ``c*x - c*y`` for a
    single non-zero coefficient ``c`` (two symbol terms of equal magnitude and
    opposite sign, no constant term).  Handles the bare ``x == y`` case too.
    """
    diff = sympy.expand(a - b)
    if diff == 0:
        return None
    coeffs = diff.as_coefficients_dict()
    symbol_terms: list[tuple[sympy.Symbol, sympy.Rational]] = []
    for term, coeff in coeffs.items():
        if term == 1:  # constant term present -> not a clean leaf equality
            return None
        if isinstance(term, sympy.Symbol):
            symbol_terms.append((term, coeff))
        else:
            return None
    if len(symbol_terms) != 2:
        return None
    (sym1, c1), (sym2, c2) = symbol_terms
    if c1 + c2 != 0:
        return None
    return (sym1.name, sym2.name)


def _reduce_common_factor(a: sympy.Expr, b: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    """Cancel the positive common factor shared by both sides of ``a == b``.

    Reshape records a *numel equality* ``product(input_dims) == product(output
    dims)`` — always true because reshape preserves element count.  For a flatten
    such as ``[a, b, c] -> [a, b, 2, c//2]`` this is
    ``a*b*c == a*b*2*floor(c/2)``.  Dividing both sides by the shared factor
    ``a*b`` (their polynomial GCD, provably positive since every dim is
    ``positive``) yields the *provenance* fact ``c == 2*floor(c/2)`` — i.e. the
    reshape asserts ``c`` is even.  This is what lets ``2*floor(c/2)`` be rewritten
    back to ``c`` **only** where a reshape introduced the divisibility, and never
    for a floor merely inherited from an operand dim (``2*floor(H/2) ==
    2*floor(H/2)`` cancels to a tautology and constrains nothing).

    The reduction is truth-preserving: ``a == b`` with ``g = gcd(a, b) > 0``
    implies ``a/g == b/g``.  ``floor(...)`` terms are treated as opaque
    generators, so only whole monomial factors are cancelled.
    """
    if a == b:
        return a, b
    try:
        g = sympy.gcd(a, b)
    except sympy.PolynomialError:
        return a, b
    if g is None or g == 1 or g == 0:
        return a, b
    reduced_a = sympy.simplify(a / g)
    reduced_b = sympy.simplify(b / g)
    if reduced_a.has(sympy.zoo, sympy.nan, sympy.oo) or reduced_b.has(
        sympy.zoo, sympy.nan, sympy.oo
    ):
        return a, b
    return reduced_a, reduced_b


def _canonical_name(names: list[str], is_generated: Callable[[str], bool]) -> str:
    """Pick the canonical name for an equivalence class.

    Prefer a user-declared (non-generated) name; break ties deterministically by
    (length, lexicographic order).  Falls back to the same ordering over the
    generated names when the class has no declared name.
    """
    preferred = [n for n in names if not is_generated(n)]
    pool = preferred or names
    return min(pool, key=lambda n: (len(n), n))


def _build_replacements(
    equalities: list[tuple[str, str]],
    is_generated: Callable[[str], bool],
) -> tuple[dict[sympy.Symbol, sympy.Symbol], list[tuple[sympy.Expr, sympy.Symbol]]]:
    """Build the leaf-symbol map and compound-expression replacements."""
    union = _UnionFind()
    compound: list[tuple[sympy.Expr, sympy.Symbol]] = []

    for a_str, b_str in equalities:
        a_expr = _symbolic_shapes.parse_symbolic_expression(a_str)
        b_expr = _symbolic_shapes.parse_symbolic_expression(b_str)
        # Cancel any shared positive factor first so numel equalities such as
        # ``a*b*c == a*b*2*floor(c/2)`` reduce to the useful ``c == 2*floor(c/2)``.
        a_expr, b_expr = _reduce_common_factor(a_expr, b_expr)
        if a_expr == b_expr:
            continue
        leaf = _leaf_equality(a_expr, b_expr)
        if leaf is not None:
            union.union(*leaf)
            continue
        # No clean leaf equality: if exactly one side is a single declared
        # symbol, replace the (compound) other side by that symbol.
        pairs = [(a_expr, b_expr), (b_expr, a_expr)]
        for source, target in pairs:
            if (
                isinstance(target, sympy.Symbol)
                and not is_generated(target.name)
                and not isinstance(source, sympy.Symbol)
            ):
                compound.append((source, target))
                break

    symbol_map: dict[sympy.Symbol, sympy.Symbol] = {}
    for members in union.members().values():
        canonical = _canonical_name(members, is_generated)
        target = _symbolic_shapes.parse_symbolic_expression(canonical)
        for name in members:
            if name == canonical:
                continue
            # Only rename engine-generated symbols; never rewrite a declared name.
            if is_generated(name):
                symbol_map[sympy.Symbol(name, integer=True, positive=True)] = target

    # Rename internal symbols inside compound sources so they match the graph.
    resolved_compound = [(source.subs(symbol_map), target) for source, target in compound]
    return symbol_map, resolved_compound


def _rewrite_dim(
    dim: int | ir.SymbolicDim,
    symbol_map: dict[sympy.Symbol, sympy.Symbol],
    compound: list[tuple[sympy.Expr, sympy.Symbol]],
) -> int | ir.SymbolicDim | None:
    """Return the rewritten dim, or ``None`` when unchanged."""
    expr = _symbolic_shapes.dim_to_expr(dim)
    if expr is None or not expr.free_symbols:
        return None
    new_expr = expr.subs(symbol_map)
    for source, target in compound:
        new_expr = new_expr.subs(source, target)
    if new_expr == expr:
        return None
    # Safe canonicalization only (no divisibility assumptions).
    new_expr = _symbolic_shapes.simplify_expression(new_expr)
    return _symbolic_shapes.expr_to_dim(new_expr)


def _iter_subgraphs(node: ir.Node) -> list[ir.Graph]:
    """Return the subgraphs referenced by a node's GRAPH/GRAPHS attributes."""
    subgraphs: list[ir.Graph] = []
    for attr in node.attributes.values():
        if not isinstance(attr, ir.Attr):
            continue
        if attr.type == ir.AttributeType.GRAPH:
            sub = attr.as_graph()
            if sub is not None:
                subgraphs.append(sub)
        elif attr.type == ir.AttributeType.GRAPHS:
            for sub in attr.as_graphs() or ():
                if sub is not None:
                    subgraphs.append(sub)
    return subgraphs


def _collect_values(graph: ir.Graph) -> list[ir.Value]:
    """Collect every value whose shape may need rewriting (deduplicated).

    Walks the root graph and every nested subgraph (``If``/``Loop``/``Scan``
    bodies), gathering graph inputs, outputs, initializers, and each node's
    inputs and outputs.  Subgraph inputs/outputs are included even when a body
    has no nodes, so declared names propagate everywhere.
    """
    seen: dict[int, ir.Value] = {}

    def _add(value: ir.Value | None) -> None:
        if value is not None and id(value) not in seen:
            seen[id(value)] = value

    def _visit(g: ir.Graph) -> None:
        for value in g.inputs:
            _add(value)
        for value in g.outputs:
            _add(value)
        for value in g.initializers.values():
            _add(value)
        for node in g:
            for value in node.inputs:
                _add(value)
            for value in node.outputs:
                _add(value)
            for subgraph in _iter_subgraphs(node):
                _visit(subgraph)

    _visit(graph)
    return list(seen.values())


def _iter_nodes(graph: ir.Graph) -> list[ir.Node]:
    """Return every node in *graph* and nested subgraphs (depth-first)."""
    nodes: list[ir.Node] = []

    def _visit(g: ir.Graph) -> None:
        for node in g:
            nodes.append(node)
            for subgraph in _iter_subgraphs(node):
                _visit(subgraph)

    _visit(graph)
    return nodes


def _shape_numel_expr(shape: ir.Shape | None) -> sympy.Expr | None:
    """Return the product of *shape*'s dims as a SymPy expression, or ``None``.

    Returns ``None`` when the shape is missing or any dim is unknown
    (``SymbolicDim(None)``), since the element count is then not expressible.
    """
    if shape is None:
        return None
    product: sympy.Expr = sympy.Integer(1)
    for dim in shape.dims:
        if isinstance(dim, int):
            product = product * dim
            continue
        expr = _symbolic_shapes.dim_to_expr(dim)
        if expr is None:
            return None
        product = product * expr
    return sympy.expand(product)


def _is_divisibility_source(source: sympy.Expr) -> bool:
    """Return whether a compound rewrite source is a divisibility rewrite.

    A source containing ``floor``/``ceiling`` (e.g. ``2*floor(c/2)``) is only
    sound to collapse where a reshape's numel equality supplied the divisibility
    provenance.  Compound sources without rounding (e.g. ``past_seq + seq``)
    carry no such caveat and stay globally applicable.
    """
    return source.has(sympy.floor, sympy.ceiling)


def _reshape_numel_expressions(graph: ir.Graph) -> list[sympy.Expr]:
    """Collect the input/output numel expressions of every Reshape in *graph*.

    These identify the values eligible for divisibility (floor/ceiling) compound
    rewrites: only a value whose element count matches a reshape's numel may have
    inherited that reshape's divisibility, so only such values are collapsed (see
    :func:`propagate_symbolic_constraints`).
    """
    numels: list[sympy.Expr] = []
    for node in _iter_nodes(graph):
        if node.domain != "" or node.op_type != "Reshape":
            continue
        if not node.inputs or not node.outputs:
            continue
        data, out = node.inputs[0], node.outputs[0]
        for value in (data, out):
            if value is None:
                continue
            numel = _shape_numel_expr(value.shape)
            if numel is not None:
                numels.append(numel)
    return numels


def record_reshape_numel_equalities(
    ctx: _context.ShapeInferenceContext,
    graph: ir.Graph,
) -> None:
    """Record ``product(input) == product(output)`` for every Reshape node.

    Reshape preserves element count, so this equality is always true for a valid
    model.  When the output carries a factor absent from the input (e.g. a split
    ``c -> [2, c//2]``), :func:`_reduce_common_factor` distills the divisibility
    provenance (``c == 2*floor(c/2)``) that lets the propagation pass rewrite the
    inherited floor back to the original symbol — soundly, and only where a
    reshape actually introduced the divisibility.
    """
    for node in _iter_nodes(graph):
        if node.domain != "" or node.op_type != "Reshape":
            continue
        if not node.inputs or not node.outputs:
            continue
        data, out = node.inputs[0], node.outputs[0]
        if data is None or out is None:
            continue
        in_numel = _shape_numel_expr(data.shape)
        out_numel = _shape_numel_expr(out.shape)
        if in_numel is None or out_numel is None or in_numel == out_numel:
            continue
        ctx.add_symbolic_equality(str(in_numel), str(out_numel))


def collect_symbol_names(graph: ir.Graph) -> set[str]:
    """Collect every symbolic dimension name already present in *graph*.

    Walks the root graph and nested subgraphs (via :func:`_collect_values`) and
    returns the free-symbol names appearing in any declared shape.  The engine
    reserves these before inference so freshly minted anonymous names can never
    collide with an author-declared symbol (see
    :meth:`ShapeInferenceContext.reserve_symbol_names`).
    """
    names: set[str] = set()
    for value in _collect_values(graph):
        shape = value.shape
        if shape is None:
            continue
        for dim in shape.dims:
            expr = _symbolic_shapes.dim_to_expr(dim)
            if expr is None:
                continue
            for symbol in expr.free_symbols:
                names.add(symbol.name)
    return names


def propagate_symbolic_constraints(
    ctx: _context.ShapeInferenceContext,
    graph: ir.Graph,
) -> bool:
    """Rename engine-anonymous symbols to declared names across *graph*.

    Uses the equality constraints recorded on *ctx* (from anchor merges or op
    inference).  Returns ``True`` if any shape was modified.
    """
    equalities = list(ctx.symbolic_equalities)
    if not equalities:
        return False

    symbol_map, compound = _build_replacements(equalities, ctx.is_generated_symbol)
    if not symbol_map and not compound:
        return False

    # Divisibility (floor/ceiling) compound rewrites are only sound where a
    # reshape's numel equality established the divisibility.  Restrict them to
    # values whose element count matches a reshape numel, so an unrelated genuine
    # ``2*floor(c/2)`` elsewhere is never collapsed.  Non-rounding compound
    # rewrites (e.g. ``past_seq + seq -> total_seq``) remain globally applicable.
    global_compound = [(s, t) for s, t in compound if not _is_divisibility_source(s)]
    divisibility_compound = [(s, t) for s, t in compound if _is_divisibility_source(s)]
    reshape_numels = _reshape_numel_expressions(graph) if divisibility_compound else []

    def _applicable_compound(
        value: ir.Value,
    ) -> list[tuple[sympy.Expr, sympy.Symbol]]:
        if not divisibility_compound:
            return global_compound
        numel = _shape_numel_expr(value.shape)
        if numel is not None and any(numel == r for r in reshape_numels):
            return global_compound + divisibility_compound
        return global_compound

    values = _collect_values(graph)
    modified = False

    for _ in range(_MAX_REWRITE_ITERATIONS):
        changed_this_pass = False
        for value in values:
            shape = value.shape
            if shape is None:
                continue
            applicable_compound = _applicable_compound(value)
            new_dims: list[int | ir.SymbolicDim] = []
            dim_changed = False
            for dim in shape.dims:
                rewritten = _rewrite_dim(dim, symbol_map, applicable_compound)
                if rewritten is None:
                    new_dims.append(dim)
                else:
                    new_dims.append(rewritten)
                    dim_changed = True
            if dim_changed:
                value.shape = ir.Shape(new_dims)
                changed_this_pass = True
                modified = True
        if not changed_this_pass:
            break

    return modified
