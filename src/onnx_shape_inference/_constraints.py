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
    "propagate_symbolic_constraints",
]

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


def _canonical_name(names: list[str]) -> str:
    """Pick the canonical name for an equivalence class.

    Prefer a user-declared (non-anonymous) name; break ties deterministically by
    (length, lexicographic order).  Falls back to the same ordering over the
    anonymous names when the class has no declared name.
    """
    preferred = [n for n in names if not _context._is_anonymous_symbol_name(n)]
    pool = preferred or names
    return min(pool, key=lambda n: (len(n), n))


def _build_replacements(
    equalities: list[tuple[str, str]],
) -> tuple[dict[sympy.Symbol, sympy.Symbol], list[tuple[sympy.Expr, sympy.Symbol]]]:
    """Build the leaf-symbol map and compound-expression replacements."""
    union = _UnionFind()
    compound: list[tuple[sympy.Expr, sympy.Symbol]] = []

    for a_str, b_str in equalities:
        a_expr = _symbolic_shapes.parse_symbolic_expression(a_str)
        b_expr = _symbolic_shapes.parse_symbolic_expression(b_str)
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
                and not _context._is_anonymous_symbol_name(target.name)
                and not isinstance(source, sympy.Symbol)
            ):
                compound.append((source, target))
                break

    symbol_map: dict[sympy.Symbol, sympy.Symbol] = {}
    for members in union.members().values():
        canonical = _canonical_name(members)
        target = _symbolic_shapes.parse_symbolic_expression(canonical)
        for name in members:
            if name == canonical:
                continue
            # Only rename engine-anonymous symbols; never rewrite a declared name.
            if _context._is_anonymous_symbol_name(name):
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
    if not isinstance(dim, ir.SymbolicDim) or dim.value is None:
        return None
    expr = _symbolic_shapes.parse_symbolic_expression(dim.value)
    new_expr = expr.subs(symbol_map)
    for source, target in compound:
        new_expr = new_expr.subs(source, target)
    if new_expr == expr:
        return None
    # Safe canonicalization only (no divisibility assumptions).
    new_expr = _symbolic_shapes.simplify_expression(new_expr)
    if new_expr.is_Integer:
        return int(new_expr)
    return ir.SymbolicDim(str(new_expr))


def _collect_values(graph: ir.Graph) -> list[ir.Value]:
    """Collect every value whose shape may need rewriting (deduplicated)."""
    seen: dict[int, ir.Value] = {}

    def _add(value: ir.Value | None) -> None:
        if value is not None and id(value) not in seen:
            seen[id(value)] = value

    for value in graph.inputs:
        _add(value)
    for value in graph.outputs:
        _add(value)
    for node in graph.all_nodes():
        for value in node.inputs:
            _add(value)
        for value in node.outputs:
            _add(value)
    return list(seen.values())


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

    symbol_map, compound = _build_replacements(equalities)
    if not symbol_map and not compound:
        return False

    values = _collect_values(graph)
    modified = False

    for _ in range(_MAX_REWRITE_ITERATIONS):
        changed_this_pass = False
        for value in values:
            shape = value.shape
            if shape is None:
                continue
            new_dims: list[int | ir.SymbolicDim] = []
            dim_changed = False
            for dim in shape.dims:
                rewritten = _rewrite_dim(dim, symbol_map, compound)
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
