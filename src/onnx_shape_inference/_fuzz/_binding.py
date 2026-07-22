# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic-dimension binding and concrete-model materialization."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator

import onnx_ir as ir

from onnx_shape_inference._fuzz._types import FuzzCase, SymbolConstraint
from onnx_shape_inference._symbolic_shapes import parse_symbolic_expression

__all__ = [
    "bind_symbols",
    "evaluate_dim",
    "iter_values",
    "materialize_model",
]

_PRIMES = (2, 3, 5, 7)


def iter_values(graph: ir.Graph) -> Iterator[ir.Value]:
    """Yield graph values once, including values held by nested subgraphs."""
    seen: set[int] = set()

    def add(value: ir.Value | None) -> Iterator[ir.Value]:
        if value is not None and id(value) not in seen:
            seen.add(id(value))
            yield value

    for value in graph.inputs:
        yield from add(value)
    for value in graph.outputs:
        yield from add(value)
    for value in graph.initializers.values():
        yield from add(value)
    for node in graph:
        for value in node.inputs:
            yield from add(value)
        for value in node.outputs:
            yield from add(value)
        for attr in node.attributes.values():
            if attr is not None and attr.type == ir.AttributeType.GRAPH:
                subgraph = attr.as_graph()
                if subgraph is not None:
                    yield from iter_values(subgraph)


def _symbols_from_dim(dim: int | ir.SymbolicDim) -> set[str]:
    if isinstance(dim, int):
        return set()
    return {symbol.name for symbol in parse_symbolic_expression(str(dim)).free_symbols}


def bind_symbols(case: FuzzCase, *, include_edge_dims: bool = False) -> dict[str, int]:
    """Create deterministic concrete bindings for every model symbol.

    Small distinct primes prevent accidental equal-dimension successes. Edge
    values are deliberately opt-in because a zero binding invalidates many
    otherwise well-formed operator plans.
    """
    names: set[str] = set(case.symbolic_dims)
    for value in iter_values(case.model.graph):
        if value.shape is not None:
            for dim in value.shape:
                names.update(_symbols_from_dim(dim))

    bindings = dict(case.symbol_bindings)
    rng = random.Random(case.seed)
    remaining = sorted(names - set(bindings))
    for index, name in enumerate(remaining):
        constraint = case.symbol_constraints.get(name, SymbolConstraint())
        if include_edge_dims and constraint.minimum == 0 and rng.randrange(8) == 0:
            bindings[name] = rng.choice((0, 1))
        else:
            candidate = _PRIMES[index % len(_PRIMES)]
            divisor = max(1, constraint.divisible_by)
            if candidate % divisor:
                candidate = max(constraint.minimum, divisor)
            bindings[name] = candidate
        if constraint.maximum is not None and bindings[name] > constraint.maximum:
            bindings[name] = constraint.maximum
    return bindings


def evaluate_dim(dim: int | ir.SymbolicDim, bindings: dict[str, int]) -> int | None:
    """Evaluate a concrete or symbolic dimension under *bindings*."""
    if isinstance(dim, int):
        return dim
    expression = parse_symbolic_expression(str(dim))
    value = expression.subs(
        {
            symbol: bindings[symbol.name]
            for symbol in expression.free_symbols
            if symbol.name in bindings
        }
    )
    if value.is_Integer:
        return int(value)
    return None


def materialize_model(case: FuzzCase, bindings: dict[str, int]) -> ir.Model:
    """Deep-copy *case* and replace every resolvable shape expression with an int."""
    model = copy.deepcopy(case.model)
    for value in iter_values(model.graph):
        if value.shape is None:
            continue
        dims = [evaluate_dim(dim, bindings) for dim in value.shape]
        if all(dim is not None for dim in dims):
            value.shape = ir.Shape(dims)
    return model
