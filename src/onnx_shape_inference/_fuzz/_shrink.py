# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Failure-signature-preserving delta shrinking for fuzz cases."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from onnx_shape_inference._fuzz._binding import materialize_model
from onnx_shape_inference._fuzz._oracles import Oracle
from onnx_shape_inference._fuzz._types import FuzzCase, OracleResult, OracleStatus

__all__ = ["DeltaShrinker", "FailureSignature", "failure_signature"]


@dataclass(frozen=True)
class FailureSignature:
    """The stable part of an oracle failure that a shrink must preserve."""

    oracle: str
    value_name: str | None
    kind: str | None


def failure_signature(oracle: Oracle, result: OracleResult) -> FailureSignature:
    """Create the signature used to reject reductions that expose another bug."""
    return FailureSignature(oracle.name, result.value_name, result.kind)


class DeltaShrinker:
    """Greedily retain only reductions with the original oracle signature.

    Node deletion is intentionally delegated to generator-specific reconstruction:
    ONNX IR nodes retain producer/consumer ownership, so rebuilding a valid DAG
    is safer than mutating its internal graph storage. The common and reliable
    reductions below (symbol concretization and dim minimization) work for every
    generated model today; planners can register a ``rebuild_with_nodes`` hook
    in ``case.metadata`` for backward-slice deletion.
    """

    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle

    def shrink(self, case: FuzzCase) -> FuzzCase:
        """Return the smallest signature-preserving case found greedily."""
        initial = self.oracle.check(case)
        if initial.status is not OracleStatus.FAIL:
            return case
        signature = failure_signature(self.oracle, initial)
        current = copy.deepcopy(case)
        current = self._try_concretizing_symbols(current, signature)
        current = self._try_smaller_bindings(current, signature)
        return current

    def _preserves(self, candidate: FuzzCase, signature: FailureSignature) -> bool:
        result = self.oracle.check(candidate)
        return (
            result.status is OracleStatus.FAIL
            and failure_signature(self.oracle, result) == signature
        )

    def _try_concretizing_symbols(
        self, case: FuzzCase, signature: FailureSignature
    ) -> FuzzCase:
        candidate = copy.deepcopy(case)
        bindings = dict(candidate.symbol_bindings)
        if not bindings:
            return case
        candidate.model = materialize_model(candidate, bindings)
        candidate.symbol_bindings = bindings
        candidate.metadata["shrink_bindings"] = bindings
        return candidate if self._preserves(candidate, signature) else case

    def _try_smaller_bindings(self, case: FuzzCase, signature: FailureSignature) -> FuzzCase:
        current = case
        for name, value in sorted(case.symbol_bindings.items()):
            if value <= 1:
                continue
            candidate = copy.deepcopy(current)
            candidate.symbol_bindings[name] = 1
            if self._preserves(candidate, signature):
                current = candidate
        return current
