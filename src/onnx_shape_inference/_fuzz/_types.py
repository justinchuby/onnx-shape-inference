# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared generator, oracle, and runtime data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import onnx_ir as ir

__all__ = [
    "DATA_DEPENDENT_OPS",
    "FuzzCase",
    "OracleResult",
    "OracleStatus",
    "SymbolConstraint",
]


# The generator marks individual values from these operators in
# ``data_dependent_values``. The soundness oracle then ignores only the affected
# dimensions while continuing to check rank, dtype, and concrete sibling dims.
DATA_DEPENDENT_OPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("", "Compress"),
        ("", "NonZero"),
        ("", "TopK"),
        ("", "Unique"),
    }
)


class OracleStatus(str, Enum):
    """The outcome of an oracle check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class OracleResult:
    """An oracle outcome with failure details suitable for a reproducer."""

    status: OracleStatus
    reason: str = ""
    value_name: str | None = None
    kind: str | None = None
    expected: object | None = None
    actual: object | None = None

    @classmethod
    def passed(cls) -> OracleResult:
        """Construct a passing result."""
        return cls(OracleStatus.PASS)

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        value_name: str | None = None,
        kind: str | None = None,
        expected: object | None = None,
        actual: object | None = None,
    ) -> OracleResult:
        """Construct a failing result."""
        return cls(OracleStatus.FAIL, reason, value_name, kind, expected, actual)

    @classmethod
    def skipped(cls, reason: str) -> OracleResult:
        """Construct a skipped result."""
        return cls(OracleStatus.SKIP, reason)


@dataclass
class SymbolConstraint:
    """Generator-recorded validity constraints for one symbolic dimension."""

    minimum: int = 1
    maximum: int | None = None
    divisible_by: int = 1


@dataclass
class FuzzCase:
    """A deterministic generated model and the facts needed to verify it.

    The generator owns the declarative symbol metadata while oracles own the
    cache contents. Properties expose the two commonly used cache entries
    without coupling callers to their internal cache keys.
    """

    model: ir.Model
    seed: int
    opset_imports: dict[str, int]
    symbolic_dims: tuple[str, ...] = ()
    symbol_bindings: dict[str, int] = field(default_factory=dict)
    symbol_constraints: dict[str, SymbolConstraint] = field(default_factory=dict)
    data_dependent_values: frozenset[str] = frozenset()
    selected_ops: tuple[tuple[str, str, int], ...] = ()
    inference_cache: dict[str, object] = field(default_factory=dict, repr=False)
    runtime_cache: dict[str, object] = field(default_factory=dict, repr=False)
    pre_simplify_dims: dict[tuple[str, int], str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def our_inference_result(self) -> ir.Model | None:
        """Lazily cached symbolic-inference model."""
        result = self.inference_cache.get("our_inference_result")
        return result if isinstance(result, ir.Model) else None

    @our_inference_result.setter
    def our_inference_result(self, result: ir.Model | None) -> None:
        if result is None:
            self.inference_cache.pop("our_inference_result", None)
        else:
            self.inference_cache["our_inference_result"] = result

    @property
    def onnxruntime_result(self) -> dict[str, Any] | None:
        """Lazily cached intermediate runtime arrays."""
        result = self.runtime_cache.get("onnxruntime_result")
        return result if isinstance(result, dict) else None

    @onnxruntime_result.setter
    def onnxruntime_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self.runtime_cache.pop("onnxruntime_result", None)
        else:
            self.runtime_cache["onnxruntime_result"] = result
