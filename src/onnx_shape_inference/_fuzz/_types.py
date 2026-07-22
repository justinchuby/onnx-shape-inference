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
]


# The generator marks individual values from these operators in
# ``data_dependent_ports``. The soundness oracle then ignores only the affected
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
class FuzzCase:
    """A deterministic generated model and the facts needed to verify it.

    ``symbolic_dims`` maps names to their optional generator-selected bindings.
    ``constraints`` is intentionally data-only so generator planners can add
    equality/bounds without coupling the oracle package to planner classes.
    The result fields are caches populated lazily by individual oracles.
    """

    model: ir.Model
    seed: int
    opsets: dict[str, int]
    symbolic_dims: dict[str, int | None] = field(default_factory=dict)
    constraints: tuple[object, ...] = ()
    data_dependent_ports: frozenset[object] = frozenset()
    our_inference_result: ir.Model | None = None
    onnxruntime_result: dict[str, Any] | None = None
    pre_simplify_dims: dict[tuple[str, int], str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
