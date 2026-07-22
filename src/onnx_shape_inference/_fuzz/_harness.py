# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic driver shared by fast tests and nightly fuzzing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from onnx_shape_inference._fuzz._oracles import Oracle
from onnx_shape_inference._fuzz._types import FuzzCase, OracleResult, OracleStatus

__all__ = ["FuzzHarness", "FuzzSummary", "write_coverage_report"]


@dataclass
class FuzzSummary:
    """Aggregate statuses and per-op coverage for a harness invocation."""

    results: Counter[OracleStatus] = field(default_factory=Counter)
    op_coverage: Counter[tuple[str, str]] = field(default_factory=Counter)
    skips: Counter[str] = field(default_factory=Counter)


class FuzzHarness:
    """Run a seeded generator through every applicable pluggable oracle."""

    def __init__(self, generate: Callable[[int], FuzzCase], oracles: Sequence[Oracle]) -> None:
        self.generate = generate
        self.oracles = tuple(oracles)

    def run(self, seeds: Iterable[int]) -> FuzzSummary:
        """Run *seeds*, raising an actionable assertion for the first failure."""
        summary = FuzzSummary()
        for seed in seeds:
            case = self.generate(seed)
            for node in case.model.graph:
                summary.op_coverage[(node.domain or "", node.op_type)] += 1
            for oracle in self.oracles:
                if not oracle.applicable(case):
                    result = OracleResult.skipped("oracle not applicable")
                else:
                    result = oracle.check(case)
                summary.results[result.status] += 1
                if result.status is OracleStatus.SKIP:
                    summary.skips[result.reason] += 1
                if result.status is OracleStatus.FAIL:
                    raise AssertionError(self._failure_message(case, oracle.name, result))
        return summary

    @staticmethod
    def _failure_message(case: FuzzCase, oracle: str, result: OracleResult) -> str:
        command = f"FUZZ_SEED={case.seed} python3 -m pytest tests/shape_inference_fuzz_test.py"
        return (
            f"fuzz failure: oracle={oracle} seed={case.seed} value={result.value_name!r} "
            f"kind={result.kind!r}: {result.reason}; expected={result.expected!r}, "
            f"actual={result.actual!r}\nreproduce: {command}"
        )


def write_coverage_report(summary: FuzzSummary, path: str | Path) -> Path:
    """Write the deterministic per-op coverage data consumed by nightly artifacts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    coverage = {
        f"{domain or 'ai.onnx'}::{op_type}": count
        for (domain, op_type), count in sorted(summary.op_coverage.items())
    }
    destination.write_text(
        json.dumps(
            {
                "op_coverage": coverage,
                "outcomes": {status.value: count for status, count in summary.results.items()},
                "skips": dict(sorted(summary.skips.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return destination
