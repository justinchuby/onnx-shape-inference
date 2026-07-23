# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic driver shared by fast tests and nightly fuzzing."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tests.fuzz._oracles import Oracle
from tests.fuzz._types import FuzzCase, OracleResult, Status

__all__ = ["FuzzHarness", "FuzzSummary", "write_coverage_report"]


@dataclass
class FuzzSummary:
    """Aggregate statuses and per-op coverage for a harness invocation."""

    results: Counter[Status] = field(default_factory=Counter)
    op_coverage: Counter[tuple[str, str]] = field(default_factory=Counter)
    skips: Counter[str] = field(default_factory=Counter)


class FuzzHarness:
    """Run a seeded generator through every applicable pluggable oracle."""

    def __init__(self, generate: Callable[[int], FuzzCase], oracles: Sequence[Oracle]) -> None:
        self.generate = generate
        order = {"crash": 0, "differential": 1, "simplification": 2, "soundness": 3}
        self.oracles = tuple(sorted(oracles, key=lambda oracle: order.get(oracle.name, 99)))

    def run(self, seeds: Iterable[int]) -> FuzzSummary:
        """Run *seeds*, raising an actionable assertion for the first failure."""
        summary = FuzzSummary()
        for seed in seeds:
            case = self.generate(seed)
            first_failure: tuple[Oracle, OracleResult] | None = None
            for node in case.model.graph:
                summary.op_coverage[(node.domain or "", node.op_type)] += 1
            for oracle in self.oracles:
                if not oracle.applicable(case):
                    result = OracleResult.skipped(oracle.name, "oracle not applicable")
                else:
                    result = oracle.check(case)
                summary.results[result.status] += 1
                if result.status == "SKIP" and result.reason is not None:
                    summary.skips[result.reason] += 1
                if result.status == "FAIL" and first_failure is None:
                    first_failure = (oracle, result)
            if first_failure is not None:
                raise AssertionError(self._failure_message(case, *first_failure))
        return summary

    @staticmethod
    def _failure_message(case: FuzzCase, oracle: Oracle, result: OracleResult) -> str:
        command = f"FUZZ_SEED={case.seed} python3 -m pytest tests/shape_inference_fuzz_test.py"
        snippet = (
            "from tests.fuzz._generator import generate\n"
            f"case = generate({case.seed})\n"
            f"# Re-run {oracle.name}; ground truth={result.details.get('ground_truth')!r}."
        )
        artifact_dir = os.environ.get("FUZZ_ARTIFACT_DIR")
        if artifact_dir:
            from tests.fuzz._repro import emit_onnx, render_reproducer
            from tests.fuzz._shrink import DeltaShrinker, failure_signature

            minimized = DeltaShrinker(oracle).shrink(case)
            artifact = emit_onnx(minimized, Path(artifact_dir) / f"seed-{case.seed}.onnx")
            snippet = render_reproducer(
                minimized,
                failure_signature(oracle, result),
                result,
                artifact_name=str(artifact),
            )
        return (
            f"fuzz failure: oracle={oracle.name} seed={case.seed} value={result.value_name!r} "
            f"kind={result.kind!r}: {result.reason}; details={result.details!r}\n"
            f"reproduce: {command}\n{snippet}"
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
                "outcomes": dict(sorted(summary.results.items())),
                "skips": dict(sorted(summary.skips.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return destination
