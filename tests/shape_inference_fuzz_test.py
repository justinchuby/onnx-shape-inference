# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fast-tier replay for generated shape-inference graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from onnx_shape_inference._fuzz._harness import FuzzHarness, write_coverage_report
from onnx_shape_inference._fuzz._oracles import (
    CrashOracle,
    DifferentialOracle,
    SimplificationOracle,
    SoundnessOracle,
)

pytestmark = [pytest.mark.fuzz, pytest.mark.fuzz_long]

FUZZ_SEED = int(os.environ.get("FUZZ_SEED", "0"))
FAST_CASES = int(os.environ.get("FUZZ_CASES", "150"))
_CORPUS_PATH = Path(__file__).with_name("fuzz_corpus") / "seeds.json"


def _generate():
    try:
        from onnx_shape_inference._fuzz._generator import generate
    except ImportError:
        pytest.skip("fuzz generator is not installed")
    return generate


def _corpus_seeds() -> list[int]:
    return list(json.loads(_CORPUS_PATH.read_text())["seeds"])


def test_fast_seeded_fuzzing():
    generate = _generate()
    seeds = _corpus_seeds() + list(range(FUZZ_SEED, FUZZ_SEED + FAST_CASES))
    oracles = [CrashOracle(), DifferentialOracle(), SimplificationOracle()]
    soundness = SoundnessOracle()
    if soundness.applicable(generate(seeds[0])):
        oracles.append(soundness)
    summary = FuzzHarness(generate, oracles).run(seeds)
    if coverage_path := os.environ.get("FUZZ_COVERAGE_PATH"):
        write_coverage_report(summary, coverage_path)
    assert summary.results
