# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable crash, differential, soundness, and simplification oracles."""

from __future__ import annotations

import copy
import faulthandler
import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import onnx_ir as ir
import sympy

from onnx_shape_inference import OpUsageError, ShapeInferenceError, infer_symbolic_shapes
from onnx_shape_inference._fuzz._binding import (
    bind_symbols,
    evaluate_dim,
    iter_values,
    materialize_model,
)
from onnx_shape_inference._fuzz._types import FuzzCase, OracleResult
from onnx_shape_inference._symbolic_shapes import parse_symbolic_expression

__all__ = [
    "CrashOracle",
    "DifferentialOracle",
    "Oracle",
    "OracleResult",
    "SimplificationOracle",
    "SoundnessOracle",
]


@runtime_checkable
class Oracle(Protocol):
    """A cheap applicability gate and a single independent correctness check."""

    name: str

    def applicable(self, case: FuzzCase) -> bool:
        """Return whether this oracle can inspect *case* cheaply."""

    def check(self, case: FuzzCase) -> OracleResult:
        """Return PASS, FAIL, or SKIP without raising for a model discrepancy."""


class _AlarmExpiredError(TimeoutError):
    """Raised when a fast-tier graph exceeds its wall-clock budget."""


@contextmanager
def _wall_budget(seconds: float):
    """Interrupt Python inference after *seconds* on POSIX main threads."""
    if not hasattr(signal, "setitimer"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)

    def alarm_handler(_signum, _frame):
        raise _AlarmExpiredError(f"shape inference exceeded {seconds:g}s")

    signal.signal(signal.SIGALRM, alarm_handler)
    try:
        with faulthandler_timeout(seconds):
            yield
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


@contextmanager
def faulthandler_timeout(seconds: float):
    """Request a traceback dump for a slow fast-tier graph when supported."""
    try:
        faulthandler.dump_traceback_later(seconds)
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _snapshot(model: ir.Model) -> tuple[tuple[str, str | None, str | None], ...]:
    values = (
        (value.name, str(value.dtype) if value.dtype is not None else None, str(value.shape))
        for value in iter_values(model.graph)
    )
    return tuple(sorted(values))


def _infer_ours(case: FuzzCase) -> ir.Model:
    if case.our_inference_result is None:
        inferred = copy.deepcopy(case.model)
        infer_symbolic_shapes(inferred)
        case.our_inference_result = inferred
    return case.our_inference_result


class CrashOracle:
    """Detect unexpected exceptions, hangs, and non-idempotent inference."""

    name = "crash"

    def __init__(self, *, timeout_seconds: float = 5.0, malformed: bool = False) -> None:
        self.timeout_seconds = timeout_seconds
        self.malformed = malformed

    def applicable(self, case: FuzzCase) -> bool:
        return case.model is not None

    def check(self, case: FuzzCase) -> OracleResult:
        model = copy.deepcopy(case.model)
        try:
            with _wall_budget(self.timeout_seconds):
                infer_symbolic_shapes(model)
                first = _snapshot(model)
                infer_symbolic_shapes(model)
                second = _snapshot(model)
        except _AlarmExpiredError as error:
            return OracleResult.failed(
                self.name, str(error), value_name="<graph>", kind="hang"
            )
        except OpUsageError as error:
            if self.malformed:
                return OracleResult.passed(self.name)
            return OracleResult.skipped(
                self.name, f"model rejected with OpUsageError: {error}"
            )
        except ShapeInferenceError as error:
            if self.malformed:
                return OracleResult.failed(
                    self.name,
                    f"malformed mutation raised ShapeInferenceError, expected OpUsageError: {error}",
                    value_name="<graph>",
                    kind="exception",
                )
            return OracleResult.skipped(
                self.name, f"model rejected with ShapeInferenceError: {error}"
            )
        except Exception as error:
            return OracleResult.failed(
                self.name,
                f"unexpected {type(error).__name__}: {error}",
                value_name="<graph>",
                kind="exception",
            )
        if self.malformed:
            return OracleResult.failed(
                self.name,
                "malformed mutation did not raise OpUsageError",
                value_name="<graph>",
                kind="missing_intended_error",
            )
        if first != second:
            return OracleResult.failed(
                self.name,
                "inference is not idempotent",
                value_name="<graph>",
                kind="idempotence",
            )
        case.our_inference_result = model
        return OracleResult.passed(self.name)


class DifferentialOracle:
    """Compare concrete inference facts with ONNX reference shape inference."""

    name = "differential"

    def applicable(self, case: FuzzCase) -> bool:
        return case.model is not None

    def check(self, case: FuzzCase) -> OracleResult:
        try:
            import onnx
        except ImportError:
            return OracleResult.skipped(self.name, "onnx is unavailable")
        try:
            ours = _infer_ours(case)
            proto = ir.serde.serialize_model(copy.deepcopy(case.model))
            reference = onnx.shape_inference.infer_shapes(
                proto, strict_mode=False, data_prop=True
            )
            reference_model = ir.serde.deserialize_model(reference)
        except (OpUsageError, ShapeInferenceError) as error:
            return OracleResult.skipped(self.name, f"our inference is unknown: {error}")
        except Exception as error:
            return OracleResult.skipped(
                self.name, f"reference inference unavailable: {type(error).__name__}: {error}"
            )
        case.onnx_ref_result = reference_model

        ours_by_name = {value.name: value for value in iter_values(ours.graph) if value.name}
        ref_by_name = {
            value.name: value for value in iter_values(reference_model.graph) if value.name
        }
        checked = 0
        for name in sorted(ours_by_name.keys() & ref_by_name.keys()):
            ours_value, ref_value = ours_by_name[name], ref_by_name[name]
            if ours_value.dtype is None or ref_value.dtype is None:
                continue
            if ours_value.dtype != ref_value.dtype:
                return OracleResult.failed(
                    self.name,
                    "dtype contradiction",
                    value_name=name,
                    kind="dtype",
                    expected=ref_value.dtype,
                    actual=ours_value.dtype,
                )
            if ours_value.shape is None or ref_value.shape is None:
                continue
            if ours_value.shape.rank() != ref_value.shape.rank():
                return OracleResult.failed(
                    self.name,
                    "rank contradiction",
                    value_name=name,
                    kind="rank",
                    expected=ref_value.shape.rank(),
                    actual=ours_value.shape.rank(),
                )
            checked += 1
            for index, (ours_dim, ref_dim) in enumerate(
                zip(ours_value.shape, ref_value.shape)
            ):
                if (
                    isinstance(ours_dim, int)
                    and isinstance(ref_dim, int)
                    and ours_dim != ref_dim
                ):
                    return OracleResult.failed(
                        self.name,
                        "concrete dimension contradiction",
                        value_name=name,
                        kind="concrete_dim",
                        details={"index": index},
                        expected=ref_dim,
                        actual=ours_dim,
                    )
        if checked == 0:
            return OracleResult.skipped(
                self.name, "no shared values with known dtype and rank"
            )
        return OracleResult.passed(self.name)


def _np_dtype(dtype: ir.DataType) -> np.dtype | None:
    mapping = {
        ir.DataType.BOOL: np.bool_,
        ir.DataType.FLOAT: np.float32,
        ir.DataType.DOUBLE: np.float64,
        ir.DataType.INT32: np.int32,
        ir.DataType.INT64: np.int64,
        ir.DataType.UINT8: np.uint8,
    }
    return mapping.get(dtype)


def _data_dependent(case: FuzzCase, value_name: str, index: int) -> bool:
    return value_name in case.data_dependent_values


def _runtime_shapes(
    proto,
    feeds: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Run ORT in a subprocess so an unsafe kernel becomes a coverage hole."""
    import onnx

    directory = Path.cwd() / ".fuzz-runtime"
    directory.mkdir(exist_ok=True)
    stem = f"{os.getpid()}-{seed}"
    model_path = directory / f"{stem}.onnx"
    feeds_path = directory / f"{stem}.npz"
    result_path = directory / f"{stem}.json"
    try:
        onnx.save(proto, model_path)
        np.savez(feeds_path, **feeds)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_runtime_worker.py")),
                str(model_path),
                str(feeds_path),
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip() or f"runtime exited {completed.returncode}"
            )
        return json.loads(result_path.read_text())
    finally:
        for path in (model_path, feeds_path, result_path):
            path.unlink(missing_ok=True)
        with suppress(OSError):
            directory.rmdir()


class SoundnessOracle:
    """Concretize symbolic models and compare inferred facts against ONNX Runtime."""

    name = "soundness"

    def __init__(self, *, sample_rate: int = 16) -> None:
        self.sample_rate = max(1, sample_rate)

    def applicable(self, case: FuzzCase) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return case.model is not None and case.seed % self.sample_rate == 0

    def check(self, case: FuzzCase) -> OracleResult:
        try:
            import onnx
            import onnxruntime  # noqa: F401
        except ImportError:
            return OracleResult.skipped(self.name, "onnxruntime is unavailable")

        bindings = bind_symbols(case)
        try:
            symbolic = _infer_ours(case)
            concrete = materialize_model(case, bindings)
            infer_symbolic_shapes(concrete)
            proto = ir.serde.serialize_model(concrete)
            concrete_values = {
                value.name: value for value in iter_values(concrete.graph) if value.name
            }
            existing_outputs = {output.name for output in proto.graph.output}
            for name in concrete_values:
                if name in existing_outputs:
                    continue
                proto.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
            rng = np.random.default_rng(case.seed)
            feeds: dict[str, np.ndarray] = {}
            input_names = {value.name for value in concrete.graph.inputs}
            for value in concrete.graph.inputs:
                if value.name not in input_names:
                    continue
                if value.shape is None or value.dtype is None:
                    return OracleResult.skipped(
                        self.name, f"input {value.name} is not concrete"
                    )
                dtype = _np_dtype(value.dtype)
                if dtype is None:
                    return OracleResult.skipped(
                        self.name, f"ORT dtype unavailable for {value.dtype}"
                    )
                shape = tuple(int(dim) for dim in value.shape)
                if dtype == np.bool_:
                    feeds[value.name] = rng.integers(0, 2, shape, dtype=np.int8).astype(dtype)
                elif np.issubdtype(dtype, np.integer):
                    feeds[value.name] = rng.integers(1, 4, shape, dtype=dtype)
                else:
                    feeds[value.name] = rng.uniform(0.5, 1.5, shape).astype(dtype)
            runtime = _runtime_shapes(proto, feeds, seed=case.seed)
            case.onnxruntime_result = runtime
        except Exception as error:
            return OracleResult.skipped(
                self.name, f"onnxruntime cannot run graph: {type(error).__name__}: {error}"
            )

        for value in iter_values(symbolic.graph):
            if not value.name or value.name not in runtime or value.shape is None:
                continue
            actual = runtime[value.name]
            actual_dtype = np.dtype(actual["dtype"])
            actual_shape = actual["shape"]
            expected_dtype = _np_dtype(value.dtype) if value.dtype is not None else None
            if expected_dtype is not None and expected_dtype != actual_dtype:
                return OracleResult.failed(
                    self.name,
                    "runtime dtype contradiction",
                    value_name=value.name,
                    kind="dtype",
                    expected=actual_dtype,
                    actual=value.dtype,
                )
            if value.shape.rank() != len(actual_shape):
                return OracleResult.failed(
                    self.name,
                    "runtime rank contradiction",
                    value_name=value.name,
                    kind="rank",
                    expected=len(actual_shape),
                    actual=value.shape.rank(),
                )
            for index, dim in enumerate(value.shape):
                if _data_dependent(case, value.name, index):
                    continue
                predicted = evaluate_dim(dim, bindings)
                if predicted is not None and predicted != actual_shape[index]:
                    return OracleResult.failed(
                        self.name,
                        "runtime dimension contradiction",
                        value_name=value.name,
                        kind="concrete_dim",
                        details={"index": index, "binding": bindings},
                        expected=actual_shape[index],
                        actual=predicted,
                    )
        return OracleResult.passed(self.name)


class SimplificationOracle:
    """Check simplified symbolic dimensions against their recorded pre-simplify form."""

    name = "simplification"

    def __init__(self, *, samples: int = 64) -> None:
        self.samples = samples

    def applicable(self, case: FuzzCase) -> bool:
        return bool(case.pre_simplify_dims)

    def check(self, case: FuzzCase) -> OracleResult:
        try:
            inferred = _infer_ours(case)
        except (OpUsageError, ShapeInferenceError) as error:
            return OracleResult.skipped(self.name, f"our inference is unknown: {error}")
        values = {value.name: value for value in iter_values(inferred.graph) if value.name}
        rng = np.random.default_rng(case.seed)
        for (name, index), before_text in sorted(case.pre_simplify_dims.items()):
            value = values.get(name)
            if value is None or value.shape is None or index >= value.shape.rank():
                continue
            after = value.shape[index]
            if isinstance(after, int):
                after_expr = sympy.Integer(after)
            else:
                after_expr = parse_symbolic_expression(str(after))
            before_expr = parse_symbolic_expression(before_text)
            symbols = sorted(
                before_expr.free_symbols | after_expr.free_symbols, key=lambda item: item.name
            )
            for _ in range(self.samples):
                substitutions = {symbol: int(rng.integers(1, 12)) for symbol in symbols}
                if before_expr.subs(substitutions) != after_expr.subs(substitutions):
                    return OracleResult.failed(
                        self.name,
                        "symbolic simplification contradiction",
                        value_name=name,
                        kind="symbolic_dim",
                        details={"index": index},
                        expected=str(before_expr),
                        actual=str(after_expr),
                    )
        return OracleResult.passed(self.name)
