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
from onnx_shape_inference._symbolic_shapes import parse_symbolic_expression
from tests.fuzz._binding import (
    bind_symbols,
    iter_values,
    materialize_model,
)
from tests.fuzz._types import FuzzCase, OracleResult

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


def _runtime_feed(value: ir.Value, rng: np.random.Generator) -> np.ndarray | None:
    """Create a deterministic concrete feed for one materialized value."""
    if value.shape is None or value.dtype is None:
        return None
    dtype = _np_dtype(value.dtype)
    if dtype is None:
        return None
    shape = tuple(int(dim) for dim in value.shape)
    if dtype == np.bool_:
        return rng.integers(0, 2, shape, dtype=np.int8).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(1, 4, shape, dtype=dtype)
    return rng.uniform(0.5, 1.5, shape).astype(dtype)


def _single_node_model(
    node: ir.Node,
    *,
    opset_imports: dict[str, int],
    name: str,
    value_arrays: dict[str, np.ndarray],
    bake_names: set[str],
) -> tuple[ir.Model, list[ir.Value]]:
    """Build a concrete single-node model fed with ground-truth arrays.

    Every input's type and shape are derived from its ground-truth array in
    *value_arrays* (a graph input's random feed, an authored initializer, or an
    upstream node's actual ONNX Runtime output). Names in *bake_names* become
    initializers carrying that array as their ``const_value`` so that our
    isolated inference reads correct, ground-truth shape values; every other
    input becomes a fed graph input whose value inference must treat as unknown.
    """
    inputs: list[ir.Value] = []
    initializers: list[ir.Value] = []
    copied: dict[str, ir.Value] = {}
    node_inputs: list[ir.Value] = []
    for value in node.inputs:
        if value is None:
            raise ValueError("optional inputs are not supported by the node isolator")
        if value.name in copied:
            node_inputs.append(copied[value.name])
            continue
        array = value_arrays[value.name]
        dtype = ir.DataType.from_numpy(array.dtype)
        shape = ir.Shape(list(array.shape))
        if value.name in bake_names:
            clone = ir.Value(
                name=value.name,
                type=ir.TensorType(dtype),
                shape=shape,
                const_value=ir.tensor(array, name=value.name),
            )
            initializers.append(clone)
        else:
            clone = ir.Value(name=value.name, type=ir.TensorType(dtype), shape=shape)
            inputs.append(clone)
        copied[value.name] = clone
        node_inputs.append(clone)
    isolated = ir.Node(
        node.domain,
        node.op_type,
        node_inputs,
        num_outputs=len(node.outputs),
        attributes=copy.deepcopy(node.attributes),
    )
    for output, source in zip(isolated.outputs, node.outputs):
        output.name = source.name
    graph = ir.Graph(
        inputs,
        list(isolated.outputs),
        nodes=[isolated],
        initializers=initializers,
        opset_imports=opset_imports,
        name=name,
    )
    return ir.Model(graph, ir_version=10), inputs


def _runtime_shapes(
    proto,
    feeds: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Run ORT in a subprocess so an unsafe kernel becomes a coverage hole.

    Each returned fact carries the output ``dtype`` and ``shape`` and, when the
    output is numeric, its actual runtime ``array`` so that the value can be fed
    to downstream isolated nodes as ground truth.
    """
    import onnx

    directory = Path.cwd() / ".fuzz-runtime"
    directory.mkdir(exist_ok=True)
    stem = f"{os.getpid()}-{seed}"
    model_path = directory / f"{stem}.onnx"
    feeds_path = directory / f"{stem}.npz"
    result_path = directory / f"{stem}.json"
    arrays_path = directory / f"{stem}-out.npz"
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
                str(arrays_path),
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
        facts = json.loads(result_path.read_text())
        if arrays_path.exists():
            with np.load(arrays_path, allow_pickle=False) as archive:
                for output_name in archive.files:
                    if output_name in facts:
                        facts[output_name]["array"] = archive[output_name]
        return facts
    finally:
        for path in (model_path, feeds_path, result_path, arrays_path):
            path.unlink(missing_ok=True)
        with suppress(OSError):
            directory.rmdir()


def _arrays_match(ours: np.ndarray, runtime: np.ndarray) -> bool:
    """Compare a claimed constant value against the runtime array."""
    if ours.shape != runtime.shape:
        return False
    if np.issubdtype(runtime.dtype, np.floating):
        return bool(np.allclose(ours, runtime, rtol=1e-3, atol=1e-5, equal_nan=True))
    return bool(np.array_equal(ours, runtime))


class SoundnessOracle:
    """Concretize symbolic models and compare inferred facts against ONNX Runtime."""

    name = "soundness"

    def __init__(self, *, sample_rate: int = 16) -> None:
        self.sample_rate = max(1, sample_rate)

    def applicable(self, case: FuzzCase) -> bool:
        try:
            import onnxruntime  # ruff:ignore[unused-import]
        except ImportError:
            return False
        return case.model is not None and case.seed % self.sample_rate == 0

    def check(self, case: FuzzCase) -> OracleResult:
        try:
            import onnxruntime  # ruff:ignore[unused-import]
        except ImportError:
            return OracleResult.skipped(self.name, "onnxruntime is unavailable")

        bindings = bind_symbols(case)
        try:
            concrete = materialize_model(case, bindings)
        except Exception as error:
            return OracleResult.skipped(
                self.name, f"cannot materialize graph: {type(error).__name__}: {error}"
            )

        graph = concrete.graph
        graph_input_names = {value.name for value in graph.inputs if value and value.name}
        initializer_names = set(graph.initializers)

        # Ground-truth arrays keyed by value name. Graph inputs get one random
        # feed each, initializers their authored value, and every intermediate
        # its ACTUAL upstream ONNX Runtime output — never our inferred value —
        # so a wrong upstream value is caught rather than propagated to both
        # sides.
        value_arrays: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(case.seed)
        for value in graph.inputs:
            if value is None or not value.name:
                continue
            feed = _runtime_feed(value, rng)
            if feed is not None:
                value_arrays[value.name] = feed
        for init_name, initializer in graph.initializers.items():
            const = ir.convenience.get_const_tensor(initializer)
            if const is None:
                continue
            with suppress(Exception):
                value_arrays[init_name] = const.numpy()

        node_counts = {"pass": 0, "skip": 0}
        node_results: dict[str, dict[str, object]] = {}
        for index, node in enumerate(graph):
            name = node.name or f"{node.op_type}_{index}"
            input_names = [value.name for value in node.inputs if value is not None]
            if any(value is None for value in node.inputs):
                node_counts["skip"] += 1
                node_results[name] = {"status": "SKIP", "reason": "optional input"}
                continue
            if any(input_name not in value_arrays for input_name in input_names):
                node_counts["skip"] += 1
                node_results[name] = {"status": "SKIP", "reason": "upstream unavailable"}
                continue
            bake_names: set[str] = set()
            for input_name in input_names:
                if input_name in graph_input_names:
                    continue
                array = value_arrays[input_name]
                if input_name in initializer_names or (
                    array.ndim <= 1 and np.issubdtype(array.dtype, np.integer)
                ):
                    bake_names.add(input_name)
            try:
                model, feed_inputs = _single_node_model(
                    node,
                    opset_imports=case.opset_imports,
                    name=f"soundness_{case.seed}_{index}",
                    value_arrays=value_arrays,
                    bake_names=bake_names,
                )
                ours = copy.deepcopy(model)
                infer_symbolic_shapes(ours)
                expected_values = {
                    output.name: output for output in ours.graph.outputs if output.name
                }
                feeds = {value.name: value_arrays[value.name] for value in feed_inputs}
                runtime = _runtime_shapes(
                    ir.serde.serialize_model(model), feeds, seed=case.seed * 100 + index
                )
            except Exception as error:
                node_counts["skip"] += 1
                node_results[name] = {"status": "SKIP", "reason": type(error).__name__}
                continue
            for output_name, fact in runtime.items():
                array = fact.get("array")
                if isinstance(array, np.ndarray):
                    value_arrays[output_name] = array
            for output in node.outputs:
                expected = expected_values.get(output.name)
                actual = runtime.get(output.name)
                if expected is None or actual is None:
                    continue
                expected_dtype = (
                    _np_dtype(expected.dtype) if expected.dtype is not None else None
                )
                actual_dtype = np.dtype(actual["dtype"])
                if expected_dtype is not None and expected_dtype != actual_dtype:
                    return OracleResult.failed(
                        self.name,
                        "runtime dtype contradiction",
                        value_name=output.name,
                        kind="dtype",
                        expected=actual_dtype,
                        actual=expected.dtype,
                        details={"nodes": node_counts, "node": name},
                    )
                if expected.shape is None:
                    continue
                actual_shape = actual["shape"]
                if expected.shape.rank() != len(actual_shape):
                    return OracleResult.failed(
                        self.name,
                        "runtime rank contradiction",
                        value_name=output.name,
                        kind="rank",
                        expected=len(actual_shape),
                        actual=expected.shape.rank(),
                        details={"nodes": node_counts, "node": name},
                    )
                for dim_index, dim in enumerate(expected.shape):
                    if _data_dependent(case, output.name, dim_index):
                        continue
                    if not isinstance(dim, int):
                        continue
                    if dim != actual_shape[dim_index]:
                        return OracleResult.failed(
                            self.name,
                            "runtime dimension contradiction",
                            value_name=output.name,
                            kind="concrete_dim",
                            expected=actual_shape[dim_index],
                            actual=dim,
                            details={
                                "index": dim_index,
                                "binding": bindings,
                                "nodes": node_counts,
                                "node": name,
                            },
                        )
                # A concrete value claimed by our data propagation must match
                # the runtime array. This catches wrong sym_data at the node
                # that produced it, instead of feeding it to a downstream node.
                actual_array = actual.get("array")
                if (
                    output.name not in case.data_dependent_values
                    and expected.const_value is not None
                    and isinstance(actual_array, np.ndarray)
                ):
                    our_array = expected.const_value.numpy()
                    if not _arrays_match(our_array, actual_array):
                        return OracleResult.failed(
                            self.name,
                            "runtime value contradiction",
                            value_name=output.name,
                            kind="value",
                            expected=actual_array.tolist(),
                            actual=our_array.tolist(),
                            details={"nodes": node_counts, "node": name},
                        )
            node_counts["pass"] += 1
            node_results[name] = {"status": "PASS"}
        case.onnxruntime_result = node_results
        if node_counts["pass"] == 0:
            return OracleResult.skipped(
                self.name, "onnxruntime could not run any isolated node"
            )
        return OracleResult(self.name, "PASS", details={"nodes": node_counts})


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
