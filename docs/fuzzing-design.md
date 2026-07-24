# Shape-inference fuzzing design

This document describes how the shape-inference fuzzer under `tests/fuzz/` is
designed. For the practical "I have a failing seed, now what?" playbook, see
[fuzzing.md](fuzzing.md).

## Goals

The fuzzer exists to find shape-inference bugs that hand-written tests miss:
crashes, non-idempotent inference, disagreements with ONNX reference inference,
and — most importantly — **unsound** results where our symbolic inference claims
a dtype/rank/dimension that a real execution contradicts.

Three properties are non-negotiable:

- **Deterministic.** Every case is a pure function of an integer seed, so a
  failure always reproduces from its seed alone (no saved corpus required).
- **Fast by default.** A small tier runs inside the normal `pytest` suite, so
  regressions surface in CI without a nightly wait. The expensive checks are
  sampled, not run on every case.
- **Actionable.** A failure prints the seed, a minimized reproducer, and the
  ground-truth expectation to assert against — never the buggy inferred value.

## Architecture

The pipeline is a set of small, single-responsibility modules under
`tests/fuzz/`:

| Module | Responsibility |
| --- | --- |
| `_generator.py` | Turn a seed into a valid `FuzzCase` (model + symbolic metadata) |
| `_types.py` | Shared dataclasses: `FuzzCase`, `OracleResult`, `SymbolConstraint` |
| `_binding.py` | Bind symbolic dims to concrete integers and materialize a runnable model |
| `_oracles.py` | The four independent correctness checks |
| `_runtime_worker.py` | Subprocess entry point that isolates ONNX Runtime execution |
| `_harness.py` | Drive seeds through applicable oracles; aggregate coverage; format failures |
| `_shrink.py` | Delta-shrink a failing case while preserving its failure signature |
| `_repro.py` | Emit a minimized `.onnx` artifact and paste-ready regression guidance |

The flow for one seed:

```
seed ─► _generator.generate ─► FuzzCase ─► _harness ─► for each applicable Oracle:
                                                          oracle.check(case) ─► PASS/FAIL/SKIP
                                          first FAIL ─► _shrink ─► _repro ─► actionable message
```

## Generation (`_generator.py`)

`generate(seed)` builds a `FuzzCase` deterministically:

- **Template + planner selection.** The seed indexes into a set of graph
  *templates* (e.g. `shape_slice_concat_reshape`, `constant_of_shape`, or a
  plain input) and a *planner* op. Templates exercise the `sym_data`
  propagation chains (`Shape → Slice → Concat → Reshape`); planners are the
  ops that need carefully structured inputs.
- **Schema-driven op selection.** The generator reads the shape-inference
  registry (`_registry.registry.iter_supported()` / `version_boundaries()`) so
  the op set and opset boundaries track the real implementation, not a
  hand-maintained list. After the planner op, several more ops are appended by
  weighted choice to grow a multi-op DAG.
- **Structured-input planners.** Ops whose validity depends on the exact shape
  or values of their inputs (Reshape, Slice, Tile, Expand, Range, TopK, Pad,
  OneHot, ConstantOfShape, Resize, DepthToSpace/SpaceToDepth, PRelu, Conv,
  Pool, MatMul, Gemm, Gather, Concat, Split, LayerNormalization, …) have
  dedicated planners in `_op_planners`. Ops that need too much structure for
  the generic path are listed in `_GENERIC_EXCLUDED_OPS` (e.g. Attention,
  ConvTranspose, LSTM/GRU/RNN, If/Loop/Scan, QLinearConv) and only appear via
  their own planner/template.
- **Symbolic dimensions with constraints.** Dimensions may be symbolic. Each
  symbol carries a `SymbolConstraint` (`minimum`, `maximum`, `divisible_by`)
  recording the values that keep the model valid — for example a divisibility
  requirement from a DepthToSpace blocksize. These constraints drive both
  concretization and shrinking.
- **Data-dependent values.** Outputs of ops in `DATA_DEPENDENT_OPS` (NonZero,
  Compress, Unique, TopK, Range, Pad, NonMaxSuppression, StringSplit) have
  runtime-determined dimensions. The generator records the affected value names
  in `data_dependent_values` so oracles can skip *only* those dimensions while
  still checking rank, dtype, and concrete sibling dims.
- **Control flow.** `_maybe_add_control_flow` may wrap part of the graph in a
  subgraph-bearing node so nested-graph traversal is exercised.

The result is a `FuzzCase` holding the model, the seed, the symbolic-dim
metadata, and lazily-populated caches for inference/runtime results.

## Binding and materialization (`_binding.py`)

To run a symbolic model, symbols must become integers. `bind_symbols` assigns
each symbol a deterministic concrete value that respects its `SymbolConstraint`
(honoring `minimum`/`maximum`/`divisible_by`, biased toward small primes so
distinct symbols get distinct sizes and accidental coincidences are unlikely).
`materialize_model` then substitutes those bindings to produce a fully concrete,
executable model. An optional `include_edge_dims` mode probes boundary sizes.

## Oracles (`_oracles.py`)

An `Oracle` is a cheap `applicable(case)` gate plus a `check(case)` that returns
`PASS`/`FAIL`/`SKIP` **without raising** on a model discrepancy (only the
harness decides to raise). Keeping each oracle independent means a new class of
bug becomes a new oracle rather than a change to existing ones.

### CrashOracle (`crash`)

Runs `infer_symbolic_shapes` twice on a copy of the model:

- A wall-clock budget (POSIX `setitimer` + `faulthandler`) turns a hang into a
  `hang` failure instead of a stuck test.
- `OpUsageError`/`ShapeInferenceError` are treated as legitimate rejections of a
  malformed model (`SKIP`), not crashes.
- Any other exception is an `exception` failure.
- The two runs must produce identical shapes; otherwise inference is
  non-idempotent (`idempotence` failure).

A `malformed=True` variant inverts the expectation to assert that intentionally
broken mutations *do* raise `OpUsageError`.

### DifferentialOracle (`differential`)

Compares our inference against ONNX reference inference
(`onnx.shape_inference.infer_shapes(strict_mode=False, data_prop=True)`) on the
same model. For every value present in both results it checks dtype equality,
rank equality, and equality of **concrete** dimensions (symbolic dims are not
required to match names). Where our inference can't run, it `SKIP`s rather than
failing.

### SoundnessOracle (`soundness`)

The strongest check: does a real execution agree with our symbolic inference?
Because it runs ONNX Runtime it is the most expensive, so it is **sampled**
(`seed % sample_rate == 0`, default rate 16) and skipped when `onnxruntime` is
not installed.

Its key design choice is **per-node isolation with runtime ground truth**:

1. Materialize a concrete model and pick random feeds for graph inputs.
2. Walk nodes in order. For each node, build a single-node model, run *our*
   symbolic inference on it, and run the node under ONNX Runtime.
3. Feed each node its **actual upstream ONNX Runtime output arrays** — never our
   inferred values — so a wrong value is caught at the node that produced it
   instead of being masked on both sides. Small 1-D integer inputs and
   initializers are baked in as constants so shape-tensor logic is exercised.
4. Compare our inferred output dtype, rank, and concrete dims against the
   runtime facts, skipping dimensions flagged as data-dependent.

ONNX Runtime execution happens in a subprocess (`_runtime_worker.py`) so a
native crash or hang in ORT cannot take down the test process; the worker
writes back output dtype/shape facts and, when requested, the raw arrays as an
`.npz` archive for the next node.

### SimplificationOracle (`simplification`)

Guards the symbolic-dimension simplifier. The engine records each dimension's
pre-simplification expression; this oracle re-parses both the recorded
expression and the final simplified one, substitutes many random positive
integer assignments, and fails (`symbolic_dim`) if they ever disagree. This
catches an unsound algebraic rewrite (e.g. an invalid cancellation) without
needing a runtime.

## Harness (`_harness.py`)

`FuzzHarness.run(seeds)` iterates seeds, orders oracles deterministically
(`crash → differential → simplification → soundness`), records per-op coverage,
and tallies `PASS`/`FAIL`/`SKIP`. On the **first** failing oracle it raises an
`AssertionError` whose message contains:

- the seed and a ready-to-run replay command,
- the failing value name, failure `kind`, and the ground-truth expectation, and
- when `FUZZ_ARTIFACT_DIR` is set, a **minimized** case (via the shrinker) plus a
  saved `.onnx` artifact and a paste-ready reproducer snippet.

`write_coverage_report` emits per-op counts and outcome tallies that nightly
runs publish as an artifact.

## Shrinking and reproducers (`_shrink.py`, `_repro.py`)

`DeltaShrinker` greedily reduces a failing case — concretizing symbols and
minimizing dimensions — and keeps a reduction only if it preserves the original
`FailureSignature` (`oracle`, `value_name`, `kind`), so shrinking can't
accidentally "reproduce" a *different* bug. `_repro.emit_onnx` serializes the
minimized model and `render_reproducer` prints guidance that points at the
correct colocated test file and uses the oracle's ground-truth expectation as
the assertion target.

## Test tiers

- **Fast tier (in-suite).** `tests/shape_inference_fuzz_test.py` runs a fixed
  corpus of seeds plus a window starting at `FUZZ_SEED` (default 0,
  `FUZZ_CASES` cases). It uses the crash, differential, and simplification
  oracles on every case and samples the soundness oracle. It is marked `fuzz`
  and runs under a plain `pytest` (no marker exclusion), so it must stay within
  a small time budget.
- **Nightly tier.** A time-boxed run (marked `fuzz_long`) widens the seed range,
  raises the soundness sample rate, and sets `FUZZ_ARTIFACT_DIR` /
  `FUZZ_COVERAGE_PATH` to publish minimized artifacts and coverage.

Environment knobs: `FUZZ_SEED`, `FUZZ_CASES`, `FUZZ_SOUNDNESS_SAMPLE_RATE`,
`FUZZ_ARTIFACT_DIR`, `FUZZ_COVERAGE_PATH`.

## Turning a finding into a regression test

See [fuzzing.md](fuzzing.md). In short: reproduce with the printed
`FUZZ_SEED=<seed> …` command, then add a permanent test — a parameterized case
in the op's colocated `_ops/_<op>_test.py` for a single-op bug, or a builder in
`tests/shape_inference_cases_test.py` for a multi-op graph — asserting the
oracle's ONNX Runtime/reference expectation, never the buggy inferred shape. Do
not commit the standalone `.onnx` artifact.
