# Squad Decisions

## Active Decisions

### 2026-07-22T23-44-24: PR-A fuzz generator uses deterministic registry/schema-driven DAG planning
**By:** dallas
**What:** PR-A generator uses one seeded RNG, registry/schema-driven op selection, shared typed-port DAG planning, and a small table of constant-input and sym_data planners.
**Why:** Reproducible fuzzing with broad operator coverage works better when generic generation stays deterministic and specialized shape logic remains explicit.

### 2026-07-23T07:41:00+0000: soundness-oracle dtype contradictions are fixed centrally
**By:** Lambert, Coordinator
**What:** SoundnessOracle now runs our inference on each isolated single-node model with fresh, untyped outputs instead of inheriting generator-declared dtypes. This removes the dtype-contradiction false-positive class generally; per-op generator dtype planners are not the fix. PRelu remains a separate broadcast-validity generator fix.
**Why:** Keep the oracle DRY and general, eliminate the whole false-positive class, and avoid proliferating op-specific planner logic.

### 2026-07-23T08:16:00+0000: ORT runtime ground truth closes the sym_data masking gap
**By:** Ripley, Coordinator
**What:** SoundnessOracle now runs the concrete graph node-by-node in ORT, caches actual runtime output arrays, and feeds those real upstream values to downstream isolated nodes instead of inferred const_value data. It also compares claimed concrete const_value results against the runtime array at the producer node to catch wrong propagated values early.
**Why:** Use DRY runtime ground truth to eliminate downstream masking, keep oracle checks aligned with real execution, and avoid per-op propagation logic.

### 2026-07-23T07:41:00+0000: symbolic anchor adoption and reshape provenance for sound propagation (consolidated)
**By:** Ripley, Dallas
**What:** Declared output/value_info symbols are adopted by default, minted symbol identity is tracked via `ctx.is_generated_symbol`, child contexts share a symbol allocator, and reshape numel equalities/provenance drive exact divisibility instead of blind floor cancellation.
**Why:** Preserve authored names without collisions, avoid unsound renaming/cancellation, and make propagation general across subgraphs and reshape chains.

### 2026-07-22T21-36-42Z: coverage regression harness for symbolic/anchor gaps
**By:** Lambert
**What:** Added a local coverage harness for symbolic reshape/slice/resize/tile, TopK, NonZero, If anchors, tiny-LLM concat anchors, qwen forwarding, and dtype-only boundaries.
**Why:** Keep the remaining propagation gaps explicit while the engine and oracle fixes land.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
