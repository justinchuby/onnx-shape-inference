# Released operator support

## Release audit (2026-10-19)

The audit uses the latest **released**, non-prerelease versions advertised by both
GitHub releases and the package index:

| Upstream | Release | Schema source |
|----------|---------|---------------|
| ONNX | [1.23.2](https://github.com/onnx/onnx/releases/tag/v1.23.2), default-domain opset 28 | `onnx.defs.get_all_schemas()` and `get_all_schemas_with_history()` |
| ONNX Runtime | [1.31.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0) | `onnxruntime.capi._pybind_state.get_all_operator_schema()` |
| ONNX Runtime comparison baseline | 1.29.0 | The same API, imported in a separate process from an isolated installation |

Every latest schema was compared with `registry.get(domain, name, since_version)`.
The ORT comparison includes all domains, not just the operators present in the
ONNX backend corpus. A separate comparison of attributes, input/output contracts,
and release-tag C++ inference definitions checks meaningful changes to existing
operator names. An older registration remains valid for a newer schema when its
shape/type rule is unchanged; registration version numbers are not coverage gaps
by themselves.

Authoritative ORT sources:
[generated contrib schemas](https://github.com/microsoft/onnxruntime/blob/v1.31.0/docs/ContribOperators.md),
[attention/recurrent definitions](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/graph/contrib_ops/bert_defs.cc),
[other contrib definitions](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/graph/contrib_ops/contrib_defs.cc),
[quantization definitions](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/graph/contrib_ops/quantization_defs.cc),
and [CSA window planning](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/contrib_ops/cpu/sparse/sparse_attention_indexer_common.h).

## New operator coverage

`SwiGLU` is new in standard ONNX opset 28. Its two input shapes must be identical:
it does **not** broadcast. The output preserves the common shape and dtype.

All 13 contrib names added between the ORT 1.29.0 and 1.31.0 schema APIs are
registered at `com.microsoft` version 1:

| Operator | Inference behavior |
|----------|--------------------|
| `BranchwiseRMSNorm`, `ScaledSiLU` | Preserve activation shape and dtype, including unknown rank |
| `HyperConnectionPreMix` | Remove the branch axis for grouped streams; divide the final width by `num_branches` for flattened streams |
| `HyperConnectionPostMix` | Preserve the stream shape and dtype |
| `EngramGate` | Both the primary and optional normalized outputs preserve the key's hyper-connection layout |
| `NGramHashMapping` | Append `(max_ngram_size - 1) * n_head_per_ngram` to the ID shape; infer trailing-ID carry state |
| `VarlenNGramHashMapping` | Infer packed hash IDs, per-request ID/INT32 segment state, and compact prefix-state updates |
| `VarlenCausalConvWithState` | Preserve token-major activations; infer dilated per-request carry state and optional compact updates |
| `GatedDeltaNet` | Support separate and packed QKV, token-major and padded inputs, V-major recurrent state, and compact transition factors; state outputs are always FLOAT |
| `SparseAttentionIndexer` | Infer QSA/CSA selection capacity, growing versus fixed-capacity keys, and CSA overlap/leftover buffer shapes |
| `PackedSparseAttentionIndexer` | Infer packed indices/counts, fixed-size state outputs, and optional QSA compact updates; preserve the empty QSA gate-output slot |
| `SparsePagedAttention` | Reuse paged QKV output geometry and preserve each cache's storage dtype and shape |
| `DynamicSparseAttention` | Infer packed or separate dense QKV output width, fixed-capacity caches, and constant-total-length prefill caches |

Symbolic arithmetic preserves relationships such as batch size being the length
of cumulative offsets minus one, dilated state length, flattened stream widths,
and growing QSA cache lengths. Truly unresolved dimensions receive fresh symbols.
Optional empty output slots are not populated.

## New semantics of existing operators

- Contrib `CausalConvWithState` supports `dilation` and `channels_last`, including
  split trailing channel axes and a leading `state_window` axis.
- Contrib `GroupQueryAttention` with `sliding_window_cache=1` preserves cache
  capacity instead of appending the current sequence length. Its optional QK
  output still uses absolute total sequence length, not window capacity.
- Contrib `PagedAttention` accepts packed INT4/FLOAT4 latent cache widths,
  including the rounded-up byte count for an odd logical head width.
- Packed-token `MoE`/`QMoE`, inverse-GQA `LinearAttention`, and quantized
  `PagedAttention` cache aliases already follow the released output rules; they
  do not need duplicate registrations. Changes to gate activations, expert weight
  quantization, cache metadata bounds, and activation scales do not change their
  output shape/type rules.
- ONNX opset 28 adds signed `BitShift`, floating-point floor `Mod`, and a
  `SpaceToDepth` `mode` attribute. Existing inference already handles their dtype,
  broadcast, and spatial shape rules. Other opset-28 schema revisions extend
  accepted types or numerical behavior without changing output geometry.

The generated ONNX backend corpus remains enabled in full; no cases are skipped
or filtered to achieve this coverage.

## Remaining limitations

- **FLOAT6 representation:** ONNX 1.23.2 adds `float6e2m3` and `float6e3m2` to
  `Cast`, `QuantizeLinear`, and `DequantizeLinear`. The audited `onnx-ir 1.0.0`
  `DataType` enum cannot represent either type. These require upstream IR dtype
  support before they can be supported here; substituting another dtype would be
  incorrect. Other dtype extensions use the existing generic propagation rules.
- CSA buffer sizes that depend on unresolved symbolic window-plan branches remain
  fresh unknown dimensions rather than assuming whether a window closes.
  Runtime selection values, hash values, and recurrent state contents are not
  propagated as shape-tensor data.
- Older missing schemas are not new release gaps. Standard-domain ONNX operator
  names are covered, but historical `ai.onnx.ml` operators (including map-based
  transforms, classifiers/regressors, and `ZipMap`) and training `Gradient` still
  have gaps. ORT also has older unsupported search/decoder, quantization, and
  hashing/embedding operators, deprecated standard-domain operators, TensorRT
  plugin schemas, and `com.ms.internal.nhwc` / `com.microsoft.nchwc` layout ops.
  Those need separate schema-specific implementations, not aliases based on names.

To repeat the name audit after collecting the registry:

```python
import onnx
from onnxruntime.capi import _pybind_state

from onnx_shape_inference import registry

registry.collect()
for schemas in (
    onnx.defs.get_all_schemas(),
    _pybind_state.get_all_operator_schema(),
):
    missing = sorted({
        (schema.domain, schema.name, schema.since_version)
        for schema in schemas
        if registry.get(schema.domain, schema.name, schema.since_version) is None
    })
    print(missing)
```

This reports name/version dispatch only. Check changed schema contracts and
upstream inference code as well before claiming semantic coverage.
