# Native xLLM hybrid attention (experimental)

This port adds native-format dense, hybrid Inkling, and hybrid Gekko model
adapters to SGLang. It includes a strict PyTorch Distributed Checkpoint (DCP)
exporter, independent PyTorch reference operators, and differential qualification
tools.

**Status: experimental, with partial numerical qualification.** All three
architectures execute through the actual Engine, and specific tested cases now
pass full-prompt comparisons. The combined qualification reports still fail:
independent reference-model comparisons and dense/Inkling Engine cases remain
outside the strict NLL gates. Gekko's corrected raw-logit checks now match native
through near 128K, within the measured scope below. This work is not yet ready for an upstream PR or
production use. Experimental performance measurements are separate from
numerical qualification.

## Historical eager qualification and its scope

The table and counts in this section record the completed eager qualification
before Dense/Inkling decode-graph support. They remain historical report counts,
not totals for the current test suite or graph validation. The CUDA runs used
NVIDIA H200 GPUs; their results do not guarantee other inputs, devices, or runtime
settings. The bounded decode-graph configuration is described separately below.

| Check | Observed result and scope |
| --- | --- |
| Checkpoint export | Complete inventory and exact tensor byte round-trip checks passed for dense, Inkling, and Gekko. |
| CPU regression suite | 217 tests and seven subtests passed, with no failures or skips; one CUDA case was deselected. All 20 new registered test files have executable entry points and are discovered by SGLang's CI scanner (nine CPU and 12 CUDA registrations). |
| Existing SGLang config regressions | 28 passed across linear-attention registration, backend selection, model shapes and parser registration. |
| CUDA RMSNorm | 67 tests passed in each of the native and serving environments, using the same frozen native oracle fixtures. Affine and non-affine BF16 outputs matched those fixtures bitwise. |
| CUDA streaming TDN | 58 tests passed in each environment against independent native fixtures, plus two Gekko adapter/state-pool tests. Exactness is scoped to the tested finite BF16 1,536-channel, 32-group configuration. |
| CUDA fused SCA | 95 tests passed, including exact native outputs across three architectural chunks, arbitrary query partitions, saved standard FA4 outputs, and fresh-process compiled-cache reuse. |
| CUDA Gekko FFN projection | 16 tests passed, including native learned rows, partition checks, shape handling, and autocast fallbacks. |
| CUDA normalized short convolution | 174 tests passed in each environment, including mandatory public native golden cases and optional learned-input comparisons. Another 20 CPU tests cover dispatch and fallbacks. |
| Rotary inverse-frequency cache | 15 CPU tests and one CUDA lifecycle test passed. All 20 native 128K frequency/Q/K comparisons remained bitwise equal with the serving cache enabled. |
| CUDA cache/state lifecycle | Eight tests passed, including the request-slot operations exercised by that suite. |
| Dense and Inkling, 31-token prompts | Original-prompt NLL comparisons passed with maximum absolute error `9.54e-7` or less. Other checks in these runs still failed. |
| Dense and Inkling, 127-token prompts | Original-prompt checks failed with prefill chunks of 64; the completed chunk-256 controls passed at maximum absolute error `9.54e-7` or less. Forced-stream failures remain. |
| Inkling, 8,191 / 8,192 / 8,193 tokens | All 25 actual Engine checks passed with FA4 and prefill chunks of 2,048, covering full prompt NLL, forced targets, greedy decode, batching, reorder, unequal completions, and completed-slot reuse. Original-prompt maximum absolute NLL error was `9.54e-7`. |
| Gekko, 31 / 127, 4,095 / 4,096, 8,191 / 8,192, and 12,287 / 12,288 tokens | All 72 actual Engine check groups passed across these four pairs, including full prompt scores, forced scores, four-token greedy traces, batching, reorder, unequal completions and completed-slot reuse. Original-prompt maximum absolute NLL error was `9.54e-7`. |
| Gekko, 8,191 / 8,192 tokens with long decode | All 532 Engine groups passed with 260 cached decode steps after the first output token, including forced scores and greedy lifecycle checks. |
| Gekko selected Engine raw logits | All 360 measured vocabulary rows were bitwise equal to native across the six short/4K/8K lengths, using independent prefill, cached forced tokens and natural greedy decode at batch sizes one and two. This separate capture used synchronous scheduling. |
| Gekko cancellation and reuse | Ten cases passed protocol and native token/NLL checks while aborting one request, completing another, draining both and reusing a bounded two-slot state pool. |
| Gekko forced retraction | Six 64-token request traces across batch, reorder and repeat matched native IDs and logprobs exactly. The 8,192-token request underwent 89 total retractions; the concurrent 8,191-token request was the control. |
| Gekko selected long-context and varied-content raw logits | After the AWM reduction-layout correction, all 276 measured vocabulary rows matched native bitwise: repeated fixtures at 16K/32K/64K/near-128K and varied document tokens at 8K/32K/64K, independent prefill and cached forced decode, batch sizes one and two. The varied-content four-token greedy traces also matched. |
| Gekko architectural boundaries during long decode | Two varied-content requests beginning at 8,191/8,192 tokens each produced 4,098 outputs (4,097 cached decode steps). All 16 sampled vocabulary rows matched native bitwise, including offsets 4,095/4,096/4,097. This checks selected positions, not every output row. |

That eager qualification matrix reran all Gekko Engine, raw-logit and lifecycle
checks above with fused normalized short convolution and cached rotary inverse
frequencies enabled. All 604 direct Engine groups passed; all 652 selected raw
vocabulary rows were bitwise equal to native. Source hashes stayed unchanged
within each of those historical runs. Raw-logit capture uses
synchronous scheduling; the separate NLL and lifecycle suites exercise ordinary
overlap scheduling. The near-128K input starts at 131,060 tokens to respect the
Engine's exclusive 131,066 input limit and generation headroom.

The short-trace Inkling 8K and Gekko results above have
`engine_checks_passed: true` and overall `passed: false`, because their paired
independent PyTorch reference-model reports failed. Engine and reference-model evidence must remain separate; successful
kernel tests or selected Engine cases do not establish complete model parity.

The RMSNorm evidence covers sampled finite BF16 inputs, affine and non-affine
operation, grouped reductions, and up to 8,192 channels per group. The native
fixtures include sequence lengths 1, 31, and 127. Both PyTorch 2.11/CUDA 12.8 and
PyTorch 2.13/CUDA 13.0 environments passed the same fixture comparisons on H200.
The test suite also checks partition invariance, noncontiguous and empty inputs,
BF16 offset rounding, and the autograd fallback. Bitwise agreement is scoped to
these checked fixtures and tests.

The TDN implementation preserves native Welford reduction and CUB scan order
across serving partitions by retaining one 256-token partial block. Each request
and converted layer stores 16,450 FP32 values for that scan, independently of its
int64 count. Tests cover whole prefill, token decode, irregular partitions,
snapshot forks, nonzero priors, and stale pool contents. Other shapes use the
mathematical reference recurrence. This removes a verified operator discrepancy;
the complete model remains outside the strict numerical gates for some serving
schedules. No native training extension is imported by the serving implementation.

Gekko's serving SCA uses the shared SM90 FA4 kernel with an explicit softmax
implementation. It preserves sequential local sums, a fused rescale/first add,
descending four-lane reduction and correctly rounded division. Standard full
attention retains its original softmax implementation. Custom kernel cache keys
include the implementation's source hash; tests verify both isolation in one
process and reuse after a process restart. This path currently requires NVIDIA
SM90 and BF16 64-channel Q/K/V heads, with one split and no autograd. The
independent mathematical reference retains its Torch attention implementation.

Gekko also preserves two measured execution choices. Its SM90 BF16
`5120 -> 1536` FFN down projections pad short inference calls to 256 matrix rows,
matching the native full-prefix oracle's arithmetic in the tested cases. This
applies to all Gekko-model layers, including full-attention layers, while
preserving checkpoint parameter names. It increases work for small batches and
uses ordinary linear behavior for other shapes/devices, autograd and autocast.
The Gekko hybrid wrapper sets FA4 prefill and decode to one split. Automatic
split-KV sizing caused batch-dependent rounding and a greedy divergence at
4,095 tokens; the isolated one-split correction removed both. It leaves global
deterministic mode and other model backends unchanged.

Eligible NVIDIA BF16 normalized short convolutions use a Triton kernel that
preserves the native chronological FP32 multiply/add sequence and BF16 output
rounding. Weight softmax remains in Torch. The kernel writes fresh output and
history tensors; CPU, other dtypes, autograd and CUDA autocast retain the
mathematical implementation. The mandatory public fixture contains 72 native
synthetic cases, replayed with contiguous and strided layouts, without a private
checkpoint or native extension dependency.

Serving attention caches each rotary module's FP32 inverse-frequency vector.
The cache is private runtime state, excluded from parameters and state dicts;
device, CUDA stream, or rotary configuration changes rebuild it. It preserves
the original frequency calculation and avoids repeated small host-to-device
copies. The independent reference defaults to uncached execution.

A captured decode graph can continue to read a frequency tensor after eager
prefill switches streams and replaces the active cache entry. Each rotary module
therefore keeps a strong reference to every distinct tensor actually used during
CUDA capture, keyed by tensor identity. Repeated captures of the same tensor
deduplicate; recapturing the same cache key with a new tensor retains both. These
references live until module deletion. They do not change the active cache or
frequency arithmetic, become checkpoint buffers, or undergo module dtype
conversion. The capture query uses the requested CUDA device.

Retained payload is 128 bytes per tensor for a 64-dimensional rotary embedding,
plus Python and allocator overhead. A fixed capture set is bounded; repeated
recapture can accumulate tensors over the module lifetime. Clearing the active
cache or destroying a graph does not release these module-owned references.

Two portable regressions in
[`test_xllm_rope_cache.py`](../../test/registered/unit/models/test_xllm_rope_cache.py)
protect this ownership contract without private checkpoints or native fixtures.
The CPU test checks eager-only churn, identity deduplication, same-key recapture,
explicit-device selection, FP32/state-dict behavior and module destruction. The
CUDA test captures two graphs, switches to eager execution on another stream,
churns the allocator, and requires both replays to remain bitwise equal to eager
outputs. It also verifies that both captured tensors remain alive.

AWM reduces time-contiguous FP32 keys, matching native xLLM's layout before
`logsumexp`. Reducing a strided view changed FP32 normalizers by up to one ULP,
which could change later BF16 memory values and accumulate into long-context
logit differences. The correction made the isolated learned 17-chunk AWM
output, final memory and normalizer exact, and all measured long-context Engine
logits above became bitwise equal. Saved failing controls remain part of the
diagnostic record; no gate or native oracle was changed.

The mandatory SM90 test `test_xllm_awm_golden.py` checks all output, memory,
mask and FP32 normalizer bytes against an independent native capture of three
synthetic 4,096-token chunks. Its checked-in 3.6 KB JSON fixture contains hashes,
samples and native source provenance, with no checkpoint data or external
fixture dependency. The test passed in both Torch environments on H200. An
isolated copy of the old strided reduction fails 14 FP32 normalizer values
(maximum difference `9.54e-7`), even though this short fixture's BF16 output and
memory remain equal. This explicitly protects the reduction-layout correction.

Ordinary GPU CI also runs four mandatory synthetic TDN golden cases in
`test/registered/unit/models/test_xllm_tdn_golden.py`: zero and nonzero priors,
each processed whole or across the 256-token boundary. All four passed in both
environments. The checked-in 7 KiB JSON fixture contains native output hashes,
diagnostic samples, and provenance, with no checkpoint data. These tests need
neither a private native extension nor external fixtures; they are separate
from the 58 tests above that replay the larger external native fixture sets.

Four mandatory SM90 SCA golden cases in `test_xllm_sca_golden.py` also pass:
a 257-token prefix, partitions of 31/225/1 tokens, and suffixes of 31 and one
token. Their checked-in 3.3 KiB JSON fixture was captured independently from
native xattn using deterministic synthetic inputs. It contains output hashes,
diagnostic samples, and provenance, without checkpoint data or a native-extension
dependency during replay. These cases supplement the 95 external-fixture SCA
tests above.

## Historical eager serving performance

A serial comparison on one otherwise idle H200 used the same GPU and CPU
affinity, the same learned Gekko export, an 8,192-token prompt, batch size one,
BF16 eager execution, FA4, 2,048-token prefill chunks and disabled prefix reuse.
Each case had one warmup and three timed repeats. The baseline already included
all arithmetic corrections; the only effective model differences were fused
normalized short convolution and cached rotary inverse frequencies.

| Median measurement | Before performance changes | Optimized eager implementation |
| --- | ---: | ---: |
| Prefill through first token | 294.9 ms | 257.1 ms |
| Decode after first token, 127 tokens | 6.002 s | 5.071 s |
| Per cached decode token | 47.3 ms | 39.9 ms |
| Whole request, 128 output tokens | 6.297 s | 5.328 s |

All measured output token IDs were identical. This is a 15.5% reduction in the
measured decode latency, with unchanged source hashes during each run. The
comparison used a fixed baseline-then-optimized order and includes Engine and
client overhead; it is limited latency evidence, not a general throughput
qualification. The eager backend still performs many small launches and copies
request histories in that historical implementation. The pooled Gekko decode
path described below replaces those ordinary-token copies and enables graphs.
The historical timings above do not measure that path.

## Gekko pooled decode and CUDA graphs

Gekko supports FULL decode graphs with eager prefill on NVIDIA SM90, BF16,
TP1/PP1, FA4, and disabled radix caching. The pooled decode geometry is
1,536 model channels, 32 query heads, eight KV heads, 64-channel key/value heads,
4,096-token chunks, and 32 temporal-normalization groups. Other geometries
retain functional eager decode and reject graph configuration early.

Ordinary decode updates the allocator's existing convolution, partial TDN and
local-attention pools in place. Device request slots, positions and valid
history lengths change between replays; the four retained histories are not
gathered or copied each token. Padded graph rows never update request state.
Capture uses invalid slots, so warmup cannot advance a live request.

The entire batch falls back to eager execution when any real incoming token
completes a 256-token TDN block. This includes every 4,096-token AWM boundary.
Both paths share the same pool representation; boundary output uses the old
memory before committing its update. The backend eligibility hook is checked
even without MLP synchronization. Native TDN scan order, convolution rounding,
SCA softmax, and padded FFN projection arithmetic are preserved.

Normalized convolution taps are cached with parameter-version invalidation.
RMSNorm fuses the BF16-rounded offset gain, and small SM90 BF16 RoPE calls fuse
the native complex arithmetic. Captured rotary and convolution constants keep
their owners alive across eager cache replacement. Updating model weights
requires graph recapture.

Use the Engine graph settings below for Gekko too, with
`max_mamba_cache_size=4` for four concurrent requests. Qualification covers
batches one through four, including three real rows padded to four, request
reordering, cancellation, slot reuse, and graph/eager boundary handoff. The
seven full-attention layers retain single-split FA4. Automatic splitting is a
separate numerical experiment; it is not enabled by these changes.

## Experimental Dense and Inkling decode graphs

Dense and Inkling can use FULL CUDA graphs for decode while prefill remains
eager. The tested configuration is NVIDIA H200, FA4, TP1/PP1, BF16 parameters
and KV cache, disabled radix prefix caching, 2,048-token prefill chunks, and at
most four running requests. Capture sizes are `[1, 2, 4]`; three-request decode
uses padding to the four-request graph. Eager execution remains available.
Gekko uses the same graph configuration with the additional pooled-state and
boundary restrictions described above.

Both the constructor policy and the captured RoPE constant ownership described
above are required. The bounded graph/eager checks exercise batch sizes one
through four, mixed prompt lengths, reorder, completed-slot reuse, and 8,192-
and 130,816-token contexts. Raw-logit diagnostics use synchronous scheduling;
ordinary serving uses overlap scheduling. Graph/eager agreement does not remove
existing Dense/Inkling native-model NLL failures or establish full model parity.
All 208 selected full-vocabulary rows in the final graph/eager comparison match
bit-for-bit. Separate cancellation checks use five concurrent pairs per family
and mode: all 520 selected token IDs and sampled-token logprobs match eager
execution exactly. They verify actual two-to-one graph batch transitions,
terminal draining and subsequent slot reuse. Inkling also passes the unchanged
native gates for those lifecycle positions on checkpoint 0007825; that bounded
result does not resolve its other native failures.

Matched single-request timings on one H200 use the final graph/lifetime code,
the same prompt tokens across families, 128 greedy outputs, one warmup and three
measured repetitions. Cached decode excludes the first output; model startup
and capture are excluded.

| Model | Prompt tokens | Decode ms/token | Decode tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Dense | 8,192 | 2.9523 | 338.7 | 77.8 |
| Inkling | 8,192 | 3.2639 | 306.4 | 102.8 |
| Dense | 130,816 | 6.1262 | 163.2 | 5,601.7 |
| Inkling | 130,816 | 4.1000 | 243.9 | 3,818.9 |

Inkling is 1.49 times faster near 128K; dense remains slightly faster at 8K.
Maximum decode variation is 0.0814%, and output IDs match the eager/earlier graph
controls. These are latency measurements on stage-4 checkpoints Dense 0001051
and Inkling 0000750, with the fixed context and runtime above; they do not
establish general throughput or native numerical qualification.

For a completed Dense or Inkling export, construct the Engine inside your
program's `main()` function with these settings; the complete guarded example
below shows startup and shutdown:

```python
from sglang import Engine

engine = Engine(
    model_path="artifacts/export",
    dtype="bfloat16",
    kv_cache_dtype="auto",  # BF16 with this model dtype.
    tp_size=1,
    pp_size=1,
    attention_backend="fa4",
    disable_radix_cache=True,
    speculative_algorithm=None,
    cuda_graph_config={
        "decode": {"backend": "full", "bs": [1, 2, 4], "max_bs": 4},
        "prefill": {"backend": "disabled"},
    },
    chunked_prefill_size=2048,
    max_running_requests=4,
    mem_fraction_static=0.6,
    disable_overlap_schedule=False,
    allow_auto_truncate=False,
)
```

Call `engine.shutdown()` when finished, as in the complete example below.
Single-request timing uses `max_running_requests=1`; upstream then reduces the
captured sizes to one. Token-ID diagnostics additionally skip tokenizer setup,
which is unnecessary for normal text serving. FULL decode at larger batches,
other attention backends/devices/dtypes, and prefill graphs remain
outside this evidence. The `breakable` and `tc_piecewise` graph modes are rejected.

## Model and configuration contract

All three architectures use `model_type: "xllm_hybrid"` and
`xllm_export_schema_version: 1`.

| Native `model.arch` | Exported architecture | Layer families |
| --- | --- | --- |
| `transformer` | `XllmDenseForCausalLM` | `full` |
| `hybrid_inkling` | `XllmInklingForCausalLM` | `full`, `inkling` |
| `hybrid_gekko` | `XllmGekkoForCausalLM` | `full`, `gekko` |

[`schema.py`](schema.py) retains the saved model configuration under
`native_model_config`. It also writes resolved `full_attention`,
`inkling_attention`, or `gekko_attention` dictionaries and the exact
`layer_kinds` schedule. Family overrides take precedence over shared settings.
Unsupported operators and incomplete inventories fail validation. Combining
Inkling and Gekko layers in one model is not supported.

The implementation is intended to preserve these native invariants:

- Head dimensions are explicit. They are not inferred from hidden size divided
  by head count. Input embeddings and output projection remain separate.
- Affine RMSNorm stores an offset: its effective gain is `1 + weight`.
  Export does not fold this offset into the saved tensor.
- RoPE uses adjacent complex pairs and the saved configuration for each family.
  Full-attention YaRN scaling, correction context, and Q/K amplitude remain
  distinct from converted-layer RoPE. Positions continue across prefill pieces
  and decode steps.
- SwiGLU keeps native gate/up/down projection order. The final normalization
  remains the native Transformer RMSNorm.
- Inkling uses local softmax attention, with the current token included in the
  saved window. Its four raw short convolutions operate on K, V, attention
  output, and feed-forward output in native order, including their residuals.
- Gekko preserves convolution normalization, saved gates and value activation,
  temporal distribution normalization (TDN) priors, and delayed memory updates.
  Its architectural chunk boundaries are independent of scheduler prefill
  boundaries. Persistent `prior_count`, `prior_mean`, and `prior_var` buffers are
  required checkpoint state.

The exporter preserves every weight's values, dtype, and layout. The only name
conversion is legacy dense `layers.i.norm.weight` to
`layers.i.attention.norm.weight`. The serving loader adapts Inkling convolution
shape from `[channels, width]` to `[channels, 1, width]` without changing values.
Runtime parameter conversion to BF16 is separate from export.

## Inventory and export

Run commands from the SGLang repository root. Set `XLLM_CHECKPOINT`, `XLLM_REPO`,
and `XLLM_TOKENIZER` to the source checkpoint, matching native source checkout,
and tokenizer directory. Use an isolated Python environment with a DCP-compatible
PyTorch version and `safetensors`. The native training environment can remain
unchanged.

The checkpoint directory must contain its saved `config.json` and
`sharded_model.tp00` DCP directory. A model-only checkpoint may symlink that DCP
directory. DCP metadata is Python serialized; use trusted source checkpoints.

```bash
python tools/xllm_hybrid/export.py inventory "$XLLM_CHECKPOINT" \
  --output artifacts/inventory.json

python tools/xllm_hybrid/export.py export "$XLLM_CHECKPOINT" \
  --source-repo "$XLLM_REPO" \
  --tokenizer "$XLLM_TOKENIZER" \
  --shard-bytes 268435456 \
  --output artifacts/export
```

`inventory` reads metadata and validates all names, shapes, dtypes, parameter or
buffer roles, chunk coverage, and payload locations without restoring weights.
`export` explicitly restores all model tensors using DCP's planner. It processes
deterministic, sorted groups to bound CPU memory; a single tensor may exceed the
requested shard size. Optimizer state is not requested.

Outputs must be new paths. Export records source repository identity, source and
output hashes, tensor byte hashes, and exporter source hashes in `manifest.json`.
Each saved tensor is reloaded and verified exactly. `native_config.json` retains
the original saved configuration, and an available graft receipt is copied.
`EXPORT_COMPLETE.json`, written last, binds the completed export to its manifest.
A directory without that marker is not a completed export.

Tokenizer files are copied. The exported model and generation configuration use
native BOS `0`, EOS `1`, and no padding token; any generation-config correction is
recorded in the manifest. A native padding sentinel such as `-1` must not be fed
to the embedding. Qualification uses explicit token IDs, independently of chat
template selection.

## Native differential qualification

Use an independently allocated GPU and a separate environment containing the
native model's dependencies, `torch`, `safetensors`, and `wandb`. The tool imports
native source read-only. Dense and Inkling use their native module classes; the
helper's logits are checked against native forward loss and selected/full
projection agreement. Gekko requires a native checkout with the full-prefix
`tools.reference_hybrid_generation._padded_prefix` oracle and
`logits_positions` support.

When the native checkout selects FlashAttention 4 through
`ENABLE_FLASH_ATTENTION_4`, set that variable to match the checkpoint's reference
evaluation environment:

```bash
ENABLE_FLASH_ATTENTION_4=TRUE \
python tools/xllm_hybrid/qualify.py \
  --native-repo "$XLLM_REPO" \
  --checkpoint "$XLLM_CHECKPOINT" \
  --export artifacts/export \
  --output artifacts/native-qualification \
  --lengths 31,127 \
  --decode-steps 3 \
  --wandb-mode offline
```

The source DCP is restored independently and every native tensor is checked
against the export. Parameters execute in BF16; native persistent buffer types
are retained. Checks cover full prefill, irregularly partitioned prefill,
selected logits, per-token prompt NLL, and forced-token decode. Gekko boundary
qualification should additionally include lengths around its saved architectural
chunk size `C`, such as `C-1`, `2*C-1`, and `3*C-1`, with decode crossing those
boundaries. Its native full-prefix oracle can require substantially more memory
and work than short cases.

The strict gates are:

| Measurement | Gate |
| --- | --- |
| Mean absolute per-token NLL difference | `<= 0.005` |
| Maximum absolute per-token NLL difference | `<= 0.08` |
| Selected raw-logit relative L2 difference | `<= 0.01` |
| Compared values | Finite |

Raw-logit mean and maximum errors are diagnostic; the first two gates apply to
NLL. Reports also retain signed NLL differences and error percentiles.

The new output directory contains `report.json`, per-case reports,
`native-proof.json`, the token stream, and `serving_fixture-*.json`. Fixtures
include native target scores and greedy continuations for the Engine qualifier.
The `prompt_logprobs` array contains one value per prompt source position `p`,
scoring token `stream[p + 1]`. Its final entry predicts the first continuation
token. There is no entry for the unconditioned first token. The existing `scores`
list retains selected prompt positions and every forced-decode position.
An unsuccessful parity comparison exits with status 1 after writing its reports
and fixtures. Setup exceptions instead produce `failure.json` with the failing
phase. A failed reference-model comparison does not by itself invalidate the
native oracle; its separate `native_oracle.passed` proof determines whether
serving comparisons can use that case.

Both qualification tools write `source_provenance_start.json` before validation
or model imports. Their reports retain those initial hashes, capture source
hashes again after the checks, and require `source_provenance_unchanged` for an
overall pass. Run qualification against a stable checkout; source edits during
a run leave the numerical evidence available but invalidate qualification.

## Actual Engine qualification

For an interactive experiment on the tested H200 configuration, use the completed
export and the same eager runtime settings:

```python
from sglang import Engine
from transformers import AutoTokenizer


def main():
    model_path = "artifacts/export"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = Engine(
        model_path=model_path,
        dtype="bfloat16",
        tp_size=1,
        attention_backend="fa4",
        disable_radix_cache=True,
        cuda_graph_config={
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
        chunked_prefill_size=2048,
        max_running_requests=2,
        mem_fraction_static=0.6,
    )
    try:
        # Native plain-text formatting: one explicit BOS, without a chat template.
        ids = [0] + tokenizer.encode(
            "Explain matrix multiplication.", add_special_tokens=False
        )
        response = engine.generate(
            input_ids=ids,
            sampling_params={"temperature": 0, "max_new_tokens": 128},
        )
        print(tokenizer.decode(response["output_ids"]))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

This example is an experimental launch configuration, not a claim of general
numerical qualification. The following harness measures that separately.

Use a separate SGLang environment built for this checkout and a compatible CUDA
driver/toolchain. At least two distinct prompt fixtures are required to exercise
batching and reordered requests. Run this command separately from the native
qualifier so an expected numerical failure does not suppress serving evidence:

```bash
python tools/xllm_hybrid/serve_qualify.py \
  --export artifacts/export \
  --fixtures artifacts/native-qualification \
  --output artifacts/engine-qualification \
  --attention-backend fa4 \
  --chunked-prefill-size 64 \
  --wandb-mode offline
```

Available backend choices are `fa3` (default), `fa4`, `flashinfer`, and `triton`;
selection is subject to the installed SGLang backend's GPU and model support.
Selecting a backend does not qualify it. Aligning the full-attention backend with
the native run is useful when investigating numerical differences.
The current Gekko evidence uses `fa4` on H200, with global deterministic
inference disabled. The qualifier's optional
`--enable-deterministic-inference` flag tests a separate SGLang execution policy;
that policy also replaces Torch GEMMs and other operations and failed the
native comparison. Gekko's one-split FA4 correction does not need that flag.

The harness enables `SGLANG_RETURN_ORIGINAL_LOGPROB` before importing SGLang.
For new fixtures, it checks every prompt source position and each forced-decode
position through prompt scoring, without counting selected positions twice.
An additional `original-prompt-*` check submits only the original prompt. It
combines the first `L-1` target input logprobs with the final target's output
logprob to cover all `L` prompt logits at the native prompt's original input
length. This check is enabled only for fixtures containing `prompt_logprobs`.
It also checks selected targets through the output scoring API, then
compares actual greedy generation for single, batched, reordered, differently
sized, and repeated requests. It checks token/score alignment explicitly and
does not compare later conditional probabilities after greedy tokens diverge.
The NLL mean and maximum gates apply separately to prompt and forced-decode
segments, preventing one segment from diluting the other's mean error. Gates
may be tightened, but cannot be relaxed through CLI arguments.

The native prompt scores come from a full-prefix call of length `L`; the Engine
prompt-scoring request contains that prompt plus the forced continuation. Native
forced-decode scores come from separate, growing full prefixes. Although these
calls share the same causal conditioning at each compared position, BF16 results
can change with sequence length and execution shape. Native-only controls have
also shown such prefix differences. Prompt-scoring errors therefore measure both
port differences and sensitivity to evaluation shape. The original-prompt check
reduces input-length differences; scheduler chunking and kernel execution can
still differ. A same-shape native control can help diagnose a failure; this
caveat does not change the thresholds,
remove any positions, or turn a failed qualification into a pass. Actual greedy
decode remains a separate comparison of matching conditional token streams.

`report.json` separates `engine_checks_passed`,
`native_reference_report_passed`, and `native_reference_selected_logits`.
Engine checks still run when native/reference parity failed, provided the native
oracle and fixture identities are valid. Overall `passed` requires both native
and Engine checks to pass. The Engine API returns normalized logprobs, so this
harness does not independently measure Engine raw-logit relative L2; the paired
reference measurement is identified separately.

Older fixtures without `prompt_logprobs` remain usable. Their comparisons are
explicitly labeled `scores_scope: "selected_prompt_and_forced_decode"`; new
fixtures use `"full_prompt_and_forced_decode"`. A report with any older fixture
sets `full_prompt_nll_evaluated` to false and lists full-token Engine prompt NLL
as unqualified. A pass for sparse checks establishes only their recorded scope.
Mixed fixture sets report each fixture's scope separately. Regenerate fixtures
with the native qualifier to obtain full prompt coverage.

Reports and response artifacts are written before cleanup. The W&B run is
finished and its service stopped with the public `wandb.teardown()` API before
Engine shutdown, which may terminate child processes. Cleanup exceptions are
propagated, and Engine shutdown is attempted even if W&B cleanup fails. Offline
mode keeps tracking artifacts local; timing fields are diagnostic, not benchmarks.

## Runtime restrictions and remaining coverage

The current serving adapters require TP1, PP1, BF16 parameters and KV cache,
and disabled radix prefix caching. Dense and Inkling accept eager execution or
FULL decode graphs with prefill graphs disabled; the measured graph scope is
limited to the FA4 configuration and batches of at most four described above.
Gekko supports FULL decode at its pooled-kernel geometry with eager boundary
fallback, as described above. Quantization, LoRA,
speculative decoding, and disaggregated serving are rejected. Attention families
must have uniform serving head shapes, and an Inkling window must be at least
two tokens. The Engine qualifier supplies the required runtime settings.

Full attention and Inkling local attention use SGLang's paged attention path.
Inkling convolution history and Gekko persistent state use request-owned pool
slots. Gekko uses pooled ordinary decode and the functional recurrence for
prefill and state-update boundaries. Its native arithmetic remains explicit
in the RMSNorm, TDN, SCA, short-convolution and rotary kernels.

Cancellation and selected raw-logit evidence are scoped to the cases above;
they do not establish every lifecycle behavior or input. The forced-retraction
probe uses SGLang's test retraction policy; it does not qualify every natural
memory-pressure schedule. Prefix reuse, prefill graphs, larger
decode-graph batches, multi-GPU parallelism and general long-context capacity
still need separate qualification. Dense and
Inkling learned-checkpoint numerical failures remain outside the strict gates.
The near-128K raw-logit test used prefixes of 131,060 through 131,063 tokens,
within the Engine's reserved generation headroom; it did not increase the model's
131,072-token context or truncate inputs.

## CPU tests

With `torch`, `safetensors`, and `pytest` installed, run:

```bash
python -m unittest discover -s tools/xllm_hybrid -p 'test_*.py' -v

python -m pytest -q \
  test/registered/unit/models/test_xllm_reference.py \
  test/registered/unit/models/test_xllm_gekko_reference.py \
  test/registered/unit/models/test_xllm_gekko_backend.py \
  test/registered/unit/models/test_xllm_hybrid.py \
  test/registered/unit/models/test_xllm_gemm.py \
  test/registered/unit/models/test_xllm_fa4_custom_softmax.py \
  test/registered/unit/models/test_xllm_rope_cache.py \
  test/registered/unit/models/test_xllm_shortconv_dispatch.py \
  test/registered/unit/models/test_xllm_serve_qualify.py \
  -k 'not test_device_and_cuda_stream_switch_rebuild_safely and not test_capture_constants_survive_stream_switch_recapture_and_allocator_churn'
```

These tests cover strict DCP export and loading, native tensor naming, reference
operators, chunk/decode state, adapter wiring with CPU stand-ins, and
qualification/report alignment. A native-source comparison may skip when its
optional native checkout is unavailable. CPU stand-ins do not qualify CUDA
kernels or the SGLang scheduler. Learned-checkpoint GPU comparisons and actual
Engine checks remain necessary before making numerical or performance claims.
