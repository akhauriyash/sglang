# Gekko and Inkling support in SGLang: agent handoff

This branch is an **experimental xLLM hybrid-attention port**. It adds SGLang
adapters for native-format dense, Inkling, and Gekko checkpoints, including
interleaved full-attention layers. It is not part of upstream SGLang, and this
repository does not contain model weights or the native xLLM training code.

The source branch is
[`akhauriyash/sglang:feat/gekko-inkling-sglang-handoff-20261031`](https://github.com/akhauriyash/sglang/tree/feat/gekko-inkling-sglang-handoff-20261031).
Start with this file, then read [`tools/xllm_hybrid/README.md`](tools/xllm_hybrid/README.md)
for the model contract, numerical evidence, CUDA graph details, and test commands.
Keep the branch pinned to a commit while validating it; installing a released
`sglang` wheel alone will not provide these adapters.

## 1. Set up the serving checkout

Use an NVIDIA H200 or comparable SM90 environment with a compatible CUDA,
PyTorch, SGLang kernel, and FlashAttention 4 stack. The measured Gekko path uses
H200, FA4, BF16, and one GPU. Follow this checkout's
[source-install instructions](docs/docs/get-started/install.mdx) for the target
machine; from a fresh environment, the basic steps are:

```bash
git clone --branch feat/gekko-inkling-sglang-handoff-20261031 \
  --single-branch https://github.com/akhauriyash/sglang.git
cd sglang
python -m pip install -e "python"
python -c 'import sglang; print(sglang.__file__)'
```

Confirm that `sglang.__file__` resolves inside this checkout. Install/verify
the CUDA kernels and FA4 required by your chosen SGLang environment before
launching a model. Do not silently substitute a PyPI SGLang checkout or another
attention backend when interpreting the numerical results below.

## 2. Export a native checkpoint

Supply a **trusted** matching native xLLM source checkout, a tokenizer, and a
TP1 PyTorch Distributed Checkpoint (DCP) directory. The checkpoint directory
must contain `config.json` and `sharded_model.tp00/`. Its saved `model.arch`
should be `hybrid_gekko` for Gekko or `hybrid_inkling` for Inkling. The exporter
uses DCP metadata that is Python serialized, so do not load untrusted sources.

Run from the SGLang repository root. Use new output paths:

```bash
export XLLM_CHECKPOINT=/path/to/native/checkpoint
export XLLM_REPO=/path/to/matching/native/xllm/source
export XLLM_TOKENIZER=/path/to/matching/tokenizer
export XLLM_WORKDIR=/path/to/new/workdir
mkdir -p "$XLLM_WORKDIR"

python tools/xllm_hybrid/export.py inventory "$XLLM_CHECKPOINT" \
  --output "$XLLM_WORKDIR/inventory.json"
python tools/xllm_hybrid/export.py export "$XLLM_CHECKPOINT" \
  --source-repo "$XLLM_REPO" \
  --tokenizer "$XLLM_TOKENIZER" \
  --shard-bytes 268435456 \
  --output "$XLLM_WORKDIR/export"

test -s "$XLLM_WORKDIR/export/EXPORT_COMPLETE.json"
```

The exporter validates the complete tensor inventory and exact tensor-byte
round trips, then writes the SGLang `config.json`, safetensors, tokenizer, and
source/hash manifest. `EXPORT_COMPLETE.json` is the completion receipt. Do not
point SGLang directly at the DCP checkpoint or assume a partially written
export is usable. The output config should name `XllmGekkoForCausalLM` or
`XllmInklingForCausalLM` and retain the exact `layer_kinds` schedule.

## 3. Smoke-test Gekko through the actual Engine

First use eager decode; it is simpler for diagnosing installation and export
problems. The native tokenizer uses one explicit BOS token (`0`), and the
following example does not apply a chat template:

```bash
export XLLM_EXPORT="$XLLM_WORKDIR/export"
python - <<'PY'
import os

from sglang import Engine
from transformers import AutoTokenizer

model_path = os.environ["XLLM_EXPORT"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
engine = Engine(
    model_path=model_path,
    dtype="bfloat16",
    kv_cache_dtype="auto",
    tp_size=1,
    pp_size=1,
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
PY
```

After eager serving works, the qualified **bounded** Gekko graph configuration
changes only the Engine settings below. It is for the documented 1,536-channel,
32-query/8-KV-head, head-width-64, 4,096-token-chunk, 32-TDN-group geometry.
Ordinary decode uses a full CUDA graph; a 256-token TDN boundary falls back to
eager. Prefill stays eager.

```python
cuda_graph_config={
    "decode": {"backend": "full", "bs": [1, 2, 4], "max_bs": 4},
    "prefill": {"backend": "disabled"},
},
max_mamba_cache_size=4,
max_running_requests=4,
```

Use those entries as `Engine(...)` keyword arguments, retaining the other
smoke-test settings. For a different Gekko geometry, keep eager execution until
its pooled decode path is implemented and qualified.

## 4. Validate against the matching native model

Generation is a smoke test, not numerical qualification. The native qualifier
creates oracle fixtures, and the SGLang qualifier compares actual Engine
prefill, forced decode, greedy decode, batching, and request reuse:

```bash
ENABLE_FLASH_ATTENTION_4=TRUE \
python tools/xllm_hybrid/qualify.py \
  --native-repo "$XLLM_REPO" \
  --checkpoint "$XLLM_CHECKPOINT" \
  --export "$XLLM_EXPORT" \
  --output "$XLLM_WORKDIR/native-qualification" \
  --lengths 31,127 \
  --decode-steps 3 \
  --wandb-mode offline

python tools/xllm_hybrid/serve_qualify.py \
  --export "$XLLM_EXPORT" \
  --fixtures "$XLLM_WORKDIR/native-qualification" \
  --output "$XLLM_WORKDIR/engine-qualification" \
  --attention-backend fa4 \
  --chunked-prefill-size 64 \
  --wandb-mode offline
```

Run these in their respective compatible native and SGLang Python environments
if necessary. Gekko boundary checks should include lengths near `C`, `2*C`,
and `3*C`, where `C` is the saved chunk size. The detailed report explains why
an independent reference-model failure and an Engine failure are separate
findings. See the [qualification guide](tools/xllm_hybrid/README.md#native-differential-qualification)
before interpreting a pass or failure.

## Current scope and useful starting points

- `python/sglang/srt/models/xllm_hybrid.py`: serving adapter and model loading.
- `python/sglang/srt/configs/xllm_hybrid.py`: export/config contract and
  mixed-layer registration.
- `python/sglang/srt/layers/attention/linear/xllm_gekko_backend.py`:
  request-owned Gekko state and graph/eager decode dispatch.
- `tools/xllm_hybrid/`: strict DCP export, qualification tools, and detailed
  results. The checked-in fixtures are small synthetic tests, not model weights.

This branch currently requires TP1/PP1, BF16 parameters and KV cache, disabled
radix prefix caching, and a supported attention backend. Quantization, LoRA,
speculative decode, and disaggregated serving are not qualified. Dense and
Inkling retain some learned-checkpoint numerical failures under the strict
qualification gates. Gekko has strong bounded H200/FA4 evidence in the detailed
README, but it is not a general production or upstream-PR qualification.
