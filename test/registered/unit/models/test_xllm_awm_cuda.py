# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional learned AWM oracle across many architectural chunk boundaries.

Set XLLM_AWM_ORACLE_FIXTURE to an immutable native operator capture directory.
The public synthetic oracle in test_xllm_awm_golden.py runs without a checkpoint.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gekko = load(
    "_xllm_awm_cuda", _ROOT / "python/sglang/srt/models/xllm_gekko_reference.py"
)
ci = load("_xllm_awm_cuda_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cuda_ci = ci.register_cuda_ci
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native BF16 AWM arithmetic requires NVIDIA CUDA",
)


@torch.no_grad()
def test_learned_long_awm_matches_native_output_and_persistent_state():
    from safetensors.torch import load_file

    directory = os.environ.get("XLLM_AWM_ORACLE_FIXTURE")
    if directory is None:
        pytest.skip("XLLM_AWM_ORACLE_FIXTURE is not set")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    tensor_path = directory / "awm21.safetensors"
    with tensor_path.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == manifest["tensor_sha256"]
        )
    assert manifest["native_selected_repeat_exact"]
    assert manifest["sources_unchanged"]
    saved = load_file(tensor_path, device="cuda")
    chunk, heads, kv_heads, dim = (
        manifest[name]
        for name in ("chunk_size", "num_heads", "num_kv_heads", "head_dim")
    )
    length = len(saved["q"])
    assert length % chunk == 0 and length >= 3 * chunk
    q = saved["q"].view(length, heads, dim)
    qk = saved["qk"].view(length, kv_heads, dim)
    k = saved["k"].view(length, kv_heads, dim)
    v = saved["v"].view(length, kv_heads, dim)
    memory = v.new_zeros(kv_heads, dim, dim)
    log_normalizer = torch.full((kv_heads, dim), -torch.inf, device="cuda")
    for begin in range(0, length, chunk):
        actual = torch.einsum(
            "thd,hdv->thv",
            q[begin : begin + chunk],
            memory.repeat_interleave(heads // kv_heads, dim=0),
        ).flatten(1)
        torch.testing.assert_close(
            actual, saved["expected"][begin : begin + chunk], rtol=0, atol=0
        )
        if begin:
            previous = slice(begin - chunk, begin)
            memory, log_normalizer = gekko.adaptive_working_memory_update(
                memory, log_normalizer, qk[previous], k[previous], v[previous]
            )
    torch.testing.assert_close(memory, saved["memory"], rtol=0, atol=0)
    torch.testing.assert_close(log_normalizer, saved["log_normalizer"], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
