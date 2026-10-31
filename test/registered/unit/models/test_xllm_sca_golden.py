"""Public native SCA output hashes without checkpoint or native dependencies.

The fixture contains independently captured SM90 BF16 outputs for synthetic
257-token Q32/KV8/D64 inputs. Hashes cover every output value; selected samples
help explain failures. Input construction uses exact integers and powers of two.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_xllm_sca_golden_recipe", Path(__file__).with_name("test_xllm_sca.py")
)
_recipe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _recipe
_SPEC.loader.exec_module(_recipe)
register_cuda_ci = _recipe._ci.register_cuda_ci
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")
pytestmark = _recipe.pytestmark
_FIXTURE = Path(__file__).with_name("fixtures") / "xllm_sca_native_golden.json"


def _digest(value):
    return hashlib.sha256(
        value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "case", ["whole", "partitions", "suffix31", "suffix1", "pooled_suffix1"]
)
@torch.no_grad()
def test_public_native_sca_golden(case):
    fixture = json.loads(
        Path(os.environ.get("XLLM_SCA_GOLDEN_FIXTURE", _FIXTURE)).read_text()
    )
    assert fixture["format_version"] == fixture["recipe_version"] == 1
    assert fixture["parameters"] == {"chunk_size": 4096, "softmax_scale": 0.125}
    inputs = dict(zip(("q", "k", "v"), _recipe.make_synthetic_inputs(), strict=True))
    for name, value in inputs.items():
        expected = fixture["inputs"][name]
        assert expected["shape"] == list(value.shape)
        assert expected["dtype"] == str(value.dtype)
        assert _digest(value) == expected["sha256"], f"Synthetic recipe changed: {name}"
    sys.path.insert(0, str(_recipe._ROOT / "python"))
    sca = importlib.import_module("sglang.srt.models.xllm_sca")
    q, k, v = (inputs[name].cuda() for name in ("q", "k", "v"))
    if case == "partitions":
        outputs, start = [], 0
        for count in (31, 225, 1):
            end = start + count
            outputs.append(sca.native_sca_attention(q[start:end], k[:end], v[:end]))
            start = end
        actual = torch.cat(outputs)
        key = "whole"
    elif case == "pooled_suffix1":
        k_pool = torch.full((1, 8192, 8, 64), torch.nan, dtype=k.dtype, device=k.device)
        v_pool = torch.full_like(k_pool, torch.nan)
        k_pool[0, : len(k)].copy_(k)
        v_pool[0, : len(v)].copy_(v)
        actual = sca.native_sca_decode(
            q[-1:],
            k_pool,
            v_pool,
            torch.tensor([0], dtype=torch.int32, device=q.device),
            torch.tensor([len(k)], dtype=torch.int32, device=q.device),
        )
        key = "suffix1"
    else:
        count = {"whole": 257, "suffix31": 31, "suffix1": 1}[case]
        actual = sca.native_sca_attention(q[-count:], k, v)
        key = case
    expected = fixture["outputs"][key]
    samples = {
        str(index): actual.flatten()[index].item()
        for index in expected["sample_indices"]
    }
    assert expected["shape"] == list(actual.shape)
    assert expected["dtype"] == str(actual.dtype)
    assert _digest(actual) == expected["sha256"], (
        f"Native SCA {case} differs: expected samples {expected['samples']}, actual {samples}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
