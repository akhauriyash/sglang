"""Ragged batching of the custom FA4 softmax, without changing global caches.

The generic API batches independent sequences with the selected arithmetic.
This does not claim equivalence to native training's segmented/varlen routing,
which can select different exponent arithmetic. Optional native outputs below
come from independent fixed-sequence calls, then are assembled into one batch.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_fa4_custom_cuda_ci", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Custom xLLM SCA arithmetic requires NVIDIA SM90",
)


@cache
def _modules():
    sys.path.insert(0, str(_ROOT / "python"))
    return (
        importlib.import_module("sglang.srt.models.xllm_sca"),
        importlib.import_module(
            "sglang.kernels.ops.attention.flash_attn.cute.interface"
        ),
    )


def _ragged_attention(requests):
    sca, interface = _modules()
    q, k, v = (
        torch.cat([request[index] for request in requests]) for index in range(3)
    )
    q_lengths = [len(request[0]) for request in requests]
    k_lengths = [len(request[1]) for request in requests]
    assert q_lengths == [31, 31]
    cu_q = torch.tensor([0, 31, 62], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)], dtype=torch.int32, device=k.device
    )
    output, lse = interface._flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        softmax_scale=0.125,
        causal=True,
        num_splits=1,
        return_lse=False,
        softmax_impl=sca.NativeXllmSoftmax,
    )
    assert lse is None
    return output


def _assert_bitwise(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    # Integer views distinguish signed zeros as well as ordinary BF16 values.
    torch.testing.assert_close(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("reverse", [False, True])
@torch.no_grad()
def test_ragged_custom_matches_separate_fixed_requests(reverse):
    """Mandatory synthetic prefix31 plus suffix31/K257 batching control."""
    sca, _ = _modules()
    generator = torch.Generator(device="cuda").manual_seed(1701)
    requests = []
    for key_length in (31, 257):
        requests.append(
            tuple(
                torch.randn(
                    (length, heads, 64),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )
                for length, heads in ((31, 32), (key_length, 8), (key_length, 8))
            )
        )
    if reverse:
        requests.reverse()
    expected = torch.cat([sca.native_sca_attention(*request) for request in requests])
    _assert_bitwise(_ragged_attention(requests), expected)
    # Warm reuse of the same cache must preserve offsets and request isolation.
    _assert_bitwise(_ragged_attention(requests), expected)


def _load_native_fixtures(record_property):
    raw_paths = os.environ.get("XLLM_SCA_RAGGED_ORACLE_FIXTURES")
    if raw_paths is None:
        pytest.skip(
            "Set XLLM_SCA_RAGGED_ORACLE_FIXTURES to JSON [layer5_dir, layer6_dir]"
        )
    paths = json.loads(raw_paths)
    assert isinstance(paths, list) and len(paths) == 2
    from safetensors.torch import load_file

    fixtures = []
    for layer, raw_path in zip((5, 6), paths, strict=True):
        path = Path(raw_path)
        metadata_file = path / "native_sca_inputs.json"
        tensor_file = path / "native_sca_inputs.safetensors"
        metadata = json.loads(metadata_file.read_text())
        assert metadata["layer"] == layer
        assert metadata["causal"] is True and metadata["q_k_are_rotated"] is True
        assert metadata["softmax_scale"] == 0.125
        assert metadata.get("previous_key_length", 0) == 0
        tensor_hash = hashlib.sha256(tensor_file.read_bytes()).hexdigest()
        if "sha256" in metadata:
            assert tensor_hash == metadata["sha256"]
        record_property(f"native_layer_{layer}_tensor_sha256", tensor_hash)
        record_property(
            f"native_layer_{layer}_metadata_sha256",
            hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
        )
        tensors = load_file(tensor_file, device="cuda")
        assert set(tensors) == {"q", "k", "v", "expected"}
        assert tensors["q"].shape == tensors["expected"].shape == (4096, 32, 64)
        assert tensors["k"].shape == tensors["v"].shape == (4096, 8, 64)
        assert all(
            value.dtype == torch.bfloat16 and torch.isfinite(value).all()
            for value in tensors.values()
        )
        fixtures.append(tensors)
    return fixtures


@pytest.mark.parametrize("reverse", [False, True])
@torch.no_grad()
def test_ragged_custom_matches_independent_native_outputs(reverse, record_property):
    """Optional independent prefix31 layer5 and suffix31/K4096 layer6 oracle."""
    prefix, suffix = _load_native_fixtures(record_property)
    requests = [
        (prefix["q"][:31], prefix["k"][:31], prefix["v"][:31]),
        (suffix["q"][-31:], suffix["k"], suffix["v"]),
    ]
    expected = [prefix["expected"][:31], suffix["expected"][-31:]]
    if reverse:
        requests.reverse()
        expected.reverse()
    _assert_bitwise(_ragged_attention(requests), torch.cat(expected))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
