"""Native Torch arithmetic, strided inputs and dynamic RoPE graph replay."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from sglang.srt.models.xllm_reference import NativeRotaryEmbedding
from sglang.srt.models.xllm_rope import native_rope_decode

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "_xllm_rope_ci", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ci
_SPEC.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native fused RoPE requires NVIDIA SM90",
)

CONFIGS = [
    (64, 500000.0, None),
    (
        64,
        1000000.0,
        {
            "rope_type": "yarn",
            "factor": 16.0,
            "beta_fast": 128.0,
            "beta_slow": 4.0,
            "attention_factor": 1.2772588722239782,
        },
    ),
    (32, 10000.0, None),
    (
        32,
        1000000.0,
        {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096},
    ),
    (64, 10000.0, {"rope_type": "yarn", "factor": 1.0, "attention_factor": 0.75}),
    (0, 500000.0, None),
]


def module(config):
    dim, base, scaling = config
    return NativeRotaryEmbedding(
        64, dim, base, scaling, cache_inverse_frequencies=True
    ).cuda()


def candidate(rope, q, k, positions):
    return native_rope_decode(
        q,
        k,
        positions,
        rope.inverse_frequencies(q.device),
        rope.attention_factor,
    )


def torch_oracle(rope, q, k, positions):
    """Original xLLM Torch polar/complex formula, independent of dispatch."""
    dim = rope.rope_head_dim
    if not dim:
        return q, k
    phases = torch.outer(positions.float(), rope.inverse_frequencies(q.device))
    frequencies = torch.polar(torch.ones_like(phases) * rope.attention_factor, phases)
    outputs = []
    for value in (q, k):
        pairs = torch.view_as_complex(
            value[..., :dim].float().reshape(*value.shape[:-1], -1, 2)
        )
        rotated = (
            torch.view_as_real(pairs * frequencies[:, None, :])
            .flatten(-2)
            .to(value.dtype)
        )
        outputs.append(
            rotated
            if dim == rope.head_dim
            else torch.cat((rotated, value[..., dim:]), -1)
        )
    return tuple(outputs)


def exact(actual, expected):
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            observed.contiguous().view(torch.int16),
            oracle.contiguous().view(torch.int16),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("config", CONFIGS)
@torch.no_grad()
def test_every_supported_position(config):
    torch.manual_seed(917)
    positions = torch.arange(131072, device="cuda", dtype=torch.int64)
    q = torch.randn((len(positions), 2, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((len(positions), 1, 64), device="cuda", dtype=torch.bfloat16)
    rope = module(config)
    exact(candidate(rope, q, k, positions), torch_oracle(rope, q, k, positions))


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("config", CONFIGS)
@torch.no_grad()
def test_decode_shapes_strides_and_replayed_positions(batch, config):
    torch.manual_seed(712 + batch)
    q = torch.randn((batch, 64, 128), device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
    k = torch.randn((batch, 16, 128), device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
    positions = torch.zeros(batch, device="cuda", dtype=torch.int32)
    rope = module(config)
    if not config[0]:
        actual = candidate(rope, q, k, positions)
        assert actual[0] is q and actual[1] is k
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rope(q, k, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = rope(q, k, positions)
    # Moving inverse-frequency ownership back to the eager stream is deliberate:
    # existing native cache lifetime handling must keep captured pointers alive.
    for start in [
        0,
        1,
        254,
        255,
        256,
        4094,
        4095,
        4096,
        8190,
        8191,
        8192,
        32766,
        65534,
        130815,
        131068,
    ]:
        positions.copy_(
            torch.arange(start, start + batch, device="cuda", dtype=torch.int32)
        )
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        expected = torch_oracle(rope, q, k, positions)
        exact(candidate(rope, q, k, positions), expected)
        exact(rope(q, k, positions), expected)
        graph.replay()
        exact(captured, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
