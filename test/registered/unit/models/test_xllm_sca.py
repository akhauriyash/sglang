"""Scoped SCA native-oracle, causal suffix, and compiled-cache regressions."""

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native xLLM SCA requires NVIDIA SM90",
)
_FIXTURES = json.loads(os.environ.get("XLLM_SCA_ORACLE_FIXTURES", "[]"))
_DEFAULT_CONTROLS = json.loads(os.environ.get("XLLM_SCA_DEFAULT_CONTROLS", "[]"))


@cache
def _modules():
    sys.path.insert(0, str(_ROOT / "python"))
    return (
        importlib.import_module("sglang.srt.models.xllm_sca"),
        importlib.import_module(
            "sglang.kernels.ops.attention.flash_attn.cute.interface"
        ),
    )


@cache
def _fixture(path):
    if path is None:
        pytest.skip("Set XLLM_SCA_ORACLE_FIXTURES for independent native SCA tensors")
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads((path / "native_sca_inputs.json").read_text())
    assert metadata["causal"] and metadata["q_k_are_rotated"]
    if "sha256" in metadata:
        with (path / "native_sca_inputs.safetensors").open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest() == metadata["sha256"]
            )
    tensors = load_file(path / "native_sca_inputs.safetensors", device="cuda")
    assert set(tensors) == {"q", "k", "v", "expected"}
    assert len(tensors["q"]) <= len(tensors["k"])
    return metadata, tensors


def _strided(value, dimension=0):
    shape = list(value.shape)
    shape[dimension] *= 2
    storage = value.new_full(shape, torch.nan)
    slices = [slice(None)] * value.ndim
    slices[dimension] = slice(None, None, 2)
    view = storage[tuple(slices)]
    view.copy_(value)
    return view


@pytest.mark.parametrize("path", _FIXTURES or [None])
@pytest.mark.parametrize(
    "case", ["whole", "prefix", "decode", "suffix31", "middle", "strided", "partitions"]
)
@torch.no_grad()
def test_saved_native_sca_exact(path, case):
    sca, _ = _modules()
    metadata, tensors = _fixture(path)
    q, k, v, expected = (tensors[key] for key in ("q", "k", "v", "expected"))
    history = len(k) - len(q)
    if case == "partitions":
        outputs, start = [], 0
        for size in (31, 225, 257, len(q)):
            end = min(start + size, len(q))
            if start == end:
                break
            outputs.append(
                sca.native_sca_attention(
                    q[start:end],
                    k[: history + end],
                    v[: history + end],
                    metadata["softmax_scale"],
                )
            )
            start = end
        actual = torch.cat(outputs)
    else:
        start, end = 0, len(q)
        if case == "prefix":
            end = min(metadata.get("real_length", 31), len(q))
        elif case == "decode":
            start = end - 1
        elif case == "suffix31":
            start = end - 31
        elif case in {"middle", "strided"}:
            end = min(257, end)
        q, k, v = q[start:end], k[: history + end], v[: history + end]
        if case == "strided":
            q, k, v = (_strided(value) for value in (q, k, v))
        actual = sca.native_sca_attention(q, k, v, metadata["softmax_scale"])
        expected = expected[start:end]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("path", _FIXTURES or [None])
@torch.no_grad()
def test_saved_native_pooled_decode_exact(path):
    sca, _ = _modules()
    metadata, tensors = _fixture(path)
    q, k, v, expected = (tensors[key] for key in ("q", "k", "v", "expected"))
    assert len(k) <= 8192
    k_pool = torch.full((3, 8192, 8, 64), torch.nan, dtype=k.dtype, device=k.device)
    v_pool = torch.full_like(k_pool, torch.nan)
    k_pool[2, : len(k)].copy_(k)
    v_pool[2, : len(v)].copy_(v)
    actual = sca.native_sca_decode(
        torch.cat((q[-1:], torch.full_like(q[-1:], torch.nan))),
        k_pool,
        v_pool,
        torch.tensor([2, -1], dtype=torch.int32, device=q.device),
        torch.tensor([len(k), 0], dtype=torch.int32, device=q.device),
        metadata["softmax_scale"],
    )
    torch.testing.assert_close(actual[0], expected[-1], rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]) == 0


def make_synthetic_inputs(device="cpu"):
    """Version 1 deterministic BF16 recipe; native golden capture shares only this."""

    def values(heads, factor):
        indices = torch.arange(257 * heads * 64, dtype=torch.int64)
        return (
            (((indices * factor) % 127 - 63).float() / 64)
            .reshape(257, heads, 64)
            .bfloat16()
            .to(device)
        )

    return values(32, 17), values(8, 23), values(8, 11)


def _synthetic_inputs():
    return make_synthetic_inputs(device="cuda")


def _default_attention(interface, q, k, v):
    output, _ = interface._flash_attn_fwd(
        q=q[None],
        k=k[None],
        v=v[None],
        softmax_scale=0.125,
        causal=True,
        num_splits=1,
        return_lse=False,
    )
    return output[0]


@torch.no_grad()
def test_synthetic_causal_suffix_matches_dense_math_and_strides():
    sca, _ = _modules()
    q, k, v = _synthetic_inputs()
    q = q[-31:]
    scores = (
        torch.einsum("thd,shd->hts", q.float(), k.repeat_interleave(4, 1).float()) / 8
    )
    query_positions = torch.arange(257 - len(q), 257, device=q.device)
    key_positions = torch.arange(257, device=q.device)
    scores.masked_fill_(key_positions[None] > query_positions[:, None], -torch.inf)
    expected = torch.einsum(
        "hts,shd->thd", scores.softmax(-1), v.repeat_interleave(4, 1).float()
    ).bfloat16()
    actual = sca.native_sca_attention(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
    for dimension in (0, 1, 2):
        strided = sca.native_sca_attention(
            *(_strided(value, dimension) for value in (q, k, v))
        )
        torch.testing.assert_close(strided, actual, rtol=0, atol=0)


@torch.no_grad()
def test_custom_and_default_equal_shapes_have_separate_cache_entries(monkeypatch):
    sca, interface = _modules()
    q, k, v = _synthetic_inputs()
    softmax = importlib.import_module(
        "sglang.kernels.ops.attention.flash_attn.cute.softmax"
    )
    original_update = softmax.Softmax._update_row_sum
    original_finalize = softmax.Softmax.finalize
    from sglang.kernels.jit.cute_aot_cache import JITCache

    # A persistent cache's clear() also removes disk objects. This test owns
    # a memory-only cache and restores the runtime's cache after the check.
    cache_object = JITCache()
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", cache_object)
    default_first = _default_attention(interface, q, k, v)
    default_keys = set(cache_object.cache)
    custom_first = sca.native_sca_attention(q, k, v)
    custom_keys = set(cache_object.cache) - default_keys
    assert len(default_keys) == len(custom_keys) == 1
    assert (
        cache_object.cache[next(iter(default_keys))]
        is not cache_object.cache[next(iter(custom_keys))]
    )
    for _ in range(2):
        torch.testing.assert_close(
            _default_attention(interface, q, k, v), default_first, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sca.native_sca_attention(q, k, v), custom_first, rtol=0, atol=0
        )
    assert len(cache_object.cache) == 2
    assert softmax.Softmax._update_row_sum is original_update
    assert softmax.Softmax.finalize is original_finalize


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@torch.no_grad()
def test_pooled_decode_exact_live_prefix_and_graph_replay(batch_size):
    """Device metadata changes under replay; stale tails and padding are unread."""
    sca, _ = _modules()
    generator = torch.Generator(device="cuda").manual_seed(713 + batch_size)
    q = torch.randn(
        (batch_size, 32, 64),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # Separate source/pool storage lets every replay poison the entire unused
    # tail without changing live values. Request slots are deliberately shuffled.
    shape = (batch_size + 2, 8192, 8, 64)
    source_k = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    source_v = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    k_pool = torch.full_like(source_k, torch.nan)
    v_pool = torch.full_like(source_v, torch.nan)
    slots = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(slots)

    def prepare(case, phase):
        host_lengths = [case[row % len(case)] for row in range(batch_size)]
        # Ensure unique slots even for batch 4 and odd/even slot permutations.
        host_slots = list(range(len(k_pool)))[phase % 2 : phase % 2 + batch_size][::-1]
        k_pool.fill_(torch.nan)
        v_pool.fill_(torch.nan)
        expected = torch.zeros_like(q)
        for row, (slot, length) in enumerate(zip(host_slots, host_lengths)):
            if length:
                k_pool[slot, :length].copy_(source_k[slot, :length])
                v_pool[slot, :length].copy_(source_v[slot, :length])
                expected[row] = sca.native_sca_attention(
                    q[row : row + 1],
                    source_k[slot, :length],
                    source_v[slot, :length],
                )[0]
            else:
                host_slots[row] = -1
        slots.copy_(torch.tensor(host_slots, dtype=torch.int32, device="cuda"))
        lengths.copy_(torch.tensor(host_lengths, dtype=torch.int32, device="cuda"))
        return expected

    prepare([257], 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sca.native_sca_decode(q, k_pool, v_pool, slots, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = sca.native_sca_decode(q, k_pool, v_pool, slots, lengths)
    cases = [
        [0],
        [1, 0, 127, 128],
        [127, 128, 129, 0],
        [129, 255, 256, 257],
        [255, 256, 257, 4095],
        [256, 257, 4095, 4096],
        [4095, 4096, 4097, 8191],
        [4096, 4097, 8191, 8192],
        [4097, 8191, 8192, 0],
        [8191, 8192, 0, 1],
        [8192, 1, 257, 4097],
        [1],
    ]
    for phase, case in enumerate(cases):
        expected = prepare(case, phase)
        eager = sca.native_sca_decode(q, k_pool, v_pool, slots, lengths)
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert torch.isnan(k_pool[0]).all()  # Unused slot remains unmodified.


@pytest.mark.parametrize("control", _DEFAULT_CONTROLS or [None])
@torch.no_grad()
def test_default_matches_saved_outputs_from_before_hook_refactor(control):
    if control is None:
        pytest.skip(
            "Set XLLM_SCA_DEFAULT_CONTROLS for immutable pre-refactor FA4 outputs"
        )
    from safetensors.torch import load_file

    sca, interface = _modules()
    _, tensors = _fixture(control["fixture"])
    previous = load_file(control["outputs"], device="cuda")
    q, k, v = (tensors[name] for name in ("q", "k", "v"))
    history = len(k) - len(q)
    for name in ("case_0", "case_1"):
        length = len(previous[name])
        selected = (q[:length], k[: history + length], v[: history + length])
        before = _default_attention(interface, *selected)
        torch.testing.assert_close(before, previous[name], rtol=0, atol=0)
        custom = sca.native_sca_attention(*selected)
        torch.testing.assert_close(custom, tensors["expected"][:length], rtol=0, atol=0)
        after = _default_attention(interface, *selected)
        torch.testing.assert_close(after, previous[name], rtol=0, atol=0)
        if name == "case_0":
            # This fixture distinguishes custom from standard arithmetic;
            # accidentally replacing the default cannot satisfy both oracles.
            assert not torch.equal(previous[name], tensors["expected"])


def _tensor_digest(value):
    return hashlib.sha256(
        value.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.no_grad()
def _persistent_worker(output, reverse):
    sca, interface = _modules()
    from cutlass import cute

    calls, original_compile = [], cute.compile

    def record_compile(function, *args, **kwargs):
        calls.append(type(function).__name__)
        return original_compile(function, *args, **kwargs)

    cute.compile = record_compile
    q, k, v = _synthetic_inputs()
    outputs = {}
    try:
        for name in ("custom", "default") if reverse else ("default", "custom"):
            value = (
                sca.native_sca_attention(q, k, v)
                if name == "custom"
                else _default_attention(interface, q, k, v)
            )
            outputs[name] = _tensor_digest(value)
    finally:
        cute.compile = original_compile
    cache_object = interface._flash_attn_fwd.compile_cache
    assert len(cache_object.cache) == 2
    report = {
        "compile_calls": calls,
        "outputs": outputs,
        "object_count": len(list(cache_object.cache_path.glob("*.o"))),
    }
    Path(output).write_text(json.dumps(report))


def test_custom_cache_survives_fresh_process_without_recompiling(tmp_path):
    environment = os.environ.copy()
    environment["SGLANG_CUTE_AOT_CACHE_DIR"] = str(tmp_path / "objects")
    environment["SGLANG_CACHE_DIR"] = str(tmp_path / "sglang")
    reports = []
    for index in range(2):
        destination = tmp_path / f"worker{index}.json"
        command = [
            sys.executable,
            str(Path(__file__)),
            "--cache-worker",
            str(destination),
            str(index),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        reports.append(json.loads(destination.read_text()))
    assert len(reports[0]["compile_calls"]) == 2, reports[0]
    assert reports[1]["compile_calls"] == [], reports[1]
    assert reports[0]["object_count"] == reports[1]["object_count"] == 2
    assert reports[0]["outputs"] == reports[1]["outputs"]


@pytest.mark.parametrize(
    "invalid", ["width", "dtype", "length", "heads", "gradient", "scale"]
)
def test_reject_unsupported_inputs(invalid):
    sca, _ = _modules()
    q, k, v = _synthetic_inputs()
    scale = 0.125
    if invalid == "width":
        q = q[..., :32]
    elif invalid == "dtype":
        v = v.float()
    elif invalid == "length":
        k, v = k[:1], v[:1]
    elif invalid == "heads":
        q = q[:, :3]
    elif invalid == "gradient":
        q.requires_grad_(True)
    else:
        scale = float("nan")
    with pytest.raises(ValueError):
        sca.native_sca_attention(q, k, v, scale)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cache-worker":
        _persistent_worker(sys.argv[2], bool(int(sys.argv[3])))
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
