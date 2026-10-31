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

"""CPU/CUDA lifecycle and gradient checks for opt-in RoPE frequency caching."""

import gc
import importlib.util
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = load(
    "_xllm_rope_cache", _ROOT / "python/sglang/srt/models/xllm_reference.py"
)
NativeRotaryEmbedding = reference.NativeRotaryEmbedding
ci = load("_xllm_rope_cache_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = ci.register_cpu_ci
register_cuda_ci = ci.register_cuda_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def make(**kwargs):
    return NativeRotaryEmbedding(
        64,
        rope_base=1000000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 128,
            "beta_slow": 4,
            "attention_factor": 1.2772588722239782,
            "original_max_position_embeddings": None,
        },
        cache_inverse_frequencies=True,
        **kwargs,
    )


def test_default_remains_uncached():
    rotary = NativeRotaryEmbedding(64)
    first = rotary.inverse_frequencies("cpu")
    second = rotary.inverse_frequencies("cpu")
    assert first is not second
    assert rotary._inverse_frequency_cache is None


def test_cpu_cache_does_not_query_cuda_and_does_not_rebuild(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU frequency lookup touched CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    rotary = make()
    frequency = rotary.inverse_frequencies("cpu")
    original = NativeRotaryEmbedding._compute_inverse_frequencies

    def forbidden_rebuild(*args, **kwargs):
        raise AssertionError("Repeated frequency lookup rebuilt the vector")

    monkeypatch.setattr(
        NativeRotaryEmbedding, "_compute_inverse_frequencies", forbidden_rebuild
    )
    for _ in range(5):
        assert rotary.inverse_frequencies(torch.device("cpu")) is frequency
    monkeypatch.setattr(NativeRotaryEmbedding, "_compute_inverse_frequencies", original)


@pytest.mark.parametrize(
    "name,value",
    [
        ("head_dim", 128),
        ("rope_head_dim", 32),
        ("base", 500000),
        ("max_position_embeddings", 65536),
        ("attention_factor", 1.5),
        ("factor", 8),
        ("beta_fast", 64),
        ("beta_slow", 2),
        ("scaling_attention_factor", 1.5),
        ("original_max_position_embeddings", 32768),
        ("scaling", None),
    ],
)
def test_frequency_configuration_mutation_rebuilds(name, value):
    rotary = make()
    first = rotary.inverse_frequencies("cpu")
    if name == "scaling_attention_factor":
        rotary.scaling["attention_factor"] = value
    elif name in {
        "factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
    }:
        rotary.scaling[name] = value
    else:
        setattr(rotary, name, value)
    actual = rotary.inverse_frequencies("cpu")
    assert actual is not first
    torch.testing.assert_close(
        actual,
        NativeRotaryEmbedding._compute_inverse_frequencies(rotary, torch.device("cpu")),
        rtol=0,
        atol=0,
    )
    assert rotary.inverse_frequencies("cpu") is actual


def test_dtype_conversion_preserves_fp32_and_state_dict_keys():
    rotary = make()
    before = rotary.state_dict()
    frequency = rotary.inverse_frequencies("cpu")
    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        rotary.to(dtype=dtype)
        assert rotary.inverse_frequencies("cpu") is frequency
        assert frequency.dtype == torch.float32
        assert rotary.state_dict().keys() == before.keys()
        assert not dict(rotary.named_buffers())
        assert not dict(rotary.named_parameters())


def test_gradient_after_inference_mode_warmup():
    rotary = make()
    with torch.inference_mode():
        cached = rotary.inverse_frequencies("cpu")
    assert not cached.is_inference()
    assert not cached.requires_grad
    inputs = (
        torch.randn(5, 2, 64),
        torch.randn(5, 1, 64),
        torch.arange(5, dtype=torch.float32),
    )
    results = []
    for enabled in (True, False):
        rotary.cache_inverse_frequencies = enabled
        q, k, positions = (x.clone().requires_grad_() for x in inputs)
        a, b = rotary(q, k, positions)
        (a.square().sum() + b.square().sum()).backward()
        results.append((a, b, q.grad, k.grad, positions.grad))
    for actual, expected in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires allocated CUDA")
def test_device_and_cuda_stream_switch_rebuild_safely():
    rotary = make()
    cpu = rotary.inverse_frequencies("cpu")
    cuda = rotary.inverse_frequencies("cuda")
    assert cuda.device.type == "cuda"
    assert cuda is rotary.inverse_frequencies(
        torch.device("cuda", torch.cuda.current_device())
    )
    assert cpu is not cuda
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        on_other = rotary.inverse_frequencies("cuda")
        assert on_other is rotary.inverse_frequencies("cuda")
        expected = NativeRotaryEmbedding._compute_inverse_frequencies(
            rotary, torch.device("cuda")
        )
        assert torch.equal(on_other, expected)
    other.synchronize()
    assert rotary.inverse_frequencies("cuda") is not on_other
    again_cpu = rotary.inverse_frequencies("cpu")
    assert again_cpu.device.type == "cpu" and again_cpu is not cpu
    torch.testing.assert_close(again_cpu, cpu, rtol=0, atol=0)


def test_captured_frequencies_own_each_identity_until_module_destruction(monkeypatch):
    rotary = make()
    compute = rotary._compute_inverse_frequencies
    state = {"stream": 11, "capturing": False, "device": 0}
    requested = torch.device("cuda:1")

    @contextmanager
    def selected_device(device):
        previous = state["device"]
        state["device"] = device.index
        try:
            yield
        finally:
            state["device"] = previous

    with monkeypatch.context() as scoped:
        scoped.setattr(
            rotary,
            "_compute_inverse_frequencies",
            lambda device, compute=compute: compute(torch.device("cpu")),
        )
        scoped.setattr(
            torch.cuda,
            "current_stream",
            lambda device: SimpleNamespace(cuda_stream=state["stream"]),
        )
        scoped.setattr(torch.cuda, "device", selected_device)
        scoped.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: state["capturing"] and state["device"] == requested.index,
        )
        for stream in (11, 22, 33):
            state["stream"] = stream
            rotary.inverse_frequencies(requested)
        assert not rotary._captured_inverse_frequencies
        references = []
        for index in range(2):
            state.update(stream=11, capturing=False)
            rotary.inverse_frequencies(requested)
            references.append(weakref.ref(rotary._inverse_frequency_cache[1]))
            state["capturing"] = True
            for _ in range(2):
                rotary.inverse_frequencies(requested)
            assert len(rotary._captured_inverse_frequencies) == index + 1
            state.update(stream=22, capturing=False)
            rotary.inverse_frequencies(requested)
            gc.collect()
            assert all(reference() is not None for reference in references)
        assert references[0]() is not references[1]()
        rotary.to(torch.bfloat16)
        assert all(
            tensor.dtype == torch.float32
            for tensor in rotary._captured_inverse_frequencies.values()
        )
        assert not rotary.state_dict()
        assert state["device"] == 0
    del compute, rotary
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires allocated CUDA")
@torch.no_grad()
def test_capture_constants_survive_stream_switch_recapture_and_allocator_churn():
    rotary = make()
    q = torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([8192], device="cuda")
    capture_stream, eager_stream = torch.cuda.Stream(), torch.cuda.Stream()
    graphs, outputs, references, churn = [], [], [], []
    torch.cuda.synchronize()
    for index in range(2):
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                rotary(q, k, positions)
            references.append(weakref.ref(rotary._inverse_frequency_cache[1]))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            output = rotary(q, k, positions)
        graphs.append(graph)
        outputs.append(output)
        with torch.cuda.stream(eager_stream):
            expected = rotary(q, k, positions)
        torch.cuda.synchronize()
        gc.collect()
        assert all(reference() is not None for reference in references)
        assert len(rotary._captured_inverse_frequencies) == index + 1
        with torch.cuda.stream(capture_stream):
            churn.extend(
                torch.empty(32, device="cuda", dtype=torch.float32).fill_(float("nan"))
                for _ in range(256)
            )
        torch.cuda.synchronize()
        for captured in graphs:
            captured.replay()
        torch.cuda.synchronize()
        for actual in outputs:
            for tensor, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(tensor, wanted, atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
