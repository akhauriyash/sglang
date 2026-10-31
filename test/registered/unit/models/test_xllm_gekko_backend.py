"""CPU tests of the actual eager Gekko pool serialization and request routing.

Only the GPU allocator/base backend imports are replaced. These checks compare
the real pooled operator with the torch reference; they do not qualify native
CUDA numerics or scheduler allocation itself.
"""

import importlib
import importlib.util
import sys
import types
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE = sys.modules.setdefault(
    "_xllm_models_cpu", types.ModuleType("_xllm_models_cpu")
)
_PACKAGE.__path__ = [str(_ROOT / "python/sglang/srt/models")]
reference = importlib.import_module("_xllm_models_cpu.xllm_gekko_reference")


def _load(name, path, modules=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**(modules or {}), name: module}):
        spec.loader.exec_module(module)
    return module


def _module(**attributes):
    result = types.ModuleType("_gekko_backend_dependency")
    result.__dict__.update(attributes)
    return result


_ci = _load("_xllm_gekko_pool_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class PoolBase:
    def __init__(self, caches):
        self.caches = caches
        self._model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    gekko_attention={
                        "model_dim": 1536,
                        "num_heads": 32,
                        "num_kv_heads": 8,
                        "head_dim": 64,
                        "chunk_size": 4096,
                        "timenorm_num_groups": 32,
                    }
                )
            )
        )

    def init_forward_metadata(self, forward_batch):
        # The real base translates allocator slots once per step. Everything
        # downstream, including Gekko's packing of variable lengths, is real.
        self._cache_indices = forward_batch.slots

    def conv_state_metadata(self, layer_id, forward_batch):
        return SimpleNamespace(layer_cache=self.caches[layer_id])


class HybridBase:
    def __init__(self, full_attn_backend, short_conv_backend, full_attn_layers):
        self.full_attn_backend = full_attn_backend
        self.short_conv_backend = short_conv_backend
        self.full_attn_layers = full_attn_layers


class FlashBackendStub:
    def __init__(self, version):
        self.fa_impl_ver = version
        self.num_splits = 0
        self.decode_num_splits = 4
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1
        return self.num_splits, self.decode_num_splits


@pytest.fixture(autouse=True)
def flash_backend_import(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.flashattention_backend",
        _module(FlashAttentionBackend=FlashBackendStub),
    )


backend_module = _load(
    "_xllm_gekko_pool_cpu",
    _ROOT / "python/sglang/srt/layers/attention/linear/xllm_gekko_backend.py",
    {
        "sglang.srt.layers.attention.hybrid_linear_attn_backend": _module(
            ShortConvHybridAttnBackend=HybridBase
        ),
        "sglang.srt.layers.attention.linear.short_conv_backend": _module(
            ShortConvAttnBackend=PoolBase
        ),
        "sglang.srt.models.xllm_gekko_reference": reference,
    },
)
Backend = backend_module.XllmGekkoAttnBackend


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([254], True),
        ([255], True),
        ([256], False),
        ([257], True),
        ([4095], True),
        ([4096], False),
        ([4097], True),
        ([8191], True),
        ([8192], False),
        ([8193], True),
        ([17, 256, 8193], False),
        ([17, 257, 8193], True),
        ([1], False),
    ],
)
def test_decode_graph_eligibility_uses_all_real_request_positions(lengths, expected):
    backend = Backend({})
    metadata = batch(
        list(range(len(lengths))),
        [value - 1 for value in lengths],
        [1] * len(lengths),
        mode="decode",
    )
    assert backend.can_run_decode_cuda_graph(metadata) is expected


@pytest.mark.parametrize("mode", ["extend", "idle"])
def test_decode_graph_rejects_non_decode_modes(mode):
    assert not Backend({}).can_run_decode_cuda_graph(batch([0], [17], [1], mode))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_dim", 768),
        ("num_heads", 16),
        ("num_kv_heads", 4),
        ("head_dim", 32),
        ("v_head_dim", 128),
        ("chunk_size", 2048),
        ("timenorm_num_groups", 16),
    ],
)
def test_other_geometries_retain_eager_decode(field, value):
    backend = Backend({})
    backend._model_runner.model_config.hf_text_config.gekko_attention[field] = value
    assert not backend.can_run_decode_cuda_graph(batch([0], [17], [1], "decode"))


@pytest.mark.parametrize("version", [3, 4])
def test_single_split_policy_is_gekko_fa4_local_and_precedes_forward(version):
    full, unrelated = FlashBackendStub(version), FlashBackendStub(4)
    recurrent, layers = object(), [0, 4, 8]
    hybrid = backend_module.XllmGekkoHybridAttnBackend(full, recurrent, layers)
    assert hybrid.full_attn_backend is full
    assert hybrid.short_conv_backend is recurrent
    assert hybrid.full_attn_layers is layers
    assert full.forward_calls == 0
    assert full.forward() == ((1, 1) if version == 4 else (0, 4))
    assert unrelated.forward() == (0, 4)


def test_unrelated_backend_with_matching_attribute_names_is_unchanged():
    full = SimpleNamespace(fa_impl_ver=4, num_splits=7, decode_num_splits=9)
    before = vars(full).copy()
    backend_module.XllmGekkoHybridAttnBackend(full, object(), [0])
    assert vars(full) == before


class Mode:
    def __init__(self, mode):
        self.mode = mode

    def is_decode(self):
        return self.mode == "decode"

    def is_extend(self):
        return self.mode == "extend"

    def is_idle(self):
        return self.mode == "idle"


def batch(slots, prefixes, lengths, mode="extend"):
    return SimpleNamespace(
        slots=torch.tensor(slots, dtype=torch.int64),
        forward_mode=Mode(mode),
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens_cpu=lengths,
        seq_lens_cpu=torch.tensor(
            [prefix + 1 for prefix in prefixes], dtype=torch.int64
        ),
    )


def layer():
    result = reference.GekkoAttention(
        {
            "model_dim": 12,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "v_head_dim": 4,
            "chunk_size": 4,
            "timenorm_num_groups": 3,
            "apply_bias_term": True,
            "gekko_value_activation": "identity",
        }
    ).eval()
    with torch.no_grad():
        for convolution in (result.q_conv, result.k_conv, result.v_conv):
            convolution.weight.normal_(0.1, 0.4)
        result.timenorm.weight.uniform_(-0.2, 0.2)
        result.timenorm.bias.uniform_(-0.1, 0.1)
        result.query_norm.weight.uniform_(-0.2, 0.2)
        result.key_norm.weight.uniform_(-0.2, 0.2)
        result.timenorm.prior_count.fill_(7)
        result.timenorm.prior_mean.fill_(0.002)
        result.timenorm.prior_var.fill_(0.001)
        for parameter in result.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    return result


def pool(module, slots=4):
    width = module.q_conv.width - 1
    qdim = module.num_heads * module.head_dim
    kdim = module.num_kv_heads * module.head_dim
    vdim = module.num_kv_heads * module.v_head_dim
    capacity = 2 * module.chunk_size
    shapes = [
        (width, qdim),
        (width, kdim),
        (width, vdim),
        (capacity, kdim),
        (capacity, vdim),
        (capacity, kdim),
        (capacity, kdim),
    ]
    temporal = (
        module.num_kv_heads * module.head_dim * module.v_head_dim
        + kdim
        + 2 * module.timenorm.num_groups
    )
    return SimpleNamespace(
        conv=[
            torch.full((slots, *shape), torch.nan, dtype=torch.bfloat16)
            for shape in shapes
        ],
        temporal=torch.full((slots, 1, 1, temporal), torch.nan, dtype=torch.float32),
    )


def assert_state_exact(actual, expected):
    for entry in fields(expected):
        first, second = getattr(actual, entry.name), getattr(expected, entry.name)
        if isinstance(first, torch.Tensor):
            assert first.dtype == second.dtype, entry.name
            torch.testing.assert_close(first, second, atol=0, rtol=0, msg=entry.name)
        else:
            assert first == second, entry.name


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("position", [1, 3, 4, 5, 7, 8, 9, 12, 15])
@torch.no_grad()
def test_state_roundtrip_across_chunk_boundaries(position):
    torch.manual_seed(138)
    operator = layer()
    hidden = torch.randn(position, 12).bfloat16()
    _, expected = operator(hidden)
    cache = pool(operator)
    Backend._write_state(cache, 2, expected)
    actual = Backend._read_state(operator, cache, 2, position, hidden)
    assert_state_exact(actual, expected)
    assert not torch.isnan(cache.temporal[2]).any()


@torch.no_grad()
def test_fresh_pool_state_rounds_priors_and_ignores_stale_slot_contents():
    operator = layer()
    hidden = torch.zeros(1, 12, dtype=torch.bfloat16)
    cache = pool(operator)
    saved_mean = operator.timenorm.prior_mean.clone()
    saved_var = operator.timenorm.prior_var.clone()
    state = Backend._read_state(operator, cache, 2, 0, hidden)
    assert state.tdn_mean.dtype == state.tdn_var.dtype == torch.float32
    torch.testing.assert_close(
        state.tdn_mean, saved_mean.bfloat16().float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        state.tdn_var, saved_var.bfloat16().float(), atol=0, rtol=0
    )
    assert not torch.equal(state.tdn_mean, saved_mean)
    assert not torch.equal(state.tdn_var, saved_var)
    torch.testing.assert_close(operator.timenorm.prior_mean, saved_mean, atol=0, rtol=0)
    torch.testing.assert_close(operator.timenorm.prior_var, saved_var, atol=0, rtol=0)
    assert state.tdn_count.item() == 7
    assert state.position == 0


@torch.no_grad()
def test_prior_count_remains_int64_above_float32_integer_precision():
    operator = layer()
    operator.timenorm.prior_count.fill_(2**30 + 1)
    hidden = torch.randn(3, 12).bfloat16()
    _, expected = operator(hidden)
    cache = pool(operator)
    Backend._write_state(cache, 0, expected)
    actual = Backend._read_state(operator, cache, 0, 3, hidden)
    assert actual.tdn_count.item() == 2**30 + 4
    assert actual.tdn_count.dtype == torch.int64
    assert_state_exact(actual, expected)


@torch.no_grad()
def test_packed_tdn_roundtrip_preserves_every_word_and_slot_ownership(monkeypatch):
    operator = layer()
    hidden = torch.randn(3, 12).bfloat16()
    _, expected = operator(hidden)
    # Unique values throughout the payload detect missing partial moments,
    # block-start statistics, powers, or incorrect slice offsets.
    packed_size = 2 * 32 * 256 + 2 * 32 + 2
    expected.tdn_packed = torch.arange(packed_size, dtype=torch.float32) / 64
    cache = pool(operator)
    cache.temporal = torch.cat(
        (
            cache.temporal,
            torch.full((4, 1, 1, packed_size), torch.nan, dtype=torch.float32),
        ),
        dim=-1,
    )
    # Pool serialization is device independent; CUDA dispatch itself has a
    # separate integration test using the real native TDN operator.
    monkeypatch.setattr(operator, "uses_native_tdn", lambda hidden: True)
    Backend._write_state(cache, 2, expected)
    restored = Backend._read_state(operator, cache, 2, 3, hidden)
    assert_state_exact(restored, expected)
    assert not torch.isnan(cache.temporal[2]).any()
    assert torch.isnan(cache.temporal[1]).all()
    copied = restored.clone()
    copied.tdn_packed.fill_(-1)
    torch.testing.assert_close(restored.tdn_packed, expected.tdn_packed, atol=0, rtol=0)
    assert restored.tdn_count.item() == 10
    assert restored.position == 3
    Backend._write_state(cache, 0, restored)
    assert_state_exact(Backend._read_state(operator, cache, 0, 3, hidden), expected)
    # A new request must use its initializer even if the reused slot contains
    # NaNs in the packed scan payload and every other state stream.
    cache.temporal[2].fill_(torch.nan)
    sentinel = object()
    monkeypatch.setattr(operator, "initial_state", lambda hidden: sentinel)
    assert Backend._read_state(operator, cache, 2, 0, hidden) is sentinel


@torch.no_grad()
def test_arbitrary_pieces_reordered_decode_and_reused_slots():
    torch.manual_seed(821)
    operator = layer()
    cache = pool(operator)
    backend = Backend({3: cache})
    hybrid = backend_module.XllmGekkoHybridAttnBackend(object(), backend, [])
    direct = {}

    def step(requests, mode="extend"):
        # Each entry is (logical request name, allocator slot, token count).
        prefixes = [
            0 if name not in direct else direct[name].position
            for name, _, _ in requests
        ]
        counts = [count for _, _, count in requests]
        current = batch([slot for _, slot, _ in requests], prefixes, counts, mode)
        hidden = torch.randn(sum(counts), 12).bfloat16()
        expected, offset = [], 0
        for name, _, count in requests:
            output, direct[name] = operator(
                hidden[offset : offset + count], direct.get(name)
            )
            expected.append(output)
            offset += count
        backend.init_forward_metadata(current)
        actual = hybrid.forward_gekko(3, operator, hidden, current)
        torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)
        for name, slot, _ in requests:
            restored = Backend._read_state(
                operator, cache, slot, direct[name].position, hidden
            )
            assert_state_exact(restored, direct[name])

    step([("a", 2, 3), ("b", 0, 5)])
    step([("b", 0, 2), ("a", 2, 6)])
    step([("a", 2, 1), ("b", 0, 1)], mode="decode")
    # Cancellation/reuse must initialize from model priors at prefix zero,
    # even if the allocator's old contents have not been cleared.
    for stream in cache.conv:
        stream[2].fill_(torch.nan)
    cache.temporal[2].fill_(torch.nan)
    step([("c", 2, 1), ("b", 0, 4)])
    step([("b", 0, 1), ("c", 2, 1)], mode="decode")
    step([("c", 2, 7)])


@torch.no_grad()
def test_pool_layers_are_independent():
    torch.manual_seed(975)
    operators = {1: layer(), 5: layer()}
    caches = {index: pool(operator) for index, operator in operators.items()}
    backend = Backend(caches)
    direct = {}
    for prefix, length in ((0, 5), (5, 2), (7, 1)):
        metadata = batch([1], [prefix], [length])
        backend.init_forward_metadata(metadata)
        hidden = torch.randn(length, 12).bfloat16()
        for index, operator in operators.items():
            expected, direct[index] = operator(hidden, direct.get(index))
            actual = backend.forward_gekko(index, operator, hidden, metadata)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert_state_exact(
                Backend._read_state(
                    operator, caches[index], 1, prefix + length, hidden
                ),
                direct[index],
            )


@torch.no_grad()
def test_idle_and_unsupported_metadata():
    operator = layer()
    backend = Backend({1: pool(operator)})
    idle = batch([], [], [], "idle")
    backend.init_forward_metadata(idle)
    hidden = torch.empty(0, 12, dtype=torch.bfloat16)
    assert backend.forward_gekko(1, operator, hidden, idle).shape == (0, 12)
    with pytest.raises(ValueError, match="prefill and decode"):
        backend.init_forward_metadata(batch([1], [0], [1], "verify"))
    padded = batch([-1], [0], [1])
    backend.init_forward_metadata(padded)
    with pytest.raises(ValueError, match="Padded"):
        backend.forward_gekko(1, operator, torch.randn(1, 12).bfloat16(), padded)
    with pytest.raises(ValueError, match="distinct"):
        backend.init_forward_metadata(batch([1, 1], [0, 0], [1, 1]))
    incomplete = batch([1], [0], [1])
    backend.init_forward_metadata(incomplete)
    with pytest.raises(ValueError, match="cover"):
        backend.forward_gekko(1, operator, torch.randn(2, 12).bfloat16(), incomplete)
    with pytest.raises(ValueError):
        backend.init_forward_metadata(batch([0, 1], [0], [1]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
