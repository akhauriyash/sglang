"""CPU adapter contract tests with explicit stand-ins for GPU attention/pools.

These exercise native tensor loading, wrapper invocation and numerical wiring.
They do not qualify CUDA kernels, the scheduler or learned-checkpoint parity.
"""

import importlib
import importlib.util
import sys
import types
from collections import Counter
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _ROOT / "python/sglang/srt/models"
_PACKAGE = sys.modules.setdefault(
    "_xllm_models_cpu", types.ModuleType("_xllm_models_cpu")
)
_PACKAGE.__path__ = [str(_MODELS)]
reference = importlib.import_module("_xllm_models_cpu.xllm_reference")
gekko_reference = importlib.import_module("_xllm_models_cpu.xllm_gekko_reference")
gekko_gemm = importlib.import_module("_xllm_models_cpu.xllm_gemm")


def _load(name, path, modules=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**(modules or {}), name: module}):
        spec.loader.exec_module(module)
    return module


_ci = _load("_xllm_adapter_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _module(**attributes):
    module = types.ModuleType("_adapter_dependency")
    module.__dict__.update(attributes)
    return module


class ConvStream(IntEnum):
    K_LOCAL = 2
    V_LOCAL = 3
    ATTN = 4
    MLP = 5


class TorchShortConvolution(nn.Module):
    def __init__(self, channels, width, stream, use_residual, layer_id):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, width))
        self.width, self.stream, self.layer_id = width, stream, layer_id
        self.use_residual = use_residual

    def forward(self, hidden_states, positions, forward_batch):
        assert positions is None
        assert not self.use_residual
        forward_batch.calls.append((self.layer_id, self.stream))
        key = (self.layer_id, self.stream)
        history = forward_batch.conv.get(key)
        if history is None:
            history = hidden_states.new_zeros((self.width - 1, hidden_states.shape[-1]))
        joined = torch.cat((history, hidden_states))
        convolution = torch.zeros_like(hidden_states, dtype=torch.float32)
        for tap in range(self.width):
            convolution += (
                joined[tap : tap + len(hidden_states)].float()
                * self.weight[:, 0, tap].float()
            )
        forward_batch.conv[key] = joined[-(self.width - 1) :].clone()
        return convolution.to(hidden_states.dtype)


class TorchRadixAttention(nn.Module):
    def __init__(
        self, num_heads, head_dim, scaling, num_kv_heads, layer_id, sliding_window_size
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.layer_id, self.window = layer_id, sliding_window_size
        self.scaling = scaling

    def forward(self, q, k, v, forward_batch):
        if self.layer_id in forward_batch.kv:
            old_k, old_v = forward_batch.kv[self.layer_id]
            k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
        forward_batch.kv[self.layer_id] = (k.clone(), v.clone())
        query_positions = forward_batch.positions
        key_positions = torch.arange(len(k), device=k.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.window >= 0:
            allowed &= key_positions[None, :] >= query_positions[:, None] - self.window
        ratio = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
        scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * self.scaling
        weights = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
        return torch.einsum("hts,shd->thd", weights, v.float()).to(q.dtype)


class TorchLogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, input_ids, hidden, output, forward_batch):
        return output(hidden).float()


class ForwardMode:
    def __init__(self, mode):
        self.mode = mode

    def is_decode(self):
        return self.mode == "decode"

    def is_extend(self):
        return self.mode in ("extend", "verify")

    def is_idle(self):
        return self.mode == "idle"

    def is_target_verify(self):
        return self.mode == "verify"


def _batch(mode="extend"):
    return SimpleNamespace(forward_mode=ForwardMode(mode), kv={}, conv={}, calls=[])


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(817)
    yield
    torch.set_default_dtype(previous_dtype)
    torch.set_num_threads(previous)


@pytest.fixture
def adapter():
    runtime = _module(
        get_parallel=lambda: SimpleNamespace(tp_size=1, pp_size=1),
        get_exec=lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config={
                    phase: SimpleNamespace(backend="disabled")
                    for phase in ("decode", "prefill")
                }
            )
        ),
        get_memory=lambda: SimpleNamespace(disable_radix_cache=True),
        get_model=lambda: SimpleNamespace(kv_cache_dtype="auto"),
        get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
        get_lora=lambda: SimpleNamespace(enable_lora=False),
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    return _load(
        "_xllm_serving_adapter_cpu",
        _MODELS / "xllm_hybrid.py",
        {
            "sglang.srt.layers.logits_processor": _module(
                LogitsProcessor=TorchLogitsProcessor
            ),
            "sglang.srt.layers.radix_attention": _module(
                RadixAttention=TorchRadixAttention
            ),
            "sglang.srt.model_executor.cuda_graph_config": _module(
                Backend=SimpleNamespace(DISABLED="disabled", FULL="full"),
                Phase=SimpleNamespace(
                    DECODE="decode", PREFILL="prefill", ALL=("decode", "prefill")
                ),
            ),
            "sglang.srt.models.inkling_common.sconv": _module(
                SconvType=ConvStream, ShortConvolution=TorchShortConvolution
            ),
            "sglang.srt.models.xllm_reference": reference,
            "sglang.srt.models.xllm_gekko_reference": gekko_reference,
            "sglang.srt.models.xllm_gemm": gekko_gemm,
            "sglang.srt.model_executor.forward_context": _module(
                get_attn_backend=lambda: None
            ),
            "sglang.srt.runtime_context": runtime,
        },
    )


class _ServingConfig(SimpleNamespace):
    @property
    def full_attention_layer_ids(self):
        return [i for i, kind in enumerate(self.layer_kinds) if kind != "gekko"]


def _config(inkling=True):
    native = {
        "arch": "hybrid_inkling" if inkling else "transformer",
        "model_dim": 12,
        "ffn_hidden_dim": 23,
        "vocab_size": 29,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 4,
        "rope_head_dim": 4,
        "rope_base": 500000.0,
        "max_position_embeddings": 128,
        "rmsnorm_eps": 1e-6,
    }
    kinds = ["full", "inkling", "inkling", "inkling"] if inkling else ["full"] * 4
    native["layer_kinds"] = kinds
    native["full_attention"] = {}
    native["inkling_attention"] = {
        "local_attention_window": 5,
        "causal_conv_width": 4,
        "causal_conv_weight_normalization": False,
    }
    return _ServingConfig(
        native_model_config=native,
        layer_kinds=kinds,
        num_hidden_layers=4,
        vocab_size=29,
        hidden_size=12,
        full_attention=reference.resolve_family(native, "full"),
        inkling_attention=reference.resolve_family(native, "inkling"),
    )


def _set_graph_backends(adapter, monkeypatch, decode, prefill):
    monkeypatch.setattr(
        adapter,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config={
                    "decode": SimpleNamespace(backend=decode),
                    "prefill": SimpleNamespace(backend=prefill),
                }
            )
        ),
    )


@pytest.mark.parametrize("inkling", [False, True])
def test_full_decode_graphs_construct_dense_and_inkling(adapter, monkeypatch, inkling):
    _set_graph_backends(adapter, monkeypatch, "full", "disabled")
    cls = adapter.XllmInklingForCausalLM if inkling else adapter.XllmDenseForCausalLM
    model = cls(_config(inkling))
    assert model.get_num_kv_cache_layers() == 4


@pytest.mark.parametrize("inkling", [False, True])
@pytest.mark.parametrize(
    "decode,prefill",
    [
        ("full", "full"),
        ("disabled", "full"),
        ("breakable", "disabled"),
        ("tc_piecewise", "disabled"),
    ],
)
def test_unsupported_graph_modes_rejected(
    adapter, monkeypatch, inkling, decode, prefill
):
    _set_graph_backends(adapter, monkeypatch, decode, prefill)
    cls = adapter.XllmInklingForCausalLM if inkling else adapter.XllmDenseForCausalLM
    with pytest.raises(ValueError, match="full decode CUDA graphs"):
        cls(_config(inkling))


def test_gekko_constructs_with_full_decode_graphs(adapter, monkeypatch):
    _set_graph_backends(adapter, monkeypatch, "full", "disabled")
    config = _config(False)
    config.layer_kinds = ["full", "gekko", "gekko", "gekko"]
    config.gekko_attention = dict(config.full_attention, chunk_size=8)
    # Constructor wiring uses a small CPU model; kernel geometry is checked
    # independently and real supported dimensions are covered by CUDA tests.
    monkeypatch.setattr(adapter, "supports_pooled_gekko_decode", lambda config: True)
    model = adapter.XllmGekkoForCausalLM(config)
    assert model.get_num_kv_cache_layers() == 1


def test_gekko_other_layouts_require_eager_graph_mode(adapter, monkeypatch):
    _set_graph_backends(adapter, monkeypatch, "full", "disabled")
    config = _config(False)
    config.layer_kinds = ["full", "gekko", "gekko", "gekko"]
    config.gekko_attention = dict(config.full_attention, chunk_size=8)
    with pytest.raises(ValueError, match="other Gekko layouts"):
        adapter.XllmGekkoForCausalLM(config)


@pytest.mark.parametrize("inkling", [False, True])
@torch.no_grad()
def test_adapter_forward_matches_core_and_pooled_call_signature(adapter, inkling):
    config = _config(inkling)
    core = reference.XllmReferenceModel(config).eval()
    cls = adapter.XllmInklingForCausalLM if inkling else adapter.XllmDenseForCausalLM
    serving = cls(config).eval()
    serving.load_weights(core.state_dict().items())
    tokens = torch.randint(0, 29, (17,))
    expected = core(tokens)
    batch, outputs, offset = _batch(), [], 0
    for count in (3, 4, 1, 1, 8):
        batch.positions = torch.arange(offset, offset + count)
        batch.forward_mode = ForwardMode("decode" if count == 1 else "extend")
        outputs.append(serving(tokens[offset : offset + count], batch.positions, batch))
        offset += count
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-5, atol=1e-6)
    assert serving.get_num_kv_cache_layers() == 4
    assert serving.get_attention_sliding_window_size() == (4 if inkling else None)
    assert len(batch.conv) == (12 if inkling else 0)
    for index, layer in enumerate(serving.layers):
        assert type(layer.nffn.fc2) is nn.Linear
        assert layer.attention.attn.window == (4 if inkling and index else -1)
        assert layer.attention.rope.cache_inverse_frequencies
        assert layer.attention.rope._inverse_frequency_cache[1].dtype == torch.float32
        assert not core.layers[index].attention.rope.cache_inverse_frequencies
    assert serving.state_dict().keys() == core.state_dict().keys()


@torch.no_grad()
def test_native_bf16_residual_add_is_after_conv_rounding(adapter):
    conv = adapter.XllmResidualShortConv(1, 4, ConvStream.K_LOCAL, 1).to(torch.bfloat16)
    conv.weight.zero_()
    conv.weight[0, 0, -1] = 1.328125
    inputs = torch.tensor([[0.72265625]], dtype=torch.bfloat16)
    batch = _batch()
    result = conv(inputs, batch)
    product = inputs.float() * conv.weight[0, 0, -1].float()
    expected = inputs + product.to(torch.bfloat16)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert not torch.equal(result, (inputs.float() + product).to(torch.bfloat16))
    assert not conv.use_residual


@torch.no_grad()
def test_gekko_adapter_full_model_pooled_decode_and_fp32_priors(adapter, monkeypatch):
    sca_calls = []

    def torch_sca(q, k, v, scale):
        sca_calls.append((len(q), len(k)))
        assert scale == q.shape[-1] ** -0.5
        attention = SimpleNamespace(
            num_heads=q.shape[1], num_kv_heads=k.shape[1], head_dim=q.shape[-1]
        )
        return gekko_reference.GekkoAttention._attend(
            attention, q, k, v, torch.arange(len(k) - len(q), len(k)), 0
        )

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.models.xllm_sca",
        _module(native_sca_attention=torch_sca),
    )
    config = _config(False)
    config.layer_kinds = ["full", "gekko", "gekko", "gekko"]
    config.native_model_config.update(
        {
            "arch": "hybrid_gekko",
            "layer_kinds": config.layer_kinds,
            "gekko_attention": {
                "v_head_dim": 4,
                "causal_conv_width": 4,
                "chunk_size": 4,
                "timenorm_num_groups": 3,
                "apply_bias_term": True,
            },
        }
    )
    config.gekko_attention = reference.resolve_family(
        config.native_model_config, "gekko"
    )
    # Native checkpoint/reference buffers retain FP32 even when all learned
    # parameters use BF16. ModelLoader constructs the serving model under BF16.
    torch.set_default_dtype(torch.float32)
    core = reference.XllmReferenceModel(config).eval()
    for parameter in core.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    for block in core.layers[1:]:
        block.gda.timenorm.prior_count.fill_(7)
        block.gda.timenorm.prior_mean.fill_(0.00312345)
        block.gda.timenorm.prior_var.fill_(0.004321)
    torch.set_default_dtype(torch.bfloat16)
    serving = adapter.XllmGekkoForCausalLM(config).eval()
    assert serving.state_dict().keys() == core.state_dict().keys()
    for source, actual in zip(core.layers, serving.layers, strict=True):
        if hasattr(actual, "gda"):
            assert actual.gda.rotary.cache_inverse_frequencies
            assert not source.gda.rotary.cache_inverse_frequencies
        else:
            assert actual.attention.rope.cache_inverse_frequencies
            assert not source.attention.rope.cache_inverse_frequencies
    # The full-attention layers belong to the Gekko model too and require the
    # same native padded-oracle down-projection arithmetic as converted layers.
    assert all(
        isinstance(layer.nffn.fc2, gekko_gemm.GekkoDownProjection)
        for layer in serving.layers
    )
    assert all(type(layer.nffn.fc2) is nn.Linear for layer in core.layers)
    serving.load_weights(core.state_dict().items())
    for source, actual in zip(core.layers[1:], serving.layers[1:], strict=True):
        assert actual.gda.timenorm.prior_mean.dtype == torch.float32
        torch.testing.assert_close(
            actual.gda.timenorm.prior_mean,
            source.gda.timenorm.prior_mean,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual.gda.timenorm.prior_var, source.gda.timenorm.prior_var, atol=0, rtol=0
        )

    class PoolBase:
        def init_forward_metadata(self, forward_batch):
            self._cache_indices = forward_batch.slots

        def conv_state_metadata(self, layer_id, forward_batch):
            return SimpleNamespace(layer_cache=self.caches[layer_id])

    module = _load(
        "_xllm_gekko_adapter_pool_cpu",
        _ROOT / "python/sglang/srt/layers/attention/linear/xllm_gekko_backend.py",
        {
            "sglang.srt.layers.attention.hybrid_linear_attn_backend": _module(
                ShortConvHybridAttnBackend=object
            ),
            "sglang.srt.layers.attention.linear.short_conv_backend": _module(
                ShortConvAttnBackend=PoolBase
            ),
            "sglang.srt.models.xllm_gekko_reference": gekko_reference,
        },
    )
    backend = module.XllmGekkoAttnBackend()
    backend.caches = {
        index: SimpleNamespace(
            conv=[
                torch.full((2, *shape), torch.nan, dtype=torch.bfloat16)
                for shape in [(3, 16), (3, 8), (3, 8)] + [(8, 8)] * 4
            ],
            temporal=torch.full((2, 1, 1, 46), torch.nan, dtype=torch.float32),
        )
        for index in (1, 2, 3)
    }
    adapter.get_attn_backend = lambda: backend
    tokens = torch.randint(0, 29, (15,))
    state = core.new_cache()
    metadata, offset = _batch(), 0
    metadata.slots = torch.tensor([1], dtype=torch.int64)
    for count in (3, 5, 1, 1, 5):
        metadata.forward_mode = ForwardMode("decode" if count == 1 else "extend")
        metadata.positions = torch.arange(offset, offset + count)
        metadata.extend_prefix_lens_cpu = [offset]
        metadata.extend_seq_lens_cpu = [count]
        metadata.seq_lens_cpu = torch.tensor([offset + 1], dtype=torch.int64)
        backend.init_forward_metadata(metadata)
        expected = core(tokens[offset : offset + count], state)
        actual = serving(tokens[offset : offset + count], metadata.positions, metadata)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        offset += count
    assert serving.get_num_kv_cache_layers() == 1
    assert Counter(sca_calls) == Counter(
        {(3, 3): 3, (1, 4): 3, (4, 8): 3, (1, 5): 3, (1, 6): 3, (2, 8): 3, (3, 7): 3}
    )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "shape"])
def test_strict_loader_rejects_incomplete_or_ambiguous_weights(adapter, failure):
    config = _config()
    model = adapter.XllmInklingForCausalLM(config)
    tensors = list(reference.XllmReferenceModel(config).state_dict().items())
    if failure == "missing":
        tensors.pop()
    elif failure == "duplicate":
        tensors.append(tensors[0])
    elif failure == "unexpected":
        tensors.append(("unknown.weight", torch.ones(1)))
    else:
        tensors[0] = (tensors[0][0], torch.ones(1))
    with pytest.raises(ValueError):
        model.load_weights(tensors)


def test_unqualified_modes_and_runtime_fail_closed(adapter):
    config = _config()
    with pytest.raises(ValueError, match="layer kinds"):
        adapter.XllmDenseForCausalLM(config)
    with pytest.raises(ValueError, match="quantization"):
        adapter.XllmInklingForCausalLM(config, quant_config=object())
    torch.set_default_dtype(torch.float32)
    with pytest.raises(ValueError, match="BF16"):
        adapter.XllmInklingForCausalLM(config)
    torch.set_default_dtype(torch.bfloat16)
    model = adapter.XllmInklingForCausalLM(config)
    with pytest.raises(ValueError, match="verification"):
        model(torch.tensor([1]), torch.tensor([0]), _batch("verify"))
    with pytest.raises(ValueError, match="forward mode"):
        model(torch.tensor([1]), torch.tensor([0]), _batch("draft"))
    adapter.get_parallel = lambda: SimpleNamespace(tp_size=2, pp_size=1)
    with pytest.raises(ValueError, match="TP1"):
        adapter.XllmInklingForCausalLM(config)


@pytest.mark.parametrize("dtype", ["float16", "fp8_e4m3", "fp8_e5m2", "mxfp8"])
def test_unqualified_kv_cache_dtypes_fail_closed(adapter, dtype):
    adapter.get_model = lambda: SimpleNamespace(kv_cache_dtype=dtype)
    with pytest.raises(ValueError, match="BF16 KV cache"):
        adapter.XllmInklingForCausalLM(_config())


def test_all_softmax_layers_dispatch_through_paged_backend():
    class BaseBackend:
        def __init__(self, full, linear, full_layers):
            self.full_layers = full_layers
            self.full, self.linear = full, linear

    module = _load(
        "_xllm_backend_cpu",
        _ROOT / "python/sglang/srt/layers/attention/linear/xllm_inkling_backend.py",
        {
            "sglang.srt.layers.attention.linear.inkling_sconv_backend": _module(
                InklingShortConvHybridAttnBackend=BaseBackend
            )
        },
    )
    full = object()
    linear = SimpleNamespace(
        _model_runner=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=_config())
        )
    )
    backend = module.XllmInklingHybridAttnBackend(full, linear, [0])
    assert backend.full_layers == [0, 1, 2, 3]
    assert backend.full is full and backend.linear is linear


def test_config_swa_layer_mapping_and_validation():
    registry = _load(
        "_xllm_linear_registry_cpu",
        _ROOT / "python/sglang/srt/configs/linear_attn_model_registry.py",
    )
    mamba_utils = _load(
        "_xllm_mamba_utils_cpu",
        _ROOT / "python/sglang/srt/configs/mamba_utils.py",
        {
            "sglang.srt.distributed.utils": _module(divide=lambda a, b: a // b),
            "sglang.srt.environ": _module(envs=SimpleNamespace()),
        },
    )
    module = _load(
        "_xllm_config_cpu",
        _ROOT / "python/sglang/srt/configs/xllm_hybrid.py",
        {
            "sglang.srt.configs.linear_attn_model_registry": registry,
            "sglang.srt.configs.mamba_utils": mamba_utils,
        },
    )
    kwargs = {
        **vars(_config()),
        "xllm_export_schema_version": 1,
        "native_norm_offset": True,
        "rotary_interleaved": True,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
    }
    config = module.XllmHybridConfig(**kwargs)
    assert config.full_attention_layer_ids == [0, 1, 2, 3]
    assert config.linear_layer_ids == [1, 2, 3]
    assert config.hybrid_layer_pattern == [0, 1, 1, 1]
    assert config.sliding_window == 5
    spec, actual = registry.get_linear_attn_config(config)
    assert actual is config
    assert spec.arch_names == ["XllmInklingForCausalLM"]
    assert not spec.support_mamba_cache
    for bad in (
        {"xllm_export_schema_version": 0},
        {"native_norm_offset": False},
        {"rotary_interleaved": False},
        {"head_dim": 3},
        {"layer_kinds": ["full"]},
        {"layer_kinds": ["full", "inkling", "gekko", "gekko"]},
        {
            "inkling_attention": {
                **kwargs["inkling_attention"],
                "local_attention_window": 1,
            }
        },
    ):
        with pytest.raises(ValueError):
            module.XllmHybridConfig(**{**kwargs, **bad})
    gekko = module.XllmHybridConfig(
        **{
            **kwargs,
            "layer_kinds": ["full", "gekko", "gekko", "gekko"],
            "gekko_attention": {
                **kwargs["full_attention"],
                "v_head_dim": 4,
                "causal_conv_width": 4,
                "chunk_size": 4,
                "timenorm_num_groups": 3,
            },
        }
    )
    state = gekko.mamba2_cache_params
    assert gekko.full_attention_layer_ids == [0]
    assert gekko.linear_layer_ids == [1, 2, 3]
    assert state.layers == [1, 2, 3]
    assert state.shape.conv == [(3, 16), (3, 8), (3, 8)] + [(8, 8)] * 4
    assert state.shape.temporal == (1, 1, 46)
    assert state.shape.disable_conv_window_dedup
    assert state.dtype.conv == torch.bfloat16
    assert state.dtype.temporal == torch.float32
    assert state.mamba_cache_per_req == ((48 + 24 + 24 + 4 * 64) * 2 + 46 * 4) * 3
    spec, actual = registry.get_linear_attn_config(gekko)
    assert actual is gekko
    assert spec.arch_names == ["XllmGekkoForCausalLM"]

    native_family = {
        **gekko.gekko_attention,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "v_head_dim": 64,
        "timenorm_num_groups": 32,
    }
    native_shape = module.XllmHybridConfig(
        **{
            **kwargs,
            "hidden_size": 1536,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "layer_kinds": gekko.layer_kinds,
            "full_attention": native_family,
            "gekko_attention": native_family,
        }
    )
    # The native 32 x 48 TDN layout retains a bounded partial scan block in
    # addition to AWM memory, its log normalizer, and current TDN statistics.
    assert native_shape.mamba2_cache_params.shape.temporal == (
        1,
        1,
        8 * 64 * 64 + 8 * 64 + 2 * 32 + 16450,
    )


@pytest.mark.parametrize("draft", [False, True])
def test_registered_backend_uses_returned_hf_config_for_layer_mapping(draft):
    class UnrelatedConfig:
        pass

    class LinearBackend:
        def __init__(self, runner):
            self.runner = runner

    class HybridBackend:
        def __init__(self, full, linear, layer_ids):
            self.full, self.linear, self.layer_ids = full, linear, layer_ids

    registered_config = SimpleNamespace(full_attention_layer_ids=[0, 4, 8])
    outer_config = SimpleNamespace(text_config=registered_config)
    # ModelConfig intentionally has no full_attention_layer_ids: this is the
    # actual generic-registry case for Gekko and non-SWA registered hybrids.
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=outer_config, hf_text_config=registered_config
        ),
        use_mla_backend=False,
        is_draft_worker=draft,
    )
    spec = SimpleNamespace(
        backend_class_name="linear", hybrid_backend_class_name="hybrid"
    )
    modules = {
        "sglang.srt.arg_groups.overrides": _module(
            attention_backends_of=lambda x: None, resolved_view=lambda x: x
        ),
        "sglang.srt.configs.hybrid_arch": _module(
            **{
                name: lambda config: None
                for name in (
                    "glm5_next_config",
                    "hybrid_gdn_config",
                    "hybrid_lightning_config",
                    "kimi_linear_config",
                    "mamba2_config",
                )
            },
            mambaish_config=lambda config: registered_config,
        ),
        "sglang.srt.configs.linear_attn_model_registry": _module(
            get_linear_attn_config=lambda config: (spec, registered_config),
            import_backend_class=lambda name: (
                LinearBackend if name == "linear" else HybridBackend
            ),
        ),
        "sglang.srt.runtime_context": _module(
            get_parallel=lambda: SimpleNamespace(),
            get_platform=lambda: SimpleNamespace(),
            get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
        ),
        "sglang.srt.utils": _module(
            get_device_capability=lambda: (9, 0),
            is_hip=lambda: False,
            is_musa=lambda: False,
            is_npu=lambda: False,
            is_blackwell=lambda: False,
            is_xpu=lambda: False,
        ),
        "sglang.srt.configs.dots3": _module(Dots3Config=UnrelatedConfig),
        "sglang.srt.configs.model_config": _module(
            is_minimax_sparse=lambda config: False
        ),
        "sglang.srt.configs.inkling": _module(
            InklingMMConfig=UnrelatedConfig, InklingModelConfig=UnrelatedConfig
        ),
        "sglang.kernels.ops.attention.fla.utils": _module(
            check_environments=lambda: None
        ),
        "sglang.srt.layers.attention.linear.kda_backend": _module(
            KDAAttnBackend=LinearBackend
        ),
        "sglang.srt.layers.attention.linear.lightning_backend": _module(
            LightningAttentionBackend=LinearBackend
        ),
        "sglang.srt.layers.attention.linear.utils": _module(
            resolve_linear_attn_backends=lambda **kwargs: None
        ),
        "sglang.srt.layers.attention.hybrid_linear_attn_backend": _module(
            HybridLinearAttnBackend=HybridBackend, Mamba2AttnBackend=LinearBackend
        ),
        "sglang.srt.layers.attention.linear.gdn_backend": _module(
            GDNAttnBackend=LinearBackend,
            flashinfer_gdn_prefill_default=lambda runner: None,
        ),
    }
    registry = _load(
        "_xllm_attention_registry_cpu",
        _ROOT / "python/sglang/srt/layers/attention/attention_registry.py",
        modules,
    )
    full = object()
    with patch.dict(sys.modules, modules):
        result = registry.attn_backend_wrapper(runner, full)
    assert result.full is full
    assert result.linear.runner is runner
    assert result.layer_ids == ([0] if draft else [0, 4, 8])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
