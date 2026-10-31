"""CPU semantic tests; these do not claim native learned BF16 parity."""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

# Exercise the torch-only core without importing SGLang's optional GPU stack.
_MODELS = Path(__file__).resolve().parents[4] / "python/sglang/srt/models"
_PACKAGE = "_xllm_models_cpu"
if _PACKAGE not in sys.modules:
    _package = types.ModuleType(_PACKAGE)
    _package.__path__ = [str(_MODELS)]
    sys.modules[_PACKAGE] = _package
reference = importlib.import_module(f"{_PACKAGE}.xllm_reference")
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _MODELS.parents[1] / "test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def tiny_config(hybrid=True):
    config = {
        "arch": "hybrid_inkling" if hybrid else "transformer",
        "model_dim": 12,
        "ffn_hidden_dim": 23,
        "vocab_size": 29,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 4,  # Deliberately not model_dim // num_heads.
        "rope_head_dim": 4,
        "rope_base": 1000000.0,
        "max_position_embeddings": 131072,
        "rmsnorm_eps": 1e-6,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16.0,
            "beta_fast": 128.0,
            "beta_slow": 4.0,
            "original_max_position_embeddings": None,
        },
    }
    if hybrid:
        config.update(
            {
                "layer_kinds": ["full", "inkling", "inkling", "inkling"],
                "full_attention": {},
                "inkling_attention": {
                    "rope_base": 500000.0,
                    "rope_scaling": None,
                    "local_attention_window": 5,
                    "causal_conv_width": 4,
                    "causal_conv_weight_normalization": False,
                },
            }
        )
    return config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1409)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("hybrid", [False, True])
@torch.no_grad()
def test_prefill_arbitrary_pieces_and_decode(hybrid):
    model = reference.XllmReferenceModel(tiny_config(hybrid)).eval()
    tokens = torch.randint(0, 29, (19,))
    full = model(tokens)
    for partition in ([19], [1] * 19, [3, 1, 7, 2, 6], [4, 5, 5, 5]):
        state, outputs, offset = model.new_cache(), [], 0
        for count in partition:
            outputs.append(model(tokens[offset : offset + count], state))
            offset += count
        torch.testing.assert_close(torch.cat(outputs), full, atol=1e-6, rtol=1e-5)
        assert state.position == len(tokens)
        for kind, layer in zip(model.layer_kinds, state.layers):
            assert len(layer.key) == (len(tokens) if kind == "full" else 4)
            assert len(layer.conv) == (0 if kind == "full" else 4)
            for history in layer.conv.values():
                assert history.shape[0] == 3


@torch.no_grad()
def test_interleaved_reordered_requests_and_reuse():
    model = reference.XllmReferenceModel(tiny_config()).eval()
    a, b = torch.randint(0, 29, (17,)), torch.randint(0, 29, (17,))
    expected_a, expected_b = model(a), model(b)
    state_a, state_b = model.new_cache(), model.new_cache()
    actual_a = [model(a[:3], state_a)]
    actual_b = [model(b[:8], state_b)]
    actual_a.append(model(a[3:8], state_a))
    reordered = model(torch.stack((b[8:12], a[8:12])), [state_b, state_a])
    actual_b.append(reordered[0])
    actual_a.append(reordered[1])
    actual_a.append(model(a[12:], state_a))
    actual_b.append(model(b[12:], state_b))
    torch.testing.assert_close(torch.cat(actual_a), expected_a, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.cat(actual_b), expected_b, atol=1e-6, rtol=1e-5)
    state_a.reset()
    torch.testing.assert_close(model(b, state_a), expected_b)
    with pytest.raises(ValueError, match="independent"):
        model(torch.stack((a[:2], b[:2])), [state_a, state_a])


@torch.no_grad()
def test_snapshot_has_no_tensor_aliases():
    model = reference.XllmReferenceModel(tiny_config()).eval()
    tokens = torch.randint(0, 29, (12,))
    state = model.new_cache()
    model(tokens[:7], state)
    snapshot = state.clone()
    for source, saved in zip(state.layers, snapshot.layers):
        assert source.key.data_ptr() != saved.key.data_ptr()
        for name in source.conv:
            assert source.conv[name].data_ptr() != saved.conv[name].data_ptr()
    expected = model(tokens[7:], snapshot)
    model(torch.tensor([3, 4]), state)
    assert snapshot.position == 12
    snapshot = model.new_cache()
    model(tokens[:7], snapshot)
    torch.testing.assert_close(model(tokens[7:], snapshot), expected)


@torch.no_grad()
def test_selected_logits_and_causal_prefix():
    model = reference.XllmReferenceModel(tiny_config()).eval()
    tokens = torch.randint(0, 29, (14,))
    full = model(tokens)
    selected = [0, 4, 7, 13]
    torch.testing.assert_close(
        model(tokens, selected_positions=selected), full[selected]
    )
    for length in (1, 4, 5, 6, 13):
        torch.testing.assert_close(
            model(tokens[:length])[-1], full[length - 1], atol=1e-6, rtol=1e-5
        )


@torch.no_grad()
def test_native_parameter_names_and_strict_load():
    model = reference.XllmReferenceModel(tiny_config())
    keys = set(model.state_dict())
    assert model.layers[0].attention.wq.weight.shape == (16, 12)
    assert "layers.0.attention.norm.weight" in keys
    assert "layers.1.attention.k_shortconv.weight" in keys
    assert "layers.1.attention.v_shortconv.weight" in keys
    assert "layers.1.attention.out_shortconv.weight" in keys
    assert "layers.1.nffn.out_shortconv.weight" in keys
    assert "output.final_norm.weight" in keys
    assert "output.output.weight" in keys
    assert "layers.0.attention.rope.inv_freq" not in keys
    restored = reference.XllmReferenceModel(tiny_config())
    restored.load_state_dict(model.state_dict(), strict=True)
    damaged = dict(model.state_dict())
    damaged.pop("layers.1.nffn.out_shortconv.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(damaged, strict=True)


@torch.no_grad()
def test_shortconv_raw_taps_residual_and_history():
    conv = reference.ResidualShortConv(1, 4)
    conv.weight.copy_(torch.tensor([[1.0, 10.0, 100.0, 1000.0]]))
    tokens = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    expected = torch.tensor([[1001.0], [2102.0], [3213.0], [4325.0], [5437.0]])
    torch.testing.assert_close(conv(tokens), expected)
    state = reference.LayerState()
    actual = torch.cat(
        [conv(tokens[:2], state), conv(tokens[2:3], state), conv(tokens[3:], state)]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state.conv["conv"], tokens[-3:])


@torch.no_grad()
def test_offset_rmsnorm_bf16_operation_order():
    norm = reference.NativeRMSNorm(4, eps=1e-6).to(torch.bfloat16)
    norm.weight.copy_(torch.tensor([0.002, 0.004, -0.006, 0.013]))
    x = torch.tensor([[0.7, -1.3, 2.1, 4.0]], dtype=torch.bfloat16)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    native = (normalized * (norm.weight + 1).float()).to(x.dtype)
    folded_in_fp32 = (normalized * (norm.weight.float() + 1)).to(x.dtype)
    assert not torch.equal(native, folded_in_fp32)
    torch.testing.assert_close(norm(x), native, rtol=0, atol=0)


@torch.no_grad()
def test_rope_adjacent_pairs_and_yarn_amplitude():
    rope = reference.NativeRotaryEmbedding(
        4, rope_base=10000, max_position_embeddings=128
    )
    q = torch.tensor([[[1.0, 0.0, 2.0, 0.0]]])
    actual, _ = rope(q, q, torch.tensor([1]))
    expected = torch.tensor(
        [
            [
                [
                    torch.cos(torch.tensor(1.0)),
                    torch.sin(torch.tensor(1.0)),
                    2 * torch.cos(torch.tensor(0.01)),
                    2 * torch.sin(torch.tensor(0.01)),
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)
    yarn = reference.NativeRotaryEmbedding(4, rope_scaling={"factor": 16.0})
    rotated, _ = yarn(q, q, torch.tensor([0]))
    torch.testing.assert_close(rotated, q * (1 + 0.1 * torch.log(torch.tensor(16.0))))


@pytest.mark.parametrize(
    "bad",
    [
        {"head_dim": None},
        {"qknorm": True},
        {"apply_attn_gate": True},
        {"scale_emb": True},
        {"residual_func": "ortho"},
        {"head_dim": 0},
    ],
)
def test_unsupported_settings_fail(bad):
    config = tiny_config(False)
    config.update(bad)
    with pytest.raises(ValueError):
        reference.XllmReferenceModel(config)


def test_context_and_selection_bounds():
    config = tiny_config(False)
    config["max_position_embeddings"] = 8
    model = reference.XllmReferenceModel(config)
    with pytest.raises(ValueError, match="max_position"):
        model(torch.ones(9, dtype=torch.long))
    with pytest.raises(ValueError, match="Selected"):
        model(torch.ones(4, dtype=torch.long), selected_positions=[4])


@torch.no_grad()
def test_bounded_attention_scores_preserve_logits():
    model = reference.XllmReferenceModel(tiny_config()).eval()
    tokens = torch.randint(0, 29, (17,))
    expected = model(tokens)
    for layer in model.layers:
        layer.attention.max_score_elements = 4 * len(tokens) * 2
    torch.testing.assert_close(model(tokens), expected, atol=1e-6, rtol=1e-5)


def test_legacy_norm_epsilon_fallback():
    config = tiny_config(False)
    config["norm_eps"] = 0.03
    assert reference.XllmReferenceModel(config).output.final_norm.eps == 1e-6
    del config["rmsnorm_eps"]
    assert reference.XllmReferenceModel(config).output.final_norm.eps == 0.03


def test_dense_ignores_native_generic_bias_flag():
    # Native MultiheadAttention always constructs bias-free projections. The
    # original saved dense ModelConf nevertheless carries this Gekko default.
    config = tiny_config(False)
    config["apply_bias_term"] = True
    model = reference.XllmReferenceModel(config)
    assert not any(name.endswith(".bias") for name in model.state_dict())


def test_dense_null_value_head_dim_inherits_query_head_dim():
    config = tiny_config(False)
    config["v_head_dim"] = None
    model = reference.XllmReferenceModel(config)
    assert model.layers[0].attention.wv.out_features == 8
    config["v_head_dim"] = 2
    with pytest.raises(ValueError, match="value dimensions"):
        reference.XllmReferenceModel(config)


@torch.no_grad()
def test_gekko_backbone_uses_native_names_and_complete_recurrent_snapshot():
    config = tiny_config(False)
    config.update(
        {
            "arch": "hybrid_gekko",
            "num_layers": 2,
            "layer_kinds": ["full", "gekko"],
            "full_attention": {},
            "gekko_attention": {
                "chunk_size": 4,
                "timenorm_num_groups": 3,
                "apply_bias_term": True,
                "rope_scaling": None,
                "gekko_value_activation": "identity",
            },
        }
    )
    model = reference.XllmReferenceModel(config).eval()
    assert "layers.1.gda.wq.weight" in model.state_dict()
    assert "layers.1.gda.timenorm.prior_var" in model.state_dict()
    tokens = torch.randint(0, 29, (13,))
    expected = model(tokens)
    state = model.new_cache()
    parts = [model(tokens[:3], state), model(tokens[3:7], state)]
    saved = state.clone()
    parts.append(model(tokens[7:], state))
    torch.testing.assert_close(torch.cat(parts), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        model(tokens[7:], saved), expected[7:], atol=1e-6, rtol=1e-5
    )
    saved.layers[1].extra = None
    with pytest.raises(ValueError, match="recurrent state"):
        model(torch.tensor([1]), saved)


@pytest.mark.skipif(
    not os.environ.get("XLLM_NATIVE_SOURCE"),
    reason="Set XLLM_NATIVE_SOURCE to the pinned native checkout",
)
@torch.no_grad()
def test_rotary_against_read_only_native_module():
    source = (
        Path(os.environ["XLLM_NATIVE_SOURCE"])
        / "xllm/modules/rotary_positional_embedding.py"
    )
    spec = importlib.util.spec_from_file_location("_native_xllm_rotary_oracle", source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    scaling = tiny_config(False)["rope_scaling"]
    for settings in (None, scaling):
        ours = reference.NativeRotaryEmbedding(
            64, rope_base=1000000, rope_scaling=settings
        )
        oracle = native.RotaryEmbedding(64, 131072, 1000000, settings, 131072)
        positions = torch.tensor([0, 1, 4095, 4096, 8191, 8192, 65535, 131071])
        q = torch.randn(1, len(positions), 4, 64, dtype=torch.bfloat16)
        k = torch.randn(1, len(positions), 2, 64, dtype=torch.bfloat16)
        expected_q, expected_k = native.apply_rotary_embeddings(
            q, k, oracle.get_freqs_cis(0, 131072, q.device)[positions], False
        )
        actual_q, actual_k = ours(q[0], k[0], positions)
        torch.testing.assert_close(actual_q, expected_q[0], atol=0, rtol=0)
        torch.testing.assert_close(actual_k, expected_k[0], atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
