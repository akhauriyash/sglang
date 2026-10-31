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

"""CPU mathematical/state qualification; these do not qualify native BF16 parity."""

import importlib
import importlib.util
import math
import sys
import types
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

# Keep the operator tests runnable with only torch and pytest, without importing
# SGLang's optional serving dependencies from its package-level initializer.
_root = Path(__file__).resolve().parents[4]
_package = sys.modules.setdefault(
    "_xllm_models_cpu", types.ModuleType("_xllm_models_cpu")
)
_package.__path__ = [str(_root / "python/sglang/srt/models")]
_gekko = importlib.import_module("_xllm_models_cpu.xllm_gekko_reference")
GekkoAttention = _gekko.GekkoAttention
NormalizedShortConv = _gekko.NormalizedShortConv
TimestepDecayNorm = _gekko.TimestepDecayNorm
adaptive_working_memory_update = _gekko.adaptive_working_memory_update
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _root / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(**overrides):
    torch.manual_seed(9811)
    config = {
        "model_dim": 12,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 4,  # Deliberately not model_dim // num_heads.
        "v_head_dim": 4,
        "chunk_size": 4,
        "timenorm_num_groups": 3,
        "apply_bias_term": True,
        "gekko_value_activation": "identity",
    }
    config.update(overrides)
    model = GekkoAttention(config).eval()
    with torch.no_grad():
        # Nonzero learned offsets and asymmetric taps catch initialization-only
        # mistakes that zero-offset or identity-convolution fixtures would miss.
        for conv in (model.q_conv, model.k_conv, model.v_conv):
            conv.weight.normal_(mean=0.1, std=0.4)
        model.query_norm.weight.uniform_(-0.3, 0.4)
        model.key_norm.weight.uniform_(-0.2, 0.6)
        model.timenorm.weight.uniform_(-0.2, 0.2)
        model.timenorm.bias.uniform_(-0.2, 0.2)
        model.timenorm.prior_count.fill_(3)
        model.timenorm.prior_mean.fill_(0.002)
        model.timenorm.prior_var.fill_(0.0003)
    return model


def assert_state_close(actual, expected, atol=2e-6):
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=atol, rtol=2e-5)
        else:
            assert a == b, field.name


def manual_tdn(norm, x):
    """Scalar-time, FP64 implementation independent of the streaming module."""
    count = int(norm.prior_count)
    mean = norm.prior_mean.to(x.dtype).double()
    var = norm.prior_var.to(x.dtype).double()
    outputs = []
    for token in x.double():
        groups = token.reshape(norm.num_groups, -1)
        mu = groups.mean(-1)
        variance = ((groups - mu[:, None]) ** 2).mean(-1)
        count += 1
        mean = norm.beta1 * mean + (1 - norm.beta1) * mu
        var = norm.beta2 * var + (1 - norm.beta2) * variance
        normalized = (groups - mean[:, None] / (1 - norm.beta1**count)) / (
            var[:, None] / (1 - norm.beta2**count) + norm.eps
        ).sqrt()
        outputs.append(
            normalized.flatten() * (1 + norm.weight).double() + norm.bias.double()
        )
    return torch.stack(outputs).to(x.dtype), count, mean.float(), var.float()


def manual_conv(conv, x):
    taps = conv.weight.float().softmax(-1).to(x.dtype)
    result = torch.zeros_like(x)
    for t in range(len(x)):
        for lag in range(conv.width):
            if t >= lag:
                result[t] += taps[:, conv.width - 1 - lag] * x[t - lag]
    return result


def manual_rms(x, groups, eps, weight=None):
    grouped = x.float().view(len(x), groups, -1)
    normalized = (
        grouped / (grouped.square().mean(-1, keepdim=True) + eps).sqrt()
    ).flatten(1)
    if weight is not None:
        normalized *= (1 + weight).float()
    return normalized.to(x.dtype)


@torch.no_grad()
def full_mask_oracle(model, x):
    """Independent unbounded full-mask/block oracle, without GekkoState.

    The deliberately slow implementation expands every key and explicitly
    constructs the architectural mask. It does not reuse the streaming AWM
    update helper, projection/convolution histories, or attention implementation.
    """
    z = manual_tdn(model.timenorm, x)[0]
    qraw = manual_conv(model.q_conv, F.linear(z, model.wq.weight))
    kraw = manual_conv(model.k_conv, F.linear(z, model.wk.weight))
    vraw = manual_conv(model.v_conv, F.linear(z, model.wv.weight, model.wv.bias))
    if model.value_activation == "silu":
        vraw = F.silu(vraw)
    q = manual_rms(qraw, model.num_heads, model.norm_eps, model.query_norm.weight).view(
        len(x), model.num_heads, model.head_dim
    )
    k = manual_rms(
        kraw, model.num_kv_heads, model.norm_eps, model.key_norm.weight
    ).view(len(x), model.num_kv_heads, model.head_dim)
    q, k = model.rotary(q, k, torch.arange(len(x)))
    values = vraw.view(len(x), model.num_kv_heads, model.v_head_dim)
    groups = model.num_heads // model.num_kv_heads
    keys = k.repeat_interleave(groups, 1)
    expanded_values = values.repeat_interleave(groups, 1)
    q_positions = torch.arange(len(x))[:, None]
    k_positions = torch.arange(len(x))[None, :]
    earliest = ((q_positions // model.chunk_size - 1) * model.chunk_size).clamp_min(0)
    mask = (k_positions <= q_positions) & (k_positions >= earliest)
    scores = torch.einsum("thd,shd->hts", q, keys) / math.sqrt(model.head_dim)
    sca = torch.einsum(
        "hts,shv->thv",
        scores.masked_fill(~mask, -math.inf).softmax(-1),
        expanded_values,
    )
    aq = qraw.view(len(x), model.num_heads, model.head_dim).softmax(-1)
    qk = kraw.view(len(x), model.num_kv_heads, model.head_dim).softmax(-1)
    kk = kraw.view(len(x), model.num_kv_heads, model.head_dim)
    memory = torch.zeros(model.num_kv_heads, model.head_dim, model.v_head_dim)
    normalizer = torch.full((model.num_kv_heads, model.head_dim), -math.inf)
    awm = torch.zeros_like(sca)
    for start in range(0, len(x), model.chunk_size):
        end = min(start + model.chunk_size, len(x))
        for h in range(model.num_heads):
            awm[start:end, h] = aq[start:end, h] @ memory[h // groups]
        if start:
            # The read above must precede this update, including when a whole
            # architectural chunk is emitted in a single serving call.
            old = slice(start - model.chunk_size, start)
            next_memory = torch.empty_like(memory)
            for h in range(model.num_kv_heads):
                chunk_lse = kk[old, h].logsumexp(0)
                next_normalizer = torch.logaddexp(normalizer[h], chunk_lse)
                ratio = (chunk_lse - next_normalizer).exp()
                prediction = qk[old, h] @ memory[h]
                chunk_update = kk[old, h].softmax(0).T @ (values[old, h] - prediction)
                next_memory[h] = memory[h] + ratio[:, None] * (chunk_update - memory[h])
                normalizer[h] = next_normalizer
            memory = next_memory
    combined = manual_rms((sca + awm).flatten(1), model.num_heads, model.norm_eps)
    gate = F.softplus(F.linear(z, model.wr.weight, model.wr.bias), beta=math.log(2))
    return F.linear(combined * gate, model.wo.weight)


def test_native_parameter_names_and_resolved_family_settings():
    model = make_model(num_heads=2, gekko_attention={"num_heads": 4, "head_dim": 4})
    state = model.state_dict()
    assert model.wq.weight.shape == (16, 12)
    assert state["wv.bias"].shape == (8,)
    assert state["wr.bias"].shape == (16,)
    assert state["query_norm.weight"].shape == (16,)
    assert state["key_norm.weight"].shape == (8,)
    assert state["timenorm.prior_count"].dtype == torch.int64
    assert not any("attn_norm" in key or "rotary" in key for key in state)


def test_normalized_convolution_history_and_bias_before_convolution():
    conv = NormalizedShortConv(2)
    with torch.no_grad():
        conv.weight[:, -1] = math.log(27)
    projected = torch.tensor([[2.0, -3.0]]).expand(5, -1).clone()
    result, history = conv(projected, torch.zeros(3, 2))
    expected = torch.tensor([0.9, 28 / 30, 29 / 30, 1, 1])[:, None] * projected
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(history, projected[-3:])
    first, carry = conv(projected[:2], torch.zeros(3, 2))
    last, _ = conv(projected[2:], carry)
    torch.testing.assert_close(torch.cat((first, last)), result)


def test_tdn_uses_population_variance_priors_and_offset_affine():
    model = make_model()
    x = torch.randn(9, 12)
    state = model.initial_state(x)
    expected, count, mean, variance = manual_tdn(model.timenorm, x)
    actual, n, m, v = model.timenorm(x, state.tdn_count, state.tdn_mean, state.tdn_var)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-6)
    assert int(n) == count == 12
    torch.testing.assert_close(m, mean)
    torch.testing.assert_close(v, variance)
    assert m.dtype == v.dtype == torch.float32
    assert int(model.timenorm.prior_count) == 3
    state.tdn_mean.fill_(99)
    assert torch.all(model.timenorm.prior_mean != 99)


def test_tdn_tracks_within_token_variance_not_variance_across_time():
    norm = TimestepDecayNorm(4, 2, beta1=0.5, beta2=0.5)
    x = torch.tensor([[1.0, 3.0, 2.0, 6.0], [11.0, 13.0, 12.0, 16.0]])
    _, count, _, variance = norm(x, torch.tensor(0), torch.zeros(2), torch.zeros(2))
    assert int(count) == 2
    torch.testing.assert_close(variance, torch.tensor([0.75, 3.0]))


def test_initial_tdn_priors_round_to_bf16_without_changing_checkpoint_buffers():
    model = make_model()
    model.timenorm.prior_mean.copy_(torch.tensor([0.00312345, -0.112345, 0.231234]))
    model.timenorm.prior_var.copy_(torch.tensor([0.004321, 0.017123, 0.213456]))
    saved_mean = model.timenorm.prior_mean.clone()
    saved_var = model.timenorm.prior_var.clone()
    state = model.initial_state(torch.zeros(1, 12, dtype=torch.bfloat16))
    assert state.tdn_mean.dtype == state.tdn_var.dtype == torch.float32
    assert state.tdn_count.dtype == torch.int64
    for actual, saved in ((state.tdn_mean, saved_mean), (state.tdn_var, saved_var)):
        torch.testing.assert_close(actual, saved.bfloat16().float(), atol=0, rtol=0)
        assert not torch.equal(actual, saved)
    # Request-local rounding must not destroy the original FP32 checkpoint
    # values or let state updates mutate the shared model buffers.
    state.tdn_mean.fill_(99)
    state.tdn_var.fill_(99)
    torch.testing.assert_close(model.timenorm.prior_mean, saved_mean, atol=0, rtol=0)
    torch.testing.assert_close(model.timenorm.prior_var, saved_var, atol=0, rtol=0)


def test_awm_update_matches_explicit_feature_and_time_axes():
    torch.manual_seed(11)
    memory = torch.randn(2, 3, 4)
    log_norm = torch.randn(2, 3)
    raw_keys = torch.randn(5, 2, 3)
    feature_keys = raw_keys.softmax(-1)
    values = torch.randn(5, 2, 4)
    new_memory, new_log_norm = adaptive_working_memory_update(
        memory, log_norm, feature_keys, raw_keys, values
    )
    expected = torch.empty_like(memory)
    expected_log = torch.empty_like(log_norm)
    for h in range(2):
        for d in range(3):
            lse = raw_keys[:, h, d].logsumexp(0)
            expected_log[h, d] = torch.logaddexp(log_norm[h, d], lse)
            rho = (lse - expected_log[h, d]).exp()
            update = torch.zeros(4)
            for t in range(5):
                prediction = sum(feature_keys[t, h, j] * memory[h, j] for j in range(3))
                update += (raw_keys[t, h, d] - lse).exp() * (values[t, h] - prediction)
            expected[h, d] = memory[h, d] + rho * (update - memory[h, d])
    torch.testing.assert_close(new_memory, expected, atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(new_log_norm, expected_log)


@pytest.mark.parametrize("length", [1, 3, 4, 5, 7, 8, 9, 12, 15])
def test_streaming_forward_matches_independent_full_mask_oracle(length):
    model = make_model()
    x = torch.randn(length, 12)
    actual, _ = model(x)
    expected = full_mask_oracle(model, x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "parts", [[15], [1] * 15, [3, 2, 6, 4], [4, 4, 4, 3], [7, 1, 1, 6]]
)
def test_prefill_partitions_and_one_token_decode(parts):
    model = make_model()
    x = torch.randn(sum(parts), 12)
    expected, expected_state = model(x)
    offset, state, output = 0, None, []
    for length in parts:
        y, state = model(x[offset : offset + length], state)
        output.append(y)
        offset += length
    torch.testing.assert_close(torch.cat(output), expected, atol=2e-6, rtol=2e-5)
    assert_state_close(state, expected_state)


def test_awm_is_empty_for_first_two_emitted_chunks_and_updates_afterwards():
    model = make_model()
    x = torch.randn(12, 12)
    _, first = model(x[:4])
    assert first.memory.count_nonzero() == 0
    assert first.log_normalizer.isneginf().all()
    _, before_second_end = model(x[4:7], first)
    assert before_second_end.memory.count_nonzero() == 0
    assert len(before_second_end.sca_k) == 7
    _, second = model(x[7:8], before_second_end)
    expected_memory, expected_norm = adaptive_working_memory_update(
        first.memory, first.log_normalizer, first.feature_k, first.raw_k, first.values
    )
    torch.testing.assert_close(second.memory, expected_memory)
    torch.testing.assert_close(second.log_normalizer, expected_norm)
    assert len(second.sca_k) == 4
    _, third_prefix = model(x[8:11], second)
    torch.testing.assert_close(third_prefix.memory, second.memory)
    _, third = model(x[11:], third_prefix)
    assert third.position == 12
    assert not torch.equal(third.memory, second.memory)
    assert first.memory.count_nonzero() == 0  # Input snapshots remain immutable.


def test_interleaved_requests_clone_and_fresh_slot_are_isolated():
    model = make_model()
    a, b = torch.randn(15, 12), torch.randn(9, 12)
    expected_a, final_a = model(a)
    expected_b, final_b = model(b)
    a1, state_a = model(a[:3])
    b1, state_b = model(b[:5])
    saved_a = state_a.clone()
    a2, state_a = model(a[3:10], state_a)
    b2, state_b = model(b[5:], state_b)
    a3, state_a = model(a[10:], state_a)
    torch.testing.assert_close(
        torch.cat((a1, a2, a3)), expected_a, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(torch.cat((b1, b2)), expected_b, atol=2e-6, rtol=2e-5)
    assert_state_close(state_a, final_a)
    assert_state_close(state_b, final_b)
    branch, _ = model(a[3:], saved_a)
    torch.testing.assert_close(branch, expected_a[3:], atol=2e-6, rtol=2e-5)
    fresh, _ = model(b)
    torch.testing.assert_close(fresh, expected_b, atol=0, rtol=0)
    saved_a.memory.fill_(123)
    assert torch.all(state_a.memory != 123)


def test_state_dtype_and_empty_call():
    model = make_model().bfloat16()
    x = torch.randn(5, 12).bfloat16()
    _, state = model(x)
    assert state.memory.dtype == state.q_history.dtype == torch.bfloat16
    assert state.tdn_mean.dtype == state.tdn_var.dtype == torch.float32
    assert state.log_normalizer.dtype == torch.float32
    assert state.tdn_count.dtype == torch.int64
    empty, same = model(x[:0], state)
    assert empty.shape == (0, 12)
    assert same is state


def test_packed_inputs_and_wrong_state_position_fail_explicitly():
    model = make_model()
    x = torch.randn(2, 12)
    with pytest.raises(ValueError, match="Packed/reset"):
        model(x, bos_mask=torch.tensor([True, False]))
    with pytest.raises(ValueError, match="one request"):
        model(x[None])
    _, state = model(x)
    state.position = 9
    with pytest.raises(ValueError, match="retention"):
        model(x, state)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
