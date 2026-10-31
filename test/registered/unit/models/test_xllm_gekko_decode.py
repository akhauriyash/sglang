"""Whole-layer pooled Gekko parity, eager boundary handoff and CUDA graphs.

The oracle runs the functional Gekko recurrence with native SCA arithmetic.
Pool serialization here follows the documented layout independently of the
serving backend so a shared backend packing error cannot hide a mismatch.
"""

import importlib
import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "_xllm_gekko_decode_ci", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ci
_SPEC.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Pooled native Gekko requires NVIDIA SM90",
)


@cache
def _modules():
    prefix = "sglang.srt.models."
    return SimpleNamespace(
        reference=importlib.import_module(prefix + "xllm_gekko_reference"),
        sca=importlib.import_module(prefix + "xllm_sca"),
        decode=importlib.import_module(prefix + "xllm_gekko_decode"),
        tdn=importlib.import_module(prefix + "xllm_tdn"),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_metadata_refresh_tracks_positions_and_padding(batch_size):
    slots = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    seq_lens = torch.ones(batch_size, dtype=torch.int64, device="cuda")
    positions = torch.empty_like(slots)
    lengths = torch.empty_like(slots)
    prepare = _modules().decode.prepare_decode_metadata
    for end in (254, 255, 256, 4095, 4096, 4097, 8192, 8193, 130816):
        seq_lens.fill_(end)
        slots.copy_(torch.arange(batch_size, dtype=torch.int32, device="cuda"))
        if batch_size > 1:
            slots[-1] = -1
        prepare(slots, seq_lens, positions, lengths)
        retained = end - 1 if end <= 4096 else 4096 + (end - 1) % 4096
        expected_positions = [end - 1] * batch_size
        expected_lengths = [retained + 1] * batch_size
        if batch_size > 1:
            expected_positions[-1] = expected_lengths[-1] = 0
        assert positions.tolist() == expected_positions
        assert lengths.tolist() == expected_lengths


@pytest.fixture(scope="module")
@torch.no_grad()
def layer():
    modules = _modules()

    class NativeSCAReference(modules.reference.GekkoAttention):
        def _attend(self, q, k, v, positions, key_start):
            return modules.sca.native_sca_attention(q, k, v, self.head_dim**-0.5)

    torch.manual_seed(7903)
    result = (
        NativeSCAReference(
            {
                "model_dim": 1536,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 64,
                "v_head_dim": 64,
                "chunk_size": 4096,
                "timenorm_num_groups": 32,
                "causal_conv_width": 4,
                "gekko_value_activation": "identity",
                "apply_bias_term": True,
            }
        )
        .cuda()
        .eval()
    )
    for convolution in (result.q_conv, result.k_conv, result.v_conv):
        convolution.weight.normal_(0.1, 0.4)
    result.timenorm.weight.uniform_(-0.2, 0.2)
    result.timenorm.bias.uniform_(-0.1, 0.1)
    result.query_norm.weight.uniform_(-0.2, 0.2)
    result.key_norm.weight.uniform_(-0.2, 0.2)
    result.timenorm.prior_count.fill_(1031)
    result.timenorm.prior_mean.copy_(torch.linspace(-0.01, 0.01, 32, device="cuda"))
    result.timenorm.prior_var.fill_(0.01)
    for parameter in result.parameters():
        parameter.data = parameter.data.bfloat16()
    result.rotary.cache_inverse_frequencies = True
    return result


@pytest.fixture(scope="module")
@torch.no_grad()
def prefixes(layer):
    generator = torch.Generator(device="cuda").manual_seed(3991)
    hidden = torch.randn(
        8194, 1536, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    states = {}
    # One contiguous reference prefill, sampled before relevant boundaries.
    # These are real historical states, including nonzero AWM after8192.
    state = layer.initial_state(hidden)
    offset = 0
    for position in (1, 31, 254, 256, 4094, 4096, 8190, 8192):
        _, state = layer(hidden[offset:position], state)
        states[position] = state.clone()
        offset = position
    return hidden, states


def _make_pool(layer, slots=5):
    qdim = layer.num_heads * layer.head_dim
    kdim = layer.num_kv_heads * layer.head_dim
    vdim = layer.num_kv_heads * layer.v_head_dim
    width, capacity = layer.q_conv.width - 1, 2 * layer.chunk_size
    shapes = [
        (width, qdim),
        (width, kdim),
        (width, vdim),
        (capacity, kdim),
        (capacity, vdim),
        (capacity, kdim),
        (capacity, kdim),
    ]
    temporal_size = (
        layer.num_kv_heads * layer.head_dim * layer.v_head_dim
        + kdim
        + 2 * layer.timenorm.num_groups
        + _modules().tdn.TDN_STATE_SIZE
    )
    return SimpleNamespace(
        conv=[
            torch.full((slots, *shape), torch.nan, device="cuda", dtype=torch.bfloat16)
            for shape in shapes
        ],
        temporal=torch.full((slots, 1, 1, temporal_size), torch.nan, device="cuda"),
    )


def _write(pool, slot, state):
    for destination, source in zip(
        pool.conv,
        (
            state.q_history,
            state.k_history,
            state.v_history,
            state.sca_k.flatten(1),
            state.values.flatten(1),
            state.feature_k.flatten(1),
            state.raw_k.flatten(1),
        ),
        strict=True,
    ):
        destination[slot, : len(source)].copy_(source)
    row, offset = pool.temporal[slot].flatten(), 0
    for value in (
        state.memory,
        state.log_normalizer,
        state.tdn_mean,
        state.tdn_var,
        state.tdn_packed,
    ):
        row[offset : offset + value.numel()].copy_(value.flatten())
        offset += value.numel()
    assert offset == row.numel()


def _read(layer, pool, slot, position):
    retained = position if position < 4096 else 4096 + position % 4096
    row = pool.temporal[slot].flatten()
    nk, dk, dv = layer.num_kv_heads, layer.head_dim, layer.v_head_dim
    matrix_end = nk * dk * dv
    normalizer_end = matrix_end + nk * dk
    return _modules().reference.GekkoState(
        position=position,
        tdn_count=layer.timenorm.prior_count + position,
        tdn_mean=row[normalizer_end : normalizer_end + 32],
        tdn_var=row[normalizer_end + 32 : normalizer_end + 64],
        q_history=pool.conv[0][slot],
        k_history=pool.conv[1][slot],
        v_history=pool.conv[2][slot],
        sca_k=pool.conv[3][slot, :retained].view(retained, nk, dk),
        values=pool.conv[4][slot, :retained].view(retained, nk, dv),
        feature_k=pool.conv[5][slot, :retained].view(retained, nk, dk),
        raw_k=pool.conv[6][slot, :retained].view(retained, nk, dk),
        memory=row[:matrix_end].view(nk, dk, dv).bfloat16(),
        log_normalizer=row[matrix_end:normalizer_end].view(nk, dk),
        tdn_packed=row[normalizer_end + 64 :],
    )


def _assert_pool(actual, expected):
    for name, first, second in [
        *[
            (f"conv[{i}]", a, b)
            for i, (a, b) in enumerate(zip(actual.conv, expected.conv))
        ],
        ("temporal", actual.temporal, expected.temporal),
    ]:
        # Compare all payload bytes, including inactive slots and unused
        # history rows that remain poisoned. Numeric NaN equality is weaker.
        assert torch.equal(first.view(torch.uint8), second.view(torch.uint8)), name


def _metadata(order, positions):
    slots = torch.tensor(order, device="cuda", dtype=torch.int32)
    absolute = [positions[slot] if slot >= 0 else 0 for slot in order]
    lengths = [
        (position + 1 if position < 4096 else 4097 + position % 4096)
        if slot >= 0
        else 0
        for slot, position in zip(order, absolute)
    ]
    return (
        slots,
        torch.tensor(absolute, device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
    )


@pytest.mark.parametrize("start", [254, 4094, 8190])
@torch.no_grad()
def test_pooled_layer_eager_boundary_handoff(layer, prefixes, start):
    hidden, states = prefixes
    state = states[start].clone()
    actual_pool, expected_pool = _make_pool(layer, 2), _make_pool(layer, 2)
    _write(actual_pool, 1, state)
    _write(expected_pool, 1, state)
    for position in range(start, start + 3):
        x = hidden[position : position + 1]
        expected, state = layer(x, state)
        _write(expected_pool, 1, state)
        if position % 256 == 255:
            actual, next_state = layer(x, _read(layer, actual_pool, 1, position))
            _write(actual_pool, 1, next_state)
        else:
            actual = _modules().decode.pooled_gekko_decode(
                layer, x, actual_pool, *_metadata([1], {1: position})
            )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        _assert_pool(actual_pool, expected_pool)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@torch.no_grad()
def test_pooled_layer_graph_reorder_padding_and_slot_reuse(layer, prefixes, batch_size):
    hidden, prefixes_by_position = prefixes
    positions = {slot: value for slot, value in enumerate((31, 256, 4096, 8192))}
    states = {
        slot: prefixes_by_position[position].clone()
        for slot, position in positions.items()
    }
    actual_pool, expected_pool = _make_pool(layer), _make_pool(layer)
    for slot, state in states.items():
        _write(actual_pool, slot, state)
        _write(expected_pool, slot, state)
    x = torch.full((batch_size, 1536), torch.nan, device="cuda", dtype=torch.bfloat16)
    slots, device_positions, lengths = _metadata([-1] * batch_size, positions)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _modules().decode.pooled_gekko_decode(
            layer, x, actual_pool, slots, device_positions, lengths
        )
    torch.cuda.current_stream().wait_stream(stream)
    _assert_pool(actual_pool, expected_pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _modules().decode.pooled_gekko_decode(
            layer, x, actual_pool, slots, device_positions, lengths
        )
    # Capture and warmup use invalid slots exclusively. Neither may modify
    # any real request, even though all real pool slots already contain state.
    _assert_pool(actual_pool, expected_pool)
    initial_order = list(range(batch_size))
    orders = [initial_order, list(reversed(initial_order)), initial_order.copy()]
    if batch_size > 1:
        orders[1][0] = -1
    for index, order in enumerate(orders):
        if index == 2:
            # Abort/reuse slot zero with a different valid prefill state.
            states[0] = prefixes_by_position[1].clone()
            positions[0] = 1
            _write(actual_pool, 0, states[0])
            _write(expected_pool, 0, states[0])
        current_slots, current_positions, current_lengths = _metadata(order, positions)
        slots.copy_(current_slots)
        device_positions.copy_(current_positions)
        lengths.copy_(current_lengths)
        expected_outputs = []
        for row, slot in enumerate(order):
            if slot < 0:
                x[row].fill_(torch.nan)
                expected_outputs.append(torch.zeros_like(x[row]))
                continue
            x[row].copy_(hidden[positions[slot]])
            expected, states[slot] = layer(x[row : row + 1], states[slot])
            expected_outputs.append(expected[0])
            _write(expected_pool, slot, states[slot])
        graph.replay()
        torch.testing.assert_close(
            output, torch.stack(expected_outputs), atol=0, rtol=0
        )
        _assert_pool(actual_pool, expected_pool)
        for slot in order:
            if slot >= 0:
                positions[slot] += 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
