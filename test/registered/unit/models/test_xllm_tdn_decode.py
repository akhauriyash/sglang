"""Pooled single-token TDN: native arithmetic, state ownership and graph replay."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_xllm_tdn_decode_helpers", Path(__file__).with_name("test_xllm_tdn.py")
)
helpers = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = helpers
_SPEC.loader.exec_module(helpers)
tdn = helpers.tdn
register_cuda_ci = helpers.register_cuda_ci
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
pytestmark = helpers.pytestmark

_OFFSETS = {"mean_offset": 17, "var_offset": 49, "packed_offset": 101}
_STRIDE = _OFFSETS["packed_offset"] + tdn.TDN_STATE_SIZE + 13


def _pool():
    return torch.full((7, 1, 1, _STRIDE), torch.nan, device="cuda")


def _put_state(pool, slot, state, mean=None, var=None):
    row = pool[slot].flatten()
    row[_OFFSETS["packed_offset"] : -13].copy_(state.storage)
    if mean is not None:
        row[_OFFSETS["mean_offset"] : _OFFSETS["mean_offset"] + 32].copy_(mean)
        row[_OFFSETS["var_offset"] : _OFFSETS["var_offset"] + 32].copy_(var)


def _prepared(position, nonzero):
    inputs = helpers._make_inputs(position + 1, nonzero)
    state = helpers._initial(inputs, helpers._PARAMETERS)
    if position:
        _, _, _, state = tdn.native_tdn_forward(
            inputs["x"][:position],
            inputs["weight"],
            inputs["bias"],
            state,
            **helpers._PARAMETERS,
        )
    # A new row must overwrite its old payload; later unused entries must
    # remain masked, even when a slot contains poisoned data from a prior use.
    state.partial_mean[:, position % 256 :].fill_(torch.nan)
    state.partial_var[:, position % 256 :].fill_(torch.nan)
    return inputs, state


def _decode(x, inputs, pool, slots, positions, **kwargs):
    return tdn.native_tdn_decode(
        x,
        inputs["weight"],
        inputs["bias"],
        pool,
        slots,
        positions,
        **(_OFFSETS | kwargs),
        **helpers._PARAMETERS,
    )


@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("position", [0, 1, 31, 32, 173, 254, 256, 4094, 4096, 8192])
@torch.no_grad()
def test_pooled_decode_preserves_native_output_and_state(position, nonzero):
    inputs, state = _prepared(position, nonzero)
    pool = _pool()
    _put_state(pool, 3, state)
    expected_pool = pool.clone()
    x = inputs["x"][position:]
    expected, mean, var, state = tdn.native_tdn_forward(
        x, inputs["weight"], inputs["bias"], state, **helpers._PARAMETERS
    )
    _put_state(expected_pool, 3, state, mean, var)
    actual = _decode(
        x,
        inputs,
        pool,
        torch.tensor([3], device="cuda", dtype=torch.int32),
        torch.tensor([position], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(pool.view(torch.int32), expected_pool.view(torch.int32))


@torch.no_grad()
def test_pooled_decode_graph_reorders_slots_and_masks_padding():
    positions = [0, 31, 253]
    states, token_inputs = [], []
    initial_pool = _pool()
    for slot, position in enumerate(positions):
        inputs, state = _prepared(position, nonzero=slot != 0)
        states.append(state)
        token_inputs.append(inputs["x"][-1].clone())
        _put_state(initial_pool, slot, state)
    # One shared affine transform applies to every request; saved temporal
    # moments are independent of those affine parameters.
    pool = initial_pool.clone()
    x = torch.stack(
        (token_inputs[0], token_inputs[1], token_inputs[2], token_inputs[0])
    )
    slots = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
    device_positions = torch.tensor(
        positions + [2**40], device="cuda", dtype=torch.int64
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _decode(x, inputs, pool, slots, device_positions)
    torch.cuda.current_stream().wait_stream(stream)
    pool.copy_(initial_pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _decode(x, inputs, pool, slots, device_positions)
    # Capture advances state, so reset before serving. This is the same pool
    # address; all future metadata changes are device-buffer updates only.
    pool.copy_(initial_pool)
    expected_pool = initial_pool.clone()
    for order in ([0, 1, 2, -1], [2, -1, 0, 1], [1, 0, -1, 2]):
        expected_outputs = []
        for row, slot in enumerate(order):
            if slot < 0:
                x[row].fill_(torch.nan)
                expected_outputs.append(torch.zeros(1536, device="cuda", dtype=x.dtype))
                continue
            if positions[slot] % 256 == 255:
                # Cross a real boundary eagerly and serialize its committed
                # block statistics back into the exact same pooled layout.
                _, mean, var, states[slot] = tdn.native_tdn_forward(
                    token_inputs[slot][None],
                    inputs["weight"],
                    inputs["bias"],
                    states[slot],
                    **helpers._PARAMETERS,
                )
                positions[slot] += 1
                _put_state(pool, slot, states[slot], mean, var)
                _put_state(expected_pool, slot, states[slot], mean, var)
            x[row].copy_(token_inputs[slot])
            expected, mean, var, states[slot] = tdn.native_tdn_forward(
                token_inputs[slot][None],
                inputs["weight"],
                inputs["bias"],
                states[slot],
                **helpers._PARAMETERS,
            )
            expected_outputs.append(expected[0])
            _put_state(expected_pool, slot, states[slot], mean, var)
        slots.copy_(torch.tensor(order, device="cuda", dtype=slots.dtype))
        device_positions.copy_(
            torch.tensor(
                [positions[slot] if slot >= 0 else -1 for slot in order],
                device="cuda",
                dtype=device_positions.dtype,
            )
        )
        graph.replay()
        torch.testing.assert_close(
            output, torch.stack(expected_outputs), atol=0, rtol=0
        )
        assert torch.equal(pool.view(torch.int32), expected_pool.view(torch.int32))
        for slot in order:
            if slot >= 0:
                positions[slot] += 1


@pytest.mark.parametrize("slot,position", [(7, 0), (0, -1), (0, 255), (0, 511)])
@torch.no_grad()
def test_pooled_decode_invalid_real_metadata_is_visible_without_state_writes(
    slot, position
):
    inputs = helpers._make_inputs(1)
    pool = _pool()
    before = pool.view(torch.int32).clone()
    output = _decode(
        inputs["x"],
        inputs,
        pool,
        torch.tensor([slot], device="cuda", dtype=torch.int32),
        torch.tensor([position], device="cuda", dtype=torch.int32),
    )
    assert torch.isnan(output).all()
    assert torch.equal(pool.view(torch.int32), before)


@pytest.mark.parametrize("invalid", ["overlap", "offset", "pool", "slots", "positions"])
def test_pooled_decode_rejects_invalid_tensor_layout(invalid):
    inputs = helpers._make_inputs(1)
    pool, kwargs = _pool(), {}
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    positions = slots.clone()
    if invalid == "overlap":
        kwargs["mean_offset"] = _OFFSETS["packed_offset"]
    elif invalid == "offset":
        kwargs["packed_offset"] = _STRIDE
    elif invalid == "pool":
        pool = pool.bfloat16()
    elif invalid == "slots":
        slots = slots.float()
    else:
        positions = positions[:0]
    with pytest.raises(ValueError):
        _decode(inputs["x"], inputs, pool, slots, positions, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
