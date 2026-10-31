"""Real CUDA TDN dispatch through Gekko and its request-pool serialization.

The allocator itself is stubbed by the shared pool test helper. Gekko's actual
operator, packed scan kernels, and pool read/write code run without substitution.
Native kernel arithmetic has its independent oracle tests in test_xllm_tdn.py.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_xllm_gekko_cuda_pool_helpers",
    Path(__file__).with_name("test_xllm_gekko_backend.py"),
)
pool_test = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pool_test
_SPEC.loader.exec_module(pool_test)
reference = pool_test.reference
tdn = importlib.import_module(reference.__package__ + ".xllm_tdn")

register_cuda_ci = pool_test._ci.register_cuda_ci
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native Gekko TDN requires NVIDIA CUDA",
)


@pytest.mark.parametrize("pieces", [(31, 225, 3), (1, 255, 3)])
@torch.no_grad()
def test_gekko_native_tdn_dispatch_pool_roundtrip_and_reuse(pieces):
    torch.manual_seed(331)
    layer = (
        reference.GekkoAttention(
            {
                "model_dim": 1536,
                "num_heads": 2,
                "num_kv_heads": 1,
                "head_dim": 4,
                "v_head_dim": 4,
                "chunk_size": 512,
                "timenorm_num_groups": 32,
                "apply_bias_term": True,
                "gekko_value_activation": "identity",
            }
        )
        .cuda()
        .eval()
    )
    for parameter in layer.parameters():
        parameter.data = parameter.data.bfloat16()
    layer.timenorm.prior_count.fill_(7)
    layer.timenorm.prior_mean.copy_(torch.linspace(-0.01, 0.01, 32, device="cuda"))
    layer.timenorm.prior_var.fill_(0.01)
    hidden = torch.randn(sum(pieces), 1536, device="cuda", dtype=torch.bfloat16)
    cache = pool_test.pool(layer)
    cache.conv = [value.cuda() for value in cache.conv]
    cache.temporal = torch.full(
        (*cache.temporal.shape[:-1], cache.temporal.shape[-1] + tdn.TDN_STATE_SIZE),
        torch.nan,
        device="cuda",
        dtype=torch.float32,
    )
    state = layer.initial_state(hidden)
    assert state.tdn_packed.numel() == tdn.TDN_STATE_SIZE
    assert state.tdn_packed.dtype == torch.float32
    expected = tdn.PackedTDNState.from_storage(state.tdn_packed.clone(), 0)
    offset = 0
    for size in pieces:
        x = hidden[offset : offset + size]
        expected_z, expected_mean, expected_var, expected = tdn.native_tdn_forward(
            x,
            layer.timenorm.weight,
            layer.timenorm.bias,
            expected,
            layer.timenorm.beta1,
            layer.timenorm.beta2,
            layer.timenorm.eps,
        )
        state = pool_test.Backend._read_state(layer, cache, 2, offset, x)
        snapshot = state.clone()
        projected_inputs = []
        hook = layer.wq.register_forward_pre_hook(
            lambda module, inputs, captured=projected_inputs: captured.append(
                inputs[0].clone()
            )
        )
        try:
            _, updated = layer(x, state)
        finally:
            hook.remove()
        torch.testing.assert_close(projected_inputs[0], expected_z, atol=0, rtol=0)
        pool_test.assert_state_exact(state, snapshot)
        torch.testing.assert_close(updated.tdn_packed, expected.storage, atol=0, rtol=0)
        torch.testing.assert_close(updated.tdn_mean, expected_mean, atol=0, rtol=0)
        torch.testing.assert_close(updated.tdn_var, expected_var, atol=0, rtol=0)
        offset += size
        assert updated.position == offset
        assert updated.tdn_count.item() == offset + 7
        pool_test.Backend._write_state(cache, 2, updated)
        restored = pool_test.Backend._read_state(layer, cache, 2, offset, x)
        pool_test.assert_state_exact(restored, updated)
        assert torch.isnan(cache.temporal[0]).all()
    # Copying a request to another slot preserves all partial scan words.
    pool_test.Backend._write_state(cache, 0, restored)
    pool_test.assert_state_exact(
        pool_test.Backend._read_state(layer, cache, 0, offset, hidden), restored
    )
    # Prefix zero ignores every stale payload byte after cancellation/reuse.
    cache.temporal[2].fill_(torch.nan)
    for value in cache.conv:
        value[2].fill_(torch.nan)
    fresh = pool_test.Backend._read_state(layer, cache, 2, 0, hidden[:1])
    pool_test.assert_state_exact(fresh, layer.initial_state(hidden[:1]))
    _, fresh = layer(hidden[:1], fresh)
    assert torch.isfinite(fresh.tdn_packed).all()
    assert fresh.position == 1
    assert fresh.tdn_count.item() == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
