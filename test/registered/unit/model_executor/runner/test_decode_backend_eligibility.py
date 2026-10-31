"""Backend state boundaries must veto replay even without MLP synchronization."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("require_mlp_sync", [False, True])
def test_backend_veto_precedes_shape_and_mlp_eligibility(require_mlp_sync):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.require_mlp_sync = require_mlp_sync
    runner.attn_backend = Mock(spec=AttentionBackend)
    runner.attn_backend.can_run_decode_cuda_graph.return_value = False
    batch = SimpleNamespace(can_run_decode_cuda_graph=True)
    assert runner.can_run_graph(batch) is False
    runner.attn_backend.can_run_decode_cuda_graph.assert_called_once_with(batch)


@pytest.mark.parametrize("children", [(True, True), (False, True), (True, False)])
def test_hybrid_requires_both_attention_backends(children):
    hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    hybrid.attn_backend_list = [Mock(spec=AttentionBackend) for _ in children]
    for backend, eligible in zip(hybrid.attn_backend_list, children, strict=True):
        backend.can_run_decode_cuda_graph.return_value = eligible
    assert hybrid.can_run_decode_cuda_graph(SimpleNamespace()) is all(children)


def test_stateless_backend_preserves_existing_eligibility():
    assert AttentionBackend().can_run_decode_cuda_graph(SimpleNamespace()) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
