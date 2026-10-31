"""CPU fallback, state ownership, and checkpoint inventory for Gekko ShortConv."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE = "_xllm_shortconv_cpu"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(_ROOT / "python/sglang/srt/models")]
sys.modules.setdefault(_PACKAGE, _package)
reference = importlib.import_module(f"{_PACKAGE}.xllm_gekko_reference")
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def forbid_cuda_helper_on_cpu(monkeypatch):
    module = types.ModuleType(f"{_PACKAGE}.xllm_shortconv")

    def forbidden(*args):
        raise AssertionError("CPU fallback must not call the CUDA helper")

    module.native_shortconv_forward = forbidden
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [0, 1, 2, 5])
def test_cpu_state_and_mathematical_convolution(width, length):
    module = reference.NormalizedShortConv(4, width)
    with torch.no_grad():
        module.weight.copy_(torch.arange(4 * width).reshape(4, width) / 7)
    x = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4) / 9
    history = (
        torch.arange((width - 1) * 4, dtype=torch.float32).reshape(width - 1, 4) / 5
    )
    snapshot = history.clone()
    output, state = module(x, history)
    assert set(module.state_dict()) == {"weight"}
    if not length:
        assert output is x and state is history
        return
    stream = torch.cat((history, x))
    expected = F.conv1d(stream.T[None], module.weight.softmax(-1)[:, None], groups=4)[
        0
    ].T
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    expected_state = stream[-(width - 1) :] if width > 1 else stream[:0]
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    state.zero_()
    assert torch.equal(history, snapshot)


@pytest.mark.parametrize("gradient", ["x", "weight", "history"])
def test_cpu_gradients_remain_available(gradient):
    module = reference.NormalizedShortConv(4, 4)
    module.weight.requires_grad_(False)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7
    history = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 3
    tensors = {"x": x, "weight": module.weight, "history": history}
    tensors[gradient].requires_grad_(True)
    output, _ = module(x, history)
    output.square().sum().backward()
    assert tensors[gradient].grad is not None
    assert torch.isfinite(tensors[gradient].grad).all()


def test_bad_history_fails_before_dispatch():
    module = reference.NormalizedShortConv(4, 4)
    with pytest.raises(ValueError, match="history"):
        module(torch.zeros(1, 4), torch.zeros(2, 4))


def test_normalized_weights_refresh_after_mutation_load_and_replacement():
    module = reference.NormalizedShortConv(4, 4).bfloat16()
    first = module._normalized_weight()
    assert module._normalized_weight() is first
    assert not first.requires_grad and first.grad_fn is None
    assert module._captured_normalized_weights == {}
    with torch.no_grad():
        module.weight[:, -1].fill_(3)
    changed = module._normalized_weight()
    assert changed is not first
    assert not torch.equal(changed, first)
    torch.testing.assert_close(
        changed, module.weight.float().softmax(-1).bfloat16(), rtol=0, atol=0
    )
    module.load_state_dict({"weight": torch.arange(16).reshape(4, 4).bfloat16()})
    loaded = module._normalized_weight()
    assert loaded is not changed
    torch.testing.assert_close(
        loaded, module.weight.float().softmax(-1).bfloat16(), rtol=0, atol=0
    )
    # A replacement Parameter can share its predecessor's storage and version.
    module.weight = torch.nn.Parameter(module.weight.detach())
    replaced = module._normalized_weight()
    assert replaced is not loaded
    assert torch.equal(replaced, loaded)
    assert set(module.state_dict()) == {"weight"}
    assert not list(module.named_buffers())


def test_normalized_weights_device_dtype_apply_clears_active_cache():
    module = reference.NormalizedShortConv(4, 4)
    first = module._normalized_weight()
    module.double()
    assert module._normalized_weight_cache is None
    changed = module._normalized_weight()
    assert changed.dtype == torch.float64 and changed is not first
    assert not changed.requires_grad and changed.grad_fn is None
    module.to("cpu")
    assert module._normalized_weight_cache is None


def test_inference_created_weights_are_not_cached_without_version_tracking():
    module = reference.NormalizedShortConv(4, 4)
    with torch.inference_mode():
        module.weight = torch.nn.Parameter(torch.zeros(4, 4))
    first = module._normalized_weight()
    assert module._normalized_weight_cache is None
    with torch.inference_mode():
        module.weight[:, 0].fill_(4)
    changed = module._normalized_weight()
    assert module._normalized_weight_cache is None
    assert not torch.equal(first, changed)
    assert not changed.requires_grad and changed.grad_fn is None


def test_pooled_decode_rejects_autograd_before_dispatch():
    module = reference.NormalizedShortConv(4, 4)
    with pytest.raises(ValueError, match="inference-only"):
        module.decode_pooled(torch.zeros(2, 4), torch.zeros(3, 3, 4), torch.arange(2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
