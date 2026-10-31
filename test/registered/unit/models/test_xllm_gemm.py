"""CPU fallback and shape contract for Gekko-only FFN down projection."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gemm = load("_xllm_gemm_cpu", _ROOT / "python/sglang/srt/models/xllm_gemm.py")
ci = load("_xllm_gemm_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7), (0, 7), (2, 0, 7)])
@pytest.mark.parametrize("bias", [False, True])
def test_cpu_fallback_preserves_shapes_values_and_keys(shape, bias):
    torch.manual_seed(129)
    ordinary = torch.nn.Linear(7, 5, bias=bias)
    actual = gemm.GekkoDownProjection(7, 5, bias=bias)
    actual.load_state_dict(ordinary.state_dict(), strict=True)
    assert actual.state_dict().keys() == ordinary.state_dict().keys()
    x = torch.randn(shape)
    torch.testing.assert_close(actual(x), ordinary(x), rtol=0, atol=0)


def test_cpu_autograd_is_ordinary_linear():
    torch.manual_seed(17)
    module = gemm.GekkoDownProjection(7, 5, bias=True)
    x = torch.randn(3, 7, requires_grad=True)
    gradients = torch.autograd.grad(
        module(x).square().sum(), (x, module.weight, module.bias)
    )
    expected = torch.autograd.grad(
        F.linear(x, module.weight, module.bias).square().sum(),
        (x, module.weight, module.bias),
    )
    for actual, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_matching_cpu_bf16_configuration_does_not_pad(monkeypatch):
    module = gemm.GekkoDownProjection(5120, 1536, bias=False, dtype=torch.bfloat16)
    observed = []
    original = F.linear

    def observe(x, *args):
        observed.append(x.shape)
        return original(x, *args)

    monkeypatch.setattr(F, "linear", observe)
    with torch.no_grad():
        output = module(torch.zeros(1, 5120, dtype=torch.bfloat16))
    assert observed == [(1, 5120)]
    assert output.shape == (1, 1536)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
