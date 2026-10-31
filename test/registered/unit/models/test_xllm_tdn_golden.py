"""Public synthetic native TDN goldens for the 32 x 48 BF16 CUDA layout.

The tiny JSON fixture records hashes of independently captured native outputs,
not model weights. Input construction uses integer arithmetic and powers of
two; it has no RNG or private-checkpoint dependency. The fixture records its
native CUDA build and hardware, without claiming other TDN layouts are exact.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _ROOT / "python/sglang/srt/models"
_FIXTURE = Path(__file__).with_name("fixtures") / "xllm_tdn_native_golden.json"
_PARAMETERS = {"beta1": 0.999, "beta2": 0.9999, "eps": 1e-5}
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native xLLM TDN goldens require NVIDIA CUDA",
)


def make_golden_inputs(nonzero: bool) -> dict[str, torch.Tensor]:
    """Recipe version 1: CPU-only exact arithmetic, shared with native capture."""
    indices = torch.arange(257 * 1536, dtype=torch.int64).reshape(257, 1536)
    channels = torch.arange(1536, dtype=torch.int64)
    groups = torch.arange(32, dtype=torch.int64)
    return {
        "x": (((indices * 17 + (indices // 1536) * 13) % 257 - 128).float() / 32).to(
            torch.bfloat16
        ),
        "weight": (((channels * 7) % 33 - 16).float() / 128).to(torch.bfloat16),
        "bias": (((channels * 11) % 17 - 8).float() / 256).to(torch.bfloat16),
        "prior_count": torch.tensor([1031 if nonzero else 0], dtype=torch.int64),
        # Values outside BF16 precision exercise the initial prior cast.
        "prior_mean": (groups * 31 - 433).float() / 4096
        if nonzero
        else torch.zeros(32, dtype=torch.float32),
        "prior_var": (groups * 29 + 97).float() / 16384
        if nonzero
        else torch.zeros(32, dtype=torch.float32),
    }


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def tdn():
    # Native fixture capture imports only make_golden_inputs; the production
    # operator is imported lazily here and never contributes expected values.
    package = "_xllm_models_cuda"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(_MODELS)]
        sys.modules[package] = module
    return importlib.import_module(f"{package}.xllm_tdn")


@pytest.mark.parametrize("nonzero", [False, True], ids=["zero-prior", "nonzero-prior"])
@pytest.mark.parametrize("pieces", [(257,), (255, 1, 1)], ids=["whole", "boundary"])
@torch.no_grad()
def test_public_native_golden(tdn, nonzero, pieces):
    fixture_file = Path(os.environ.get("XLLM_TDN_GOLDEN_FIXTURE", _FIXTURE))
    fixture = json.loads(fixture_file.read_text())
    assert fixture["format_version"] == fixture["recipe_version"] == 1
    assert fixture["parameters"] == _PARAMETERS
    expected = fixture["cases"]["nonzero" if nonzero else "zero"]
    inputs = make_golden_inputs(nonzero)
    for name, value in inputs.items():
        metadata = expected["inputs"][name]
        assert metadata["shape"] == list(value.shape)
        assert metadata["dtype"] == str(value.dtype)
        assert metadata["sha256"] == tensor_sha256(value), (
            f"Input recipe changed: {name}"
        )
    inputs = {name: value.cuda() for name, value in inputs.items()}
    state = tdn.init_tdn_state(
        inputs["prior_count"],
        inputs["prior_mean"],
        inputs["prior_var"],
        _PARAMETERS["beta1"],
        _PARAMETERS["beta2"],
    )
    outputs, offset = [], 0
    for length in pieces:
        output, mean, var, state = tdn.native_tdn_forward(
            inputs["x"][offset : offset + length],
            inputs["weight"],
            inputs["bias"],
            state,
            **_PARAMETERS,
        )
        outputs.append(output)
        offset += length
    actual = {
        "output": torch.cat(outputs),
        "count": (inputs["prior_count"] + state.position).reshape(1),
        "mean": mean.bfloat16(),
        "var": var.bfloat16(),
    }
    assert state.position == 257
    for name, value in actual.items():
        metadata = expected["outputs"][name]
        flat = value.flatten().cpu()
        samples = {
            str(index): flat[index].item() for index in metadata["sample_indices"]
        }
        assert metadata["shape"] == list(value.shape)
        assert metadata["dtype"] == str(value.dtype)
        assert tensor_sha256(value) == metadata["sha256"], (
            f"Native {name} differs: expected samples {metadata['samples']}, "
            f"actual samples {samples}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
