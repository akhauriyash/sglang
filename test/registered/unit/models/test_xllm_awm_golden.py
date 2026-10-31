"""Public native AWM goldens for three SM90 BF16 4096-token chunks.

The fixture contains hashes and samples, with no checkpoint data. Integer and
power-of-two input construction is shared with independent native capture.
FP32 normalizer hashes detect reduction-layout changes even when BF16 outputs
round to the same values.
"""

import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _ROOT / "python/sglang/srt/models"
_FIXTURE = Path(__file__).with_name("fixtures") / "xllm_awm_native_golden.json"
_PARAMETERS = {
    "chunk_size": 4096,
    "chunks": 3,
    "query_heads": 32,
    "kv_heads": 8,
    "head_dim": 64,
}
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_awm_golden_ci", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native xLLM AWM goldens require SM90 NVIDIA CUDA",
)


def make_golden_inputs() -> dict[str, torch.Tensor]:
    """Recipe 1: CPU integer arithmetic; no RNG or floating-point reductions."""
    length = _PARAMETERS["chunk_size"] * _PARAMETERS["chunks"]
    q_index = torch.arange(length * 32 * 64, dtype=torch.int64).view(length, 32, 64)
    kv_index = torch.arange(length * 8 * 64, dtype=torch.int64).view(length, 8, 64)
    return {
        "query": (((q_index * 7 + (q_index // 64) * 3) % 31 + 1).float() / 1024).to(
            torch.bfloat16
        ),
        "feature_keys": (
            ((kv_index * 11 + (kv_index // 64) * 5) % 31 + 1).float() / 1024
        ).to(torch.bfloat16),
        "raw_keys": (
            (
                (
                    kv_index * 17
                    + (kv_index // 64) * 13
                    + (kv_index // 512) * 23
                    + (kv_index // 4096) * 19
                )
                % 2047
                - 1023
            ).float()
            / 256
        ).to(torch.bfloat16),
        "values": (
            ((kv_index * 29 + (kv_index // 64) * 7) % 509 - 254).float() / 64
        ).to(torch.bfloat16),
    }


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run_updates(update, inputs):
    """Emit each chunk before incorporating its delayed predecessor."""
    query, values = inputs["query"], inputs["values"]
    chunk_size = _PARAMETERS["chunk_size"]
    memory = values.new_zeros(8, 64, 64)
    normalizer = torch.full(
        (8, 64), -torch.inf, device=query.device, dtype=torch.float32
    )
    outputs = []
    for index in range(_PARAMETERS["chunks"]):
        start = index * chunk_size
        outputs.append(
            torch.einsum(
                "thd,hdv->thv",
                query[start : start + chunk_size],
                memory.repeat_interleave(4, dim=0),
            )
        )
        if index:
            previous = slice(start - chunk_size, start)
            memory, normalizer = update(
                memory,
                normalizer,
                inputs["feature_keys"][previous],
                inputs["raw_keys"][previous],
                values[previous],
            )
    mask = torch.arange(len(query), device=query.device) < 2 * chunk_size
    return {
        "output": torch.cat(outputs),
        "mask": mask,
        "memory": memory,
        "log_normalizer": normalizer,
    }


@torch.no_grad()
def test_public_native_awm_golden():
    fixture = json.loads(_FIXTURE.read_text())
    assert fixture["format_version"] == fixture["recipe_version"] == 1
    assert fixture["parameters"] == _PARAMETERS
    inputs = make_golden_inputs()
    for name, value in inputs.items():
        metadata = fixture["inputs"][name]
        assert metadata["shape"] == list(value.shape)
        assert metadata["dtype"] == str(value.dtype)
        assert metadata["sha256"] == tensor_sha256(value), f"Recipe changed: {name}"
    # Lazy import ensures native capture shares only the input recipe.
    package = "_xllm_models_awm_golden"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(_MODELS)]
        sys.modules[package] = module
    operator = importlib.import_module(f"{package}.xllm_gekko_reference")
    actual = run_updates(
        operator.adaptive_working_memory_update,
        {name: value.cuda() for name, value in inputs.items()},
    )
    for name, value in actual.items():
        metadata = fixture["outputs"][name]
        flat = value.flatten().cpu()
        samples = {
            str(index): flat[index].item() for index in metadata["sample_indices"]
        }
        assert metadata["shape"] == list(value.shape)
        assert metadata["dtype"] == str(value.dtype)
        assert tensor_sha256(value) == metadata["sha256"], (
            f"Native AWM {name} differs: expected samples {metadata['samples']}, "
            f"actual samples {samples}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
