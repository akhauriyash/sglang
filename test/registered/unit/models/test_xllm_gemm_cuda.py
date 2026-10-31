"""SM90 Gekko GEMM learned-oracle and public synthetic partition checks."""

import hashlib
import importlib.util
import json
import os
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


gemm = load("_xllm_gemm_cuda", _ROOT / "python/sglang/srt/models/xllm_gemm.py")
ci = load("_xllm_gemm_cuda_ci", _ROOT / "python/sglang/test/ci/ci_register.py")
register_cuda_ci = ci.register_cuda_ci
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Gekko GEMM row-count policy is qualified only on NVIDIA SM90",
)


@pytest.fixture(scope="module")
def projection():
    module = gemm.GekkoDownProjection(
        5120, 1536, bias=False, device="cuda", dtype=torch.bfloat16
    ).eval()
    # Exact powers-of-two inputs give a checkpoint-independent arithmetic and
    # partition test. Learned fixtures below separately test native BF16 ties.
    indices = torch.arange(1536 * 5120, dtype=torch.int64).reshape(1536, 5120)
    weights = (((indices * 7 + indices // 5120 * 13) % 31 - 15).float() / 64).bfloat16()
    with torch.no_grad():
        module.weight.copy_(weights)
    return module


def public_input(length):
    indices = torch.arange(length * 5120, dtype=torch.int64).reshape(length, 5120)
    return (
        (((indices * 11 + indices // 5120 * 17) % 63 - 31).float() / 64)
        .bfloat16()
        .cuda()
    )


@torch.no_grad()
@pytest.mark.parametrize("pieces", [(257,), (31, 1, 96, 128, 1), (256, 1)])
def test_public_synthetic_partition_equivalence(projection, pieces):
    x = public_input(257)
    expected = F.linear(x, projection.weight)
    outputs, offset = [], 0
    for length in pieces:
        outputs.append(projection(x[offset : offset + length]))
        offset += length
    assert offset == len(x)
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("length", [0, 1, 2, 31, 127, 128, 255, 256, 257])
def test_only_short_nonempty_calls_pad_and_preserve_inputs(
    projection, length, monkeypatch
):
    x = public_input(length)
    original_x = x.clone()
    observed, original = [], F.linear

    def observe(value, *args):
        observed.append(tuple(value.shape))
        return original(value, *args)

    monkeypatch.setattr(F, "linear", observe)
    actual = projection(x)
    assert observed == [(256 if 0 < length < 256 else length, 5120)]
    assert actual.shape == (length, 1536)
    assert torch.equal(x, original_x)


@torch.no_grad()
def test_vector_and_strided_leading_dimensions(projection):
    x = public_input(12)
    expected = projection(x)
    torch.testing.assert_close(
        projection(x.reshape(2, 6, 5120)), expected.reshape(2, 6, 1536), rtol=0, atol=0
    )
    torch.testing.assert_close(projection(x[0]), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(projection(x[::2]), expected[::2], rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_autocast_uses_ordinary_linear(projection, autocast_dtype, monkeypatch):
    x = public_input(1)
    observed, original = [], F.linear

    def observe(value, *args):
        observed.append(tuple(value.shape))
        return original(value, *args)

    monkeypatch.setattr(F, "linear", observe)
    with torch.autocast("cuda", dtype=autocast_dtype):
        expected = original(x, projection.weight)
        actual = projection(x)
    assert observed == [(1, 5120)]
    assert actual.dtype == expected.dtype == autocast_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_saved_native_learned_rows_are_exact():
    from safetensors.torch import load_file

    directory = os.environ.get("XLLM_GEMM_ORACLE_FIXTURE")
    if directory is None:
        pytest.skip(
            "Set XLLM_GEMM_ORACLE_FIXTURE to independently captured native rows"
        )
    directory = Path(directory)
    manifest = json.loads((directory / "capture.json").read_text())
    tensor_file = directory / "linear_inputs.safetensors"
    with tensor_file.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == manifest["tensors_sha256"]
        )
    tensors = load_file(tensor_file, device="cuda")
    module = gemm.GekkoDownProjection(
        5120, 1536, bias=False, device="cuda", dtype=torch.bfloat16
    )
    module.weight.copy_(tensors["nffn.fc2.weight"])
    inputs, expected = (
        tensors["nffn.fc2.input_rows"],
        tensors["nffn.fc2.native_output_rows"],
    )
    for length in (1, 2, 31, 127, 128, 255, 256):
        for row, target in zip(inputs, expected, strict=True):
            actual = module(row[None].repeat(length, 1))
            torch.testing.assert_close(actual, target.expand_as(actual), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
