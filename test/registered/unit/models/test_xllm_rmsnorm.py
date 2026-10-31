"""CUDA arithmetic and optional original-native oracle checks for xLLM RMSNorm."""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest
import torch

_MODELS = Path(__file__).resolve().parents[4] / "python/sglang/srt/models"
_PACKAGE = "_xllm_models_cuda"
if _PACKAGE not in sys.modules:
    _package = types.ModuleType(_PACKAGE)
    _package.__path__ = [str(_MODELS)]
    sys.modules[_PACKAGE] = _package
reference = importlib.import_module(f"{_PACKAGE}.xllm_reference")
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _MODELS.parents[1] / "test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native xLLM RMS arithmetic requires NVIDIA CUDA",
)

# Includes native Gekko head widths, the 1536-channel shared backbone, vector
# load boundaries, and grouped reductions with independent affine parameters.
_SHAPES = [
    (64, 1),
    (96, 1),
    (128, 1),
    (256, 1),
    (512, 1),
    (768, 1),
    (1024, 1),
    (1536, 1),
    (2048, 1),
    (2048, 32),
    (4096, 1),
    (8192, 1),
    (1536, 24),
    (1536, 3),
    (384, 3),
    (189, 3),
]


def case_key(dimension, groups, length, affine):
    return f"d{dimension}_g{groups}_t{length}_a{int(affine)}"


@lru_cache(maxsize=1)
def load_oracle_fixture(path):
    from safetensors.torch import load_file

    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    tensors_file = directory / "oracle.safetensors"
    with tensors_file.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual_hash == manifest["sha256"], "Native RMS fixture changed after capture"
    tensors = load_file(tensors_file, device="cuda")
    return manifest["cases"], tensors


def apply_norm(norm, x, affine):
    if affine:
        return norm(x)
    kernel = importlib.import_module(f"{_PACKAGE}.xllm_rmsnorm")
    return kernel.native_rms_norm(x, None, norm.eps, norm.num_groups)


def make_case(dimension, groups, length):
    generator = torch.Generator(device="cuda").manual_seed(419 + dimension + groups)
    norm = reference.NativeRMSNorm(dimension, eps=1e-6, num_groups=groups)
    norm = norm.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    norm.weight.copy_(0.3 * torch.randn(dimension, generator=generator, device="cuda"))
    x = torch.randn(
        length, dimension, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    return norm, x


@pytest.mark.parametrize("dimension,groups", _SHAPES)
@pytest.mark.parametrize("affine", [False, True])
@torch.no_grad()
def test_grouped_rms_math_and_partition_invariance(dimension, groups, affine):
    norm, x = make_case(dimension, groups, 31)
    actual = apply_norm(norm, x, affine)
    grouped = x.double().reshape(31, groups, -1)
    expected = (
        (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + norm.eps))
        .reshape(x.shape)
        .mul((norm.weight + 1.0).double() if affine else 1.0)
        .to(x.dtype)
    )
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-6)
    pieces = torch.cat(
        [apply_norm(norm, chunk, affine) for chunk in (x[:1], x[1:8], x[8:])]
    )
    torch.testing.assert_close(actual, pieces, rtol=0, atol=0)
    assert actual.dtype == x.dtype and actual.shape == x.shape


@pytest.mark.parametrize("dimension,groups", _SHAPES)
@pytest.mark.parametrize("affine", [False, True])
@torch.no_grad()
def test_original_native_bf16_exact(dimension, groups, affine):
    fixture = os.environ.get("XLLM_RMS_ORACLE_FIXTURE")
    if fixture:
        cases, tensors = load_oracle_fixture(fixture)
        for length in (1, 31, 127):
            key = case_key(dimension, groups, length, affine)
            metadata = cases[key]
            assert metadata["dimension"] == dimension and metadata["groups"] == groups
            assert metadata["length"] == length and metadata["affine"] is affine
            norm = reference.NativeRMSNorm(
                dimension, eps=metadata["eps"], num_groups=groups
            )
            norm = norm.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)
            if affine:
                norm.weight.copy_(tensors[f"{key}.weight"])
            actual = apply_norm(norm, tensors[f"{key}.input"], affine)
            torch.testing.assert_close(
                actual, tensors[f"{key}.expected"], rtol=0, atol=0
            )
        return
    source = os.environ.get("XLLM_NATIVE_SOURCE")
    if source is None:
        pytest.skip("Set XLLM_NATIVE_SOURCE for unchanged native CUDA oracle")
    assert Path(source).is_dir(), f"Native source directory does not exist: {source}"
    sys.path.insert(0, source)
    try:
        from xllm_extension.ops import group_rms_norm_fwd, group_rms_norm_fwd_affine
    finally:
        sys.path.pop(0)
    for length in (1, 31, 127):
        norm, x = make_case(dimension, groups, length)
        if affine:
            expected, _ = group_rms_norm_fwd_affine(
                x[None], dimension, groups, norm.weight + 1.0, norm.eps
            )
        else:
            expected, _ = group_rms_norm_fwd(x[None], dimension, groups, norm.eps)
        torch.testing.assert_close(
            apply_norm(norm, x, affine), expected[0], rtol=0, atol=0
        )


@torch.no_grad()
def test_noncontiguous_input_empty_tokens_and_batch_shape():
    norm, x = make_case(1536, 1, 12)
    strided = x[::2]
    torch.testing.assert_close(
        norm(strided), norm(strided.contiguous()), rtol=0, atol=0
    )
    torch.testing.assert_close(
        norm(x.reshape(3, 4, -1)).flatten(0, 1), norm(x), rtol=0, atol=0
    )
    assert norm(x[:0]).shape == (0, 1536)


@torch.no_grad()
def test_offset_addition_rounds_before_fp32_normalization():
    norm = reference.NativeRMSNorm(64).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.fill_(0.00390625)
    # 1 + 1/256 is a BF16 half-way value and rounds to exactly 1.
    x = torch.ones(2, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(norm(x), x, rtol=0, atol=0)


def test_gradient_enabled_forward_keeps_torch_autograd():
    norm, x = make_case(64, 1, 3)
    x.requires_grad_(True)
    output = norm(x)
    output.float().square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
