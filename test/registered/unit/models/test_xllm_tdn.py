"""Bounded TDN state and optional immutable native CUDA oracle checks."""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from functools import cache
from pathlib import Path

import pytest
import torch

_MODELS = Path(__file__).resolve().parents[4] / "python/sglang/srt/models"
_PACKAGE = "_xllm_models_cuda"
if _PACKAGE not in sys.modules:
    _package = types.ModuleType(_PACKAGE)
    _package.__path__ = [str(_MODELS)]
    sys.modules[_PACKAGE] = _package
tdn = importlib.import_module(f"{_PACKAGE}.xllm_tdn")
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _MODELS.parents[1] / "test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native xLLM TDN arithmetic requires NVIDIA CUDA",
)

_PARAMETERS = {"beta1": 0.999, "beta2": 0.9999, "eps": 1e-5}


def _oracle_cases():
    paths = json.loads(os.environ.get("XLLM_TDN_ORACLE_FIXTURES", "[]"))
    cases = []
    for path in paths:
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        for index, row in enumerate(manifest["cases"]):
            cases.append(pytest.param(path, index, id=f"{row['kind']}-{row['length']}"))
    return cases or [
        pytest.param(None, (length, nonzero), id=f"native-{length}-{nonzero}")
        for length in (31, 257, 4096)
        for nonzero in (False, True)
    ]


@cache
def _load_fixture(path):
    from safetensors.torch import load_file

    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    tensors_file = directory / "oracle.safetensors"
    with tensors_file.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual_hash == manifest["sha256"], "Native TDN fixture changed after capture"
    return manifest, load_file(tensors_file, device="cuda")


def _make_inputs(length, nonzero=False):
    generator = torch.Generator(device="cuda").manual_seed(57103)
    return {
        "x": torch.randn(
            length, 1536, device="cuda", dtype=torch.bfloat16, generator=generator
        ),
        "weight": torch.randn(
            1536, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.2,
        "bias": torch.randn(
            1536, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.03,
        "prior_count": torch.tensor(
            [1031 if nonzero else 0], device="cuda", dtype=torch.int64
        ),
        "prior_mean": torch.linspace(-0.112345, 0.231234, 32, device="cuda")
        if nonzero
        else torch.zeros(32, device="cuda"),
        "prior_var": torch.linspace(0.017123, 0.213456, 32, device="cuda")
        if nonzero
        else torch.zeros(32, device="cuda"),
    }


@cache
def _case(path, index):
    if path:
        manifest, tensors = _load_fixture(path)
        inputs, expected = {}, {}
        for name, destination in (("input", inputs), ("native", expected)):
            prefix = f"case{index}.{name}."
            destination.update(
                (key[len(prefix) :], value)
                for key, value in tensors.items()
                if key.startswith(prefix)
            )
        assert len(inputs["x"]) == manifest["cases"][index]["length"]
        return inputs, expected, manifest["parameters"]
    source = os.environ.get("XLLM_NATIVE_SOURCE")
    if not source:
        pytest.skip(
            "Set XLLM_TDN_ORACLE_FIXTURES or XLLM_NATIVE_SOURCE for native oracle"
        )
    assert Path(source).is_dir(), f"Native source directory does not exist: {source}"
    sys.path.insert(0, source)
    try:
        from xllm_extension.ops import group_timestep_decay_norm_cub_fwd
    finally:
        sys.path.pop(0)
    inputs = _make_inputs(*index)
    x = inputs["x"]
    result = group_timestep_decay_norm_cub_fwd(
        x[None],
        None,
        inputs["prior_count"],
        inputs["prior_mean"].to(x.dtype)[None],
        inputs["prior_var"].to(x.dtype)[None],
        inputs["weight"] + 1.0,
        inputs["bias"],
        None,
        32,
        _PARAMETERS["beta1"],
        _PARAMETERS["beta2"],
        _PARAMETERS["eps"],
    )
    expected = {
        "output": result[0][0],
        "count": result[1],
        "mean": result[2][0],
        "var": result[3][0],
    }
    return inputs, expected, _PARAMETERS


def _initial(inputs, parameters):
    return tdn.init_tdn_state(
        inputs["prior_count"],
        inputs["prior_mean"],
        inputs["prior_var"],
        parameters["beta1"],
        parameters["beta2"],
    )


def _partition(length, sizes):
    result, cursor = [], 0
    while sum(result) < length:
        result.append(min(sizes[cursor % len(sizes)], length - sum(result)))
        cursor += 1
    return result


def _stream(inputs, state, parameters, sizes, start=0):
    outputs, cursor = [], start
    for size in sizes:
        output, mean, var, state = tdn.native_tdn_forward(
            inputs["x"][cursor : cursor + size],
            inputs["weight"],
            inputs["bias"],
            state,
            **parameters,
        )
        outputs.append(output)
        cursor += size
    return torch.cat(outputs), mean, var, state


def _assert_result(expected, output, mean, var, state, inputs):
    torch.testing.assert_close(output, expected["output"], rtol=0, atol=0)
    torch.testing.assert_close(mean.bfloat16(), expected["mean"], rtol=0, atol=0)
    torch.testing.assert_close(var.bfloat16(), expected["var"], rtol=0, atol=0)
    torch.testing.assert_close(
        (inputs["prior_count"] + state.position).reshape(1),
        expected["count"].reshape(1),
        rtol=0,
        atol=0,
    )
    assert state.storage.numel() == tdn.TDN_STATE_SIZE == 16450
    assert state.storage.dtype == torch.float32


@pytest.mark.parametrize("path,index", _oracle_cases())
@pytest.mark.parametrize("schedule", ["whole", "decode", "irregular", "boundaries"])
@torch.no_grad()
def test_native_oracle_partition_exact(path, index, schedule):
    inputs, expected, parameters = _case(path, index)
    length = len(inputs["x"])
    sizes = {
        "whole": [length],
        "decode": [1],
        "irregular": [7, 33, 1, 211, 4, 256, 17, 129],
        "boundaries": [255, 1, 1, 255, 256],
    }[schedule]
    state = _initial(inputs, parameters)
    # Unfilled pool contents must never affect the valid partial prefix.
    state.partial_mean.fill_(torch.nan)
    state.partial_var.fill_(torch.nan)
    before = state.storage.view(torch.int32).clone()
    result = _stream(inputs, state, parameters, _partition(length, sizes))
    _assert_result(expected, *result, inputs)
    assert state.position == 0
    assert torch.equal(state.storage.view(torch.int32), before)


@pytest.mark.parametrize("path,index", _oracle_cases())
@torch.no_grad()
def test_native_oracle_pooled_snapshot_fork_exact(path, index):
    inputs, expected, parameters = _case(path, index)
    length = len(inputs["x"])
    cut = min(173, length // 2)
    prefix, _, _, saved = _stream(
        inputs, _initial(inputs, parameters), parameters, [cut]
    )
    pool = torch.full((3, tdn.TDN_STATE_SIZE), torch.nan, device="cuda")
    pool[2].copy_(saved.storage)
    restored = tdn.PackedTDNState.from_storage(pool[2], saved.position)
    assert restored.storage.data_ptr() == pool[2].data_ptr()
    fork = restored.clone()
    assert fork.storage.data_ptr() != restored.storage.data_ptr()
    before = pool.view(torch.int32).clone()
    for state, sizes in (
        (restored, [length - cut]),
        (fork, _partition(length - cut, [1, 11, 257])),
    ):
        suffix, mean, var, state = _stream(inputs, state, parameters, sizes, start=cut)
        _assert_result(expected, torch.cat((prefix, suffix)), mean, var, state, inputs)
    assert torch.equal(pool.view(torch.int32), before)


@torch.no_grad()
def test_interleaved_requests_and_prior_rounding():
    inputs = [_make_inputs(533, False), _make_inputs(533, True)]
    states = [_initial(item, _PARAMETERS) for item in inputs]
    for item, state in zip(inputs, states):
        assert state.position == 0
        torch.testing.assert_close(
            state.block_mean, item["prior_mean"].bfloat16().float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            state.block_var, item["prior_var"].bfloat16().float(), rtol=0, atol=0
        )
    expected = [
        _stream(item, state, _PARAMETERS, [533]) for item, state in zip(inputs, states)
    ]
    outputs, cursors = [[], []], [0, 0]
    for step in range(8):
        for request in (0, 1) if step % 2 else (1, 0):
            size = min((17, 255, 1, 31)[step % 4], 533 - cursors[request])
            if not size:
                continue
            output, _, _, states[request] = _stream(
                inputs[request], states[request], _PARAMETERS, [size], cursors[request]
            )
            outputs[request].append(output)
            cursors[request] += size
    for request in (0, 1):
        assert cursors[request] == 533
        torch.testing.assert_close(
            torch.cat(outputs[request]), expected[request][0], rtol=0, atol=0
        )


@torch.no_grad()
def test_fresh_packed_state_is_deterministic_and_independent():
    inputs = _make_inputs(1, nonzero=True)
    # Populate the caching allocator's same-size free list with poison. The
    # initializer must define every word, regardless of allocator reuse.
    poisoned = torch.full(
        (tdn.TDN_STATE_SIZE,), torch.nan, device="cuda", dtype=torch.float32
    )
    del poisoned
    first = _initial(inputs, _PARAMETERS)
    second = _initial(inputs, _PARAMETERS)
    assert first.storage.data_ptr() != second.storage.data_ptr()
    assert first.position == second.position == 0
    torch.testing.assert_close(first.storage, second.storage, rtol=0, atol=0)
    assert torch.isfinite(first.storage).all()
    assert not first.partial_mean.count_nonzero()
    assert not first.partial_var.count_nonzero()
    for field in ("mean", "var"):
        torch.testing.assert_close(
            getattr(first, f"block_{field}"),
            inputs[f"prior_{field}"].bfloat16().float(),
            rtol=0,
            atol=0,
        )
    first.storage.fill_(torch.nan)
    assert torch.isfinite(second.storage).all()


@pytest.mark.parametrize(
    "invalid", ["width", "dtype", "empty", "weight", "bias", "beta", "epsilon"]
)
def test_reject_unsupported_inputs(invalid):
    inputs = _make_inputs(3)
    state = _initial(inputs, _PARAMETERS)
    arguments = {
        "x": inputs["x"],
        "weight": inputs["weight"],
        "bias": inputs["bias"],
        "state": state,
    }
    if invalid == "width":
        arguments["x"] = inputs["x"][:, :48]
    elif invalid == "dtype":
        arguments["x"] = inputs["x"].float()
    elif invalid == "empty":
        arguments["x"] = inputs["x"][:0]
    elif invalid in ("weight", "bias"):
        arguments[invalid] = inputs[invalid].float()
    elif invalid == "beta":
        arguments["beta1"] = 1.0
    else:
        arguments["eps"] = float("nan")
    with pytest.raises(ValueError):
        tdn.native_tdn_forward(**arguments)


@pytest.mark.parametrize("invalid", ["size", "dtype", "strided", "position"])
def test_reject_invalid_packed_state(invalid):
    storage = torch.empty(tdn.TDN_STATE_SIZE, device="cuda")
    position = 0
    if invalid == "size":
        storage = storage[:-1]
    elif invalid == "dtype":
        storage = storage.bfloat16()
    elif invalid == "strided":
        storage = torch.empty(tdn.TDN_STATE_SIZE * 2, device="cuda")[::2]
    else:
        position = -1
    with pytest.raises(ValueError):
        tdn.PackedTDNState.from_storage(storage, position)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
