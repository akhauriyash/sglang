"""Public native ShortConv goldens, dispatch guards, and optional learned replay."""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
import weakref
from functools import cache
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _ROOT / "python/sglang/srt/models"
_PACKAGE = "_xllm_shortconv_cuda"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(_MODELS)]
sys.modules.setdefault(_PACKAGE, _package)
reference = importlib.import_module(f"{_PACKAGE}.xllm_gekko_reference")
_ci_spec = importlib.util.spec_from_file_location(
    "_xllm_ci_register", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cuda_ci = _ci.register_cuda_ci
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Native BF16 ShortConv requires NVIDIA CUDA",
)
_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/xllm_shortconv_native_golden.json").read_text()
)


def _helper():
    return importlib.import_module(f"{_PACKAGE}.xllm_shortconv")


def _digest(tensor):
    values = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _inputs(dimension, width, length):
    index = torch.arange(length * dimension, dtype=torch.int64).reshape(
        length, dimension
    )
    x = ((((index % 251) * 17) % 251 - 125).float() / 32).to(torch.bfloat16)
    index = torch.arange(dimension * width, dtype=torch.int64).reshape(dimension, width)
    weight = ((((index % 97) * 11) % 97 - 48).float() / 16).to(torch.bfloat16)
    index = torch.arange((width - 1) * dimension, dtype=torch.int64).reshape(
        width - 1, dimension
    )
    history = ((((index % 127) * 7) % 127 - 63).float() / 32).to(torch.bfloat16)
    return tuple(t.cuda() for t in (x, weight, history))


def _strided(tensor):
    storage = tensor.new_full(tuple(size * 2 for size in tensor.shape), torch.nan)
    view = storage[tuple(slice(None, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


@pytest.mark.parametrize(
    "case",
    _GOLDEN["cases"],
    ids=lambda c: f"d{c['dimension']}-w{c['width']}-t{c['length']}",
)
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_public_native_shortconv_golden(case, strided):
    x, weight, history = _inputs(case["dimension"], case["width"], case["length"])
    if strided:
        x, weight, history = map(_strided, (x, weight, history))
    snapshots = [tensor.clone() for tensor in (x, weight, history)]
    y, state = _helper().native_shortconv_forward(x, weight, history)
    assert _digest(y) == case["output_sha256"]
    assert _digest(state) == case["state_sha256"]
    for original, snapshot in zip((x, weight, history), snapshots, strict=True):
        assert torch.equal(original, snapshot)
    if state.numel():
        state.zero_()
        assert torch.equal(history, snapshots[-1])


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@torch.no_grad()
def test_empty_returns_existing_tensors(width):
    x, weight, history = _inputs(64, width, 0)
    y, state = _helper().native_shortconv_forward(x, weight, history)
    assert y is x and state is history
    module = reference.NormalizedShortConv(64, width).cuda().bfloat16()
    y, state = module(x, history)
    assert y is x and state is history


@torch.no_grad()
def test_module_dispatch_and_inventory(monkeypatch):
    x, weight, history = _inputs(512, 4, 3)
    module = reference.NormalizedShortConv(512, 4).cuda().bfloat16()
    module.weight.copy_(weight)
    helper = _helper()
    actual_function = helper.native_shortconv_forward
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return actual_function(*args, **kwargs)

    monkeypatch.setattr(helper, "native_shortconv_forward", observed)
    output, state = module(x, history)
    assert calls == [True]
    assert set(module.state_dict()) == {"weight"}
    expected = next(
        c
        for c in _GOLDEN["cases"]
        if (c["dimension"], c["width"], c["length"]) == (512, 4, 3)
    )
    assert _digest(output) == expected["output_sha256"]
    assert _digest(state) == expected["state_sha256"]


@pytest.mark.parametrize("gradient", ["x", "weight", "history"])
def test_gradient_bearing_operand_keeps_autograd(monkeypatch, gradient):
    x, weight, history = _inputs(64, 4, 2)
    module = reference.NormalizedShortConv(64, 4).cuda().bfloat16()
    module.weight.requires_grad_(False)
    with torch.no_grad():
        module.weight.copy_(weight)
    values = {"x": x, "weight": module.weight, "history": history}
    values[gradient].requires_grad_(True)

    def forbidden(*args):
        raise AssertionError("Inference helper must not consume gradient-bearing state")

    monkeypatch.setattr(_helper(), "native_shortconv_forward", forbidden)
    output, state = module(x, history)
    output.float().square().sum().backward()
    assert values[gradient].grad is not None
    assert torch.isfinite(values[gradient].grad).all()
    assert state.shape == history.shape


@torch.no_grad()
@pytest.mark.parametrize(
    "fallback", ["autocast", "float32_input", "float32_weight", "float32_history"]
)
def test_ineligible_inputs_keep_torch_fallback(monkeypatch, fallback):
    x, weight, history = _inputs(64, 4, 2)
    module = reference.NormalizedShortConv(64, 4).cuda().bfloat16()
    module.weight.copy_(weight)
    if fallback == "float32_input":
        x = x.float()
    elif fallback == "float32_weight":
        module.float()
    elif fallback == "float32_history":
        history = history.float()

    def forbidden(*args):
        raise AssertionError("Ineligible input used the inference helper")

    monkeypatch.setattr(_helper(), "native_shortconv_forward", forbidden)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=fallback == "autocast"):
        output, state = module(x, history)
    assert output.dtype == x.dtype
    assert state.shape == history.shape
    assert torch.isfinite(output).all()


def _pooled_expected(x, weight, pool, slots):
    expected_pool = pool.clone()
    output = torch.zeros_like(x)
    for row, slot in enumerate(slots):
        if not 0 <= slot < len(pool):
            continue
        y, state = _helper().native_shortconv_forward(
            x[row : row + 1], weight, pool[slot]
        )
        output[row] = y[0]
        expected_pool[slot] = state
    return output, expected_pool


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("dimension", [33, 512, 2048])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_pooled_decode_matches_functional_reference(width, dimension, strided):
    x, weight, history = _inputs(dimension, width, 4)
    pool = history[None].repeat(5, 1, 1)
    slots = [3, -1, 0, 5]
    if strided:
        x, weight = _strided(x), _strided(weight)
    # Padding may contain NaNs; it must not affect valid outputs or state.
    x[1].fill_(torch.nan)
    x[3].fill_(torch.nan)
    taps = weight.float().softmax(-1).bfloat16()
    if strided:
        taps = _strided(taps)
    expected, expected_pool = _pooled_expected(x, weight, pool, slots)
    actual = _helper().native_shortconv_decode_pooled(
        x, taps, pool, torch.tensor(slots, device="cuda", dtype=torch.int64)
    )
    assert torch.equal(actual, expected)
    assert torch.equal(pool, expected_pool)
    # The caller adds the BF16 residual after the convolution's BF16 cast.
    assert torch.equal((actual + x)[[0, 2]], (expected + x)[[0, 2]])


@torch.no_grad()
def test_pooled_decode_graph_reorder_padding_and_slot_reuse():
    x, weight, history = _inputs(512, 4, 4)
    module = reference.NormalizedShortConv(512, 4).cuda().bfloat16()
    module.weight.copy_(weight)
    pool = history[None].repeat(6, 1, 1)
    slots = torch.tensor([3, 0, 1, -1], device="cuda", dtype=torch.int32)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        module.decode_pooled(x, pool, slots)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = module.decode_pooled(x, pool, slots)
    assert len(module._captured_normalized_weights) == 1
    pool.copy_(history[None].expand_as(pool))
    for step, indices in enumerate(([3, 0, 1, -1], [1, 3, -1, 0], [6, -3, 4, 2])):
        if step == 2:
            pool[4].zero_()  # A newly allocated/reused request starts clean.
        x.copy_(_inputs(512, 4, 4)[0] + step)
        slots.copy_(torch.tensor(indices, device="cuda", dtype=slots.dtype))
        for row, slot in enumerate(indices):
            if not 0 <= slot < len(pool):
                x[row].fill_(torch.nan)
        expected, expected_pool = _pooled_expected(x, weight, pool, indices)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assert torch.equal(pool, expected_pool)


@torch.no_grad()
def test_cached_weights_survive_graphs_and_refresh_after_eager_changes():
    x, weight, history = _inputs(64, 4, 1)
    module = reference.NormalizedShortConv(64, 4).cuda().bfloat16()
    module.weight.copy_(weight)
    pool = history[None].clone()
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    graphs, outputs, expected, refs = [], [], [], []
    for change in (0.0, 2.0):
        module.weight[:, 0].add_(change)
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            module.decode_pooled(x, pool, slots)
        torch.cuda.current_stream().wait_stream(warmup)
        refs.append(weakref.ref(module._normalized_weight()))
        expected.append(_pooled_expected(x, module.weight, history[None], [0])[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module.decode_pooled(x, pool, slots)
        graphs.append(graph)
        outputs.append(output)
    assert len(module._captured_normalized_weights) == 2
    module.float()  # Invalidate the active cache; captured constants stay owned.
    assert module._normalized_weight_cache is None
    assert all(ref() is not None for ref in refs)
    for graph, actual, wanted in zip(graphs, outputs, expected, strict=True):
        pool.copy_(history[None])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, wanted)


@torch.no_grad()
def test_pooled_decode_rejects_overlapping_history_layout():
    x, weight, history = _inputs(64, 4, 2)
    pool = history[None].expand(3, -1, -1)
    slots = torch.tensor([0, 1], device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        _helper().native_shortconv_decode_pooled(
            x, weight.float().softmax(-1).bfloat16(), pool, slots
        )


@cache
def _learned_fixture():
    from safetensors.torch import load_file

    directory = os.environ.get("XLLM_SHORTCONV_ORACLE_FIXTURE")
    if not directory:
        pytest.skip("Optional learned native ShortConv fixtures were not provided")
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text())
    tensors_path = path / "learned.safetensors"
    with tensors_path.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == manifest["learned_fixture_sha256"]
        )
    assert manifest["captured"] and manifest["native_sources_unchanged"]
    return load_file(tensors_path, device="cuda")


@pytest.mark.parametrize("name", ["q", "k", "v"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("pieces", [[4097], [31, 225, 3840, 1], [2048, 2048, 1]])
@torch.no_grad()
def test_optional_learned_native_module_partitions(name, strided, pieces):
    fixture = _learned_fixture()
    x, weight, history = (
        fixture[f"{name}.{key}"] for key in ("x", "weight", "history")
    )
    if strided:
        x = _strided(x)
    module = (
        reference.NormalizedShortConv(x.shape[-1], weight.shape[-1]).cuda().bfloat16()
    )
    module.weight.copy_(weight)
    outputs, offset = [], 0
    for count in pieces:
        snapshot = history.clone()
        output, state = module(x[offset : offset + count], history)
        assert torch.equal(history, snapshot)
        outputs.append(output)
        history = state
        offset += count
    assert torch.equal(torch.cat(outputs), fixture[f"{name}.expected"])
    assert torch.equal(history, fixture[f"{name}.state"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
