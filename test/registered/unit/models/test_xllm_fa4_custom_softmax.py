"""CPU contract tests for the optional internal FA4 softmax implementation.

Extract the small host helpers and default arithmetic methods so these checks
also run without CuTe or CUDA installed. GPU numerical tests cover compilation.
"""

import ast
import hashlib
import importlib.util
import inspect
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
CUTE = ROOT / "python/sglang/kernels/ops/attention/flash_attn/cute"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    body.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    )
    assert len(body) == len(names) + 1
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - trusted repository source
    return namespace


class SoftmaxBase:
    pass


HOST = load_functions(
    CUTE / "interface.py",
    {
        "_softmax_impl_cache_key",
        "_validate_custom_softmax",
        "_validate_custom_softmax_paged_decode",
    },
    {
        "Softmax": SoftmaxBase,
        "cache": cache,
        "hashlib": hashlib,
        "inspect": inspect,
        "torch": torch,
    },
)

ci_spec = importlib.util.spec_from_file_location(
    "_xllm_fa4_custom_softmax_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_module = importlib.util.module_from_spec(ci_spec)
ci_spec.loader.exec_module(ci_module)
register_cpu_ci = ci_module.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_cache_identity_tracks_class_and_custom_source(tmp_path, monkeypatch):
    path = tmp_path / "custom_softmax.py"
    path.write_text("class Custom(SoftmaxBase):\n    arithmetic_version = 1\n")

    def load():
        # Compile source directly to avoid a timestamp-based bytecode cache.
        spec = importlib.util.spec_from_file_location("_softmax_identity_test", path)
        module = importlib.util.module_from_spec(spec)
        module.SoftmaxBase = SoftmaxBase
        monkeypatch.setitem(sys.modules, spec.name, module)
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102 - test-owned source
        return module.Custom

    key = HOST["_softmax_impl_cache_key"]
    first = load()
    first_key = key(first)
    assert key(first) == first_key
    assert first_key[:2] == ("_softmax_identity_test", "Custom")
    assert first_key[2] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("class Custom(SoftmaxBase):\n    arithmetic_version = 2\n")
    second_key = key(load())
    assert second_key[:2] == first_key[:2]
    assert second_key[2] != first_key[2]


@pytest.mark.parametrize("implementation", [None, object, lambda: None])
def test_cache_rejects_non_softmax_factories(implementation):
    with pytest.raises(TypeError, match="Softmax subclass"):
        HOST["_softmax_impl_cache_key"](implementation)


def test_cache_rejects_function_local_class():
    class Local(SoftmaxBase):
        pass

    with pytest.raises(TypeError, match="module-level"):
        HOST["_softmax_impl_cache_key"](Local)


def tensors():
    return [
        SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            shape=(1, 31, heads, 64),
            requires_grad=False,
        )
        for heads in (32, 8, 8)
    ]


def validate(inputs, **overrides):
    kwargs = {
        "arch": 90,
        "causal": True,
        "num_splits": 1,
        "return_lse": False,
        "unsupported": [],
    }
    kwargs.update(overrides)
    HOST["_validate_custom_softmax"](*inputs, **kwargs)


def test_supported_fixed_and_varlen_shapes():
    inputs = tensors()
    validate(inputs)
    for tensor in inputs:
        tensor.shape = tensor.shape[1:]
    validate(inputs)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"arch": 80}, "SM90"),
        ({"arch": 100}, "SM90"),
        ({"causal": False}, "causal"),
        ({"num_splits": 0}, "one split"),
        ({"num_splits": 2}, "one split"),
        ({"return_lse": True}, "no LSE"),
        ({"unsupported": ["page_table"]}, "page_table"),
        ({"unsupported": ["score_mod", "learnable_sink"]}, "score_mod"),
    ],
)
def test_unsupported_modes_raise(override, match):
    with pytest.raises(ValueError, match=match):
        validate(tensors(), **override)


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("is_cuda", False, "CUDA BF16"),
        ("dtype", torch.float16, "CUDA BF16"),
        ("shape", (1, 31, 8, 128), "dimensions 64"),
        ("requires_grad", True, "forward-only"),
    ],
)
def test_unsupported_tensor_contracts_raise(attribute, value, match):
    inputs = tensors()
    setattr(inputs[1], attribute, value)
    with pytest.raises(ValueError, match=match):
        validate(inputs)


def test_missing_q_is_rejected():
    inputs = tensors()
    inputs[0] = None
    with pytest.raises(ValueError, match="CUDA BF16"):
        validate(inputs)


def test_custom_paging_contract_accepts_only_single_query_single_page():
    validate_paged = HOST["_validate_custom_softmax_paged_decode"]
    q = torch.empty((2, 1, 32, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 8192, 8, 64), dtype=torch.bfloat16)
    slots = torch.zeros((2, 1), dtype=torch.int32)
    lengths = torch.zeros((2,), dtype=torch.int32)
    validate_paged(q, k, k, slots, lengths)
    cases = [
        (q.expand(2, 2, 32, 64), k, k, slots, lengths),
        (q, k[:, :128], k[:, :128], slots, lengths),
        (q, k[:, :257], k[:, :257], slots, lengths),
        (q, k, k, None, lengths),
        (q, k, k, slots, None),
        (q, k, k, slots.expand(2, 2), lengths),
        (q, k, k, slots.long(), lengths),
        (q, k, k, slots, torch.empty(4, dtype=torch.int32)[::2]),
    ]
    for args in cases:
        with pytest.raises(ValueError, match="Custom softmax paging"):
            validate_paged(*args)


def test_default_arithmetic_and_sm100_call_contract_are_preserved():
    utils = SimpleNamespace(fadd_reduce=Mock(return_value=7))
    namespace = load_functions(
        CUTE / "softmax.py",
        {"_compute_row_sum", "_update_row_sum"},
        {"utils": utils},
    )
    cls = type(
        "DefaultArithmetic",
        (),
        {name: namespace[name] for name in ("_compute_row_sum", "_update_row_sum")},
    )
    instance = cls()
    instance.arch = 80
    values = object()
    assert instance._update_row_sum(values) == 7
    utils.fadd_reduce.assert_called_with(values, None, arch=80)
    instance._update_row_sum(values, prior_sum=3.0, scale=0.5)
    utils.fadd_reduce.assert_called_with(values, 1.5, arch=80)
    # SoftmaxSm100 retains the old keyword-based call into this method.
    instance.arch = 100
    instance._compute_row_sum(values, init_val=2.5)
    utils.fadd_reduce.assert_called_with(values, 2.5, arch=100)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
