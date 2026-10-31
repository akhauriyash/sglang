"""Pure CPU validation of serving qualification's token/logprob alignment."""

import copy
import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[4]
qualify = load_module(
    "_xllm_serve_qualify_cpu", ROOT / "tools/xllm_hybrid/serve_qualify.py"
)
register_cpu_ci = load_module(
    "_xllm_serve_qualify_ci", ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def fixture():
    return {
        "prompt_ids": [0, 4, 7],
        "continuation_ids": [8, 2, 9],
        "scores": [
            {"position": 0, "token_id": 4, "logprob": -0.2, "argmax": 4},
            {"position": 2, "token_id": 8, "logprob": -0.4, "argmax": 8},
            {"position": 3, "token_id": 2, "logprob": -1.3, "argmax": 5},
        ],
        "greedy_continuation_ids": [8, 5, 9],
        "greedy_logprobs": [-0.4, -0.6, -0.9],
    }


def test_fixture_validation(fixture):
    qualify.validate_fixture(fixture, 10)
    malformed = copy.deepcopy(fixture)
    malformed["scores"][1]["position"] += 1
    with pytest.raises(ValueError, match="alignment"):
        qualify.validate_fixture(malformed, 10)


def test_greedy_trace_matches_prompt_final_native_score(fixture):
    fixture["greedy_continuation_ids"][0] = 9
    with pytest.raises(ValueError, match="native argmax"):
        qualify.validate_fixture(fixture, 10)
    fixture["greedy_continuation_ids"][0] = 8
    fixture["greedy_logprobs"][0] = -0.8
    with pytest.raises(ValueError, match="native score"):
        qualify.validate_fixture(fixture, 10)


@pytest.mark.parametrize("token", [-1, 10, True])
def test_fixture_invalid_token_ids(fixture, token):
    fixture["prompt_ids"][1] = token
    with pytest.raises(ValueError, match="invalid token"):
        qualify.validate_fixture(fixture, 10)


def test_prompt_alignment_is_source_position_plus_one(fixture):
    response = {
        "meta_info": {
            "input_token_logprobs": [
                [None, 0, None],
                [-0.2, 4, None],
                [-3.5, 7, None],
                [-0.4, 8, None],
                [-1.3, 2, None],
                [-0.8, 9, None],
            ]
        }
    }
    result = qualify.prompt_comparison(response, fixture, 1e-5)
    assert result["passed"]
    assert result["actual_logprobs"] == [-0.2, -0.4, -1.3]
    response["meta_info"]["input_token_logprobs"][3][1] = 2
    with pytest.raises(ValueError, match="token IDs"):
        qualify.prompt_comparison(response, fixture, 1e-5)


def test_prompt_null_and_length_alignment(fixture):
    response = {"meta_info": {"input_token_logprobs": [[None, 0, None]]}}
    with pytest.raises(ValueError, match="length"):
        qualify.prompt_comparison(response, fixture, 1e-5)


def test_output_target_logprob_is_independent_of_generated_argmax(fixture):
    score = fixture["scores"][-1]
    response = {
        "output_ids": [5],
        "meta_info": {"output_token_ids_logprobs": [[[-1.3, 2, None]]]},
    }
    result = qualify.output_comparison(response, score, 1e-5)
    assert result["passed"]
    response["output_ids"] = [2]
    result = qualify.output_comparison(response, score, 1e-5)
    assert not result["passed"]
    assert result["max_abs"] == 0


def test_greedy_divergence_does_not_compare_different_conditionals(fixture):
    response = {
        "output_ids": [8, 4, 3],
        "meta_info": {
            "output_token_logprobs": [
                [-0.4, 8, None],
                [-0.1, 4, None],
                [-0.01, 3, None],
            ]
        },
    }
    result = qualify.decode_comparison(response, fixture, 1e-5)
    assert not result["passed"]
    assert result["matching_prefix_tokens"] == 1
    assert result["matching_prefix_logprob_error"]["count"] == 1


def test_greedy_exact_and_nonfinite_rejection(fixture):
    response = {
        "output_ids": [8, 5, 9],
        "meta_info": {
            "output_token_logprobs": [
                [-0.4, 8, None],
                [-0.6, 5, None],
                [-0.9, 9, None],
            ]
        },
    }
    assert qualify.decode_comparison(response, fixture, 1e-5)["passed"]
    response["meta_info"]["output_token_logprobs"][0][0] = math.nan
    result = qualify.decode_comparison(response, fixture, 1e-5)
    assert not result["passed"]
    assert not result["matching_prefix_logprob_error"]["finite"]


def test_response_count_mismatch():
    with pytest.raises(ValueError, match="response count"):
        qualify.response_list({"output_ids": [1]}, 2)


def test_gekko_is_an_explicit_supported_serving_architecture():
    assert (
        qualify.validate_architecture({"architectures": ["XllmGekkoForCausalLM"]})
        == "XllmGekkoForCausalLM"
    )
    with pytest.raises(ValueError, match="exported xLLM"):
        qualify.validate_architecture({"architectures": ["LlamaForCausalLM"]})


def test_nll_mean_gate_is_not_replaced_by_maximum_gate():
    metrics = qualify.error_metrics([-1.0] * 20, [-1.006] * 20, 0.08)
    assert metrics["max_abs"] < 0.08
    assert metrics["mean_abs"] > 0.005
    assert not metrics["passed"]
    assert metrics["signed_mean_nll_difference"] == pytest.approx(0.006)
    maximum_failure = qualify.error_metrics([-1.0] * 100, [-1.0] * 99 + [-1.09], 0.08)
    assert maximum_failure["mean_abs"] < 0.005
    assert not maximum_failure["passed"]
    assert qualify.error_metrics([-1.0] * 10, [-1.004] * 10, 0.08)["passed"]


@pytest.mark.parametrize(
    "maximum,mean",
    [(0.09, 0.005), (0.08, 0.006), (math.inf, 0.005), (0.08, math.nan), (0.0, 0.005)],
)
def test_cli_thresholds_cannot_relax_historical_gates(maximum, mean):
    with pytest.raises(ValueError, match="NLL error"):
        qualify.validate_thresholds(maximum, mean)


def test_fixture_requires_qualified_case_at_its_actual_prompt_length(fixture):
    native = {"checks": [{"prompt_tokens": 3, "native_oracle": {"passed": True}}]}
    qualify.validate_fixture_case(fixture, native)
    native["checks"][0]["prompt_tokens"] = 4
    with pytest.raises(ValueError, match="3-token fixture"):
        qualify.validate_fixture_case(fixture, native)
    native["checks"][0]["prompt_tokens"] = 3
    native["checks"][0]["native_oracle"]["passed"] = False
    with pytest.raises(ValueError, match="qualified native oracle"):
        qualify.validate_fixture_case(fixture, native)


def test_reference_logit_gate_stays_distinct_from_engine_logprobs():
    native = {
        "checks": [
            {
                "prompt_tokens": 3,
                "prefill": {"relative_l2": 0.008, "finite": True},
                "partitioned_prefill": {"relative_l2": 0.009, "finite": True},
                "decode": [{"relative_l2": 0.011, "finite": True}],
            }
        ]
    }
    result = qualify.reference_selected_logit_metrics(native)
    assert result["applicable"]
    assert not result["passed"]
    assert len(result["checks"]) == 3
    assert "not Engine raw logits" in result["scope"]
    native["checks"][0]["decode"][0]["relative_l2"] = 0.009
    assert qualify.reference_selected_logit_metrics(native)["passed"]
    native["checks"][0]["decode"][0]["finite"] = False
    assert not qualify.reference_selected_logit_metrics(native)["passed"]
    missing = qualify.reference_selected_logit_metrics({"checks": []})
    assert not missing["applicable"]
    assert missing["passed"] is None


def test_native_reference_failure_does_not_discard_engine_evidence(fixture):
    native = {
        "passed": False,
        "checks": [{"prompt_tokens": 3, "native_oracle": {"passed": True}}],
    }
    # A failed reference-model parity run still supplies a qualified native
    # oracle. Engine comparisons remain meaningful and must be recorded.
    qualify.validate_fixture_case(fixture, native)
    status = qualify.qualification_status([{"passed": True}], native, {"passed": True})
    assert status["engine_checks_passed"]
    assert not status["native_reference_report_passed"]
    assert not status["passed"]
    native["passed"] = True
    assert qualify.qualification_status([{"passed": True}], native, {"passed": True})[
        "passed"
    ]
    assert not qualify.qualification_status(
        [{"passed": False}], native, {"passed": True}
    )["passed"]
    assert not qualify.qualification_status([], native, {"passed": True})["passed"]
    changed_sources = qualify.qualification_status(
        [{"passed": True}],
        native,
        {"passed": True},
        source_provenance_unchanged=False,
    )
    assert changed_sources["engine_checks_passed"]
    assert not changed_sources["source_provenance_unchanged"]
    assert not changed_sources["passed"]


@pytest.mark.parametrize(
    "name",
    [
        f"python/sglang/srt/models/{module}"
        for module in ("xllm_rmsnorm.py", "xllm_tdn.py", "xllm_sca.py")
    ]
    + [
        f"python/sglang/kernels/ops/attention/flash_attn/cute/{module}"
        for module in ("interface.py", "flash_fwd_sm90.py", "softmax.py")
    ],
)
def test_operator_source_changes_invalidate_serving_provenance(monkeypatch, name):
    before = qualify.capture_source_provenance()
    original_hash = qualify.sha256_file(ROOT / name)
    assert before["port_sources"][name] == original_hash
    hash_file = qualify.sha256_file
    monkeypatch.setattr(
        qualify,
        "sha256_file",
        lambda path: (
            "changed-operator-source" if path == ROOT / name else hash_file(path)
        ),
    )
    after = qualify.capture_source_provenance()
    assert before["port_sources"][name] == original_hash
    assert after["port_sources"][name] != original_hash
    assert not qualify.qualification_status(
        [{"passed": True}],
        {"passed": True},
        {"passed": True},
        source_provenance_unchanged=before == after,
    )["passed"]


def test_telemetry_finishes_before_engine_terminates_children():
    resources = Mock()
    qualify.finish_resources(
        resources.run, resources.wandb, resources.engine, exit_code=1
    )
    assert resources.mock_calls == [
        call.run.finish(exit_code=1),
        call.wandb.teardown(exit_code=1),
        call.engine.shutdown(),
    ]


def test_engine_shutdown_runs_even_if_telemetry_finish_fails():
    resources = Mock()
    resources.run.finish.side_effect = ConnectionResetError("telemetry disconnected")
    with pytest.raises(ConnectionResetError, match="telemetry disconnected"):
        qualify.finish_resources(
            resources.run, resources.wandb, resources.engine, exit_code=1
        )
    assert resources.mock_calls == [
        call.run.finish(exit_code=1),
        call.wandb.teardown(exit_code=1),
        call.engine.shutdown(),
    ]


def test_engine_shutdown_runs_even_if_wandb_teardown_fails():
    resources = Mock()
    resources.wandb.teardown.side_effect = ConnectionResetError("service disconnected")
    with pytest.raises(ConnectionResetError, match="service disconnected"):
        qualify.finish_resources(
            resources.run, resources.wandb, resources.engine, exit_code=1
        )
    assert resources.mock_calls == [
        call.run.finish(exit_code=1),
        call.wandb.teardown(exit_code=1),
        call.engine.shutdown(),
    ]


@pytest.fixture
def full_prompt_fixture(fixture):
    fixture["prompt_logprobs"] = [-0.2, -3.5, -0.4]
    fixture["scores"].append(
        {"position": 4, "token_id": 9, "logprob": -0.8, "argmax": 9}
    )
    return fixture


@pytest.fixture
def forced_response():
    return {
        "meta_info": {
            "input_token_logprobs": [
                [None, 0, None],
                [-0.2, 4, None],
                [-3.5, 7, None],
                [-0.4, 8, None],
                [-1.3, 2, None],
                [-0.8, 9, None],
            ]
        }
    }


def test_full_prompt_covers_boundary_target_without_duplicates(
    full_prompt_fixture, forced_response
):
    qualify.validate_fixture(full_prompt_fixture, 10)
    result = qualify.prompt_comparison(forced_response, full_prompt_fixture, 0.08)
    assert result["passed"]
    assert result["scores_scope"] == "full_prompt_and_forced_decode"
    assert result["source_positions"] == [0, 1, 2, 3, 4]
    assert result["native_logprobs"] == [-0.2, -3.5, -0.4, -1.3, -0.8]
    assert result["count"] == 5
    assert result["prompt_nll"]["count"] == 3
    assert result["forced_decode_nll"]["count"] == 2
    # Prompt source position L-1 predicts continuation[0], not prompt[L-1].
    forced_response["meta_info"]["input_token_logprobs"][3][0] -= 0.1
    assert not qualify.prompt_comparison(forced_response, full_prompt_fixture, 0.08)[
        "prompt_nll"
    ]["passed"]


def test_full_prompt_detects_an_error_outside_legacy_selected_positions(
    full_prompt_fixture, forced_response
):
    forced_response["meta_info"]["input_token_logprobs"][2][0] -= 0.09
    result = qualify.prompt_comparison(forced_response, full_prompt_fixture, 0.08)
    assert not result["passed"]
    assert result["prompt_nll"]["max_abs"] > 0.08
    del full_prompt_fixture["prompt_logprobs"]
    sparse = qualify.prompt_comparison(forced_response, full_prompt_fixture, 0.08)
    assert sparse["passed"]
    assert sparse["scores_scope"] == "selected_prompt_and_forced_decode"
    assert sparse["prompt_nll"]["count"] == 2


@pytest.mark.parametrize("segment", ["prompt_nll", "forced_decode_nll"])
def test_prompt_and_decode_mean_gates_cannot_dilute_each_other(
    full_prompt_fixture, forced_response, segment
):
    rows = forced_response["meta_info"]["input_token_logprobs"]
    indices = range(1, 4) if segment == "prompt_nll" else range(4, 6)
    for index in indices:
        rows[index][0] -= 0.006
    result = qualify.prompt_comparison(forced_response, full_prompt_fixture, 0.08)
    assert result["mean_abs"] < 0.005
    assert result[segment]["mean_abs"] > 0.005
    assert not result["passed"]


@pytest.mark.parametrize("values", [None, [], [-0.2, -3.5], [-0.2] * 4])
def test_full_prompt_requires_one_value_per_source_logit(full_prompt_fixture, values):
    full_prompt_fixture["prompt_logprobs"] = values
    with pytest.raises(ValueError, match="every prompt logit"):
        qualify.validate_fixture(full_prompt_fixture, 10)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.1, True])
def test_full_prompt_rejects_invalid_probabilities(full_prompt_fixture, value):
    full_prompt_fixture["prompt_logprobs"][1] = value
    with pytest.raises(ValueError, match="finite and nonpositive"):
        qualify.validate_fixture(full_prompt_fixture, 10)


def test_full_prompt_must_agree_with_legacy_selected_scores(full_prompt_fixture):
    full_prompt_fixture["prompt_logprobs"][0] = -1.0
    with pytest.raises(ValueError, match="selected and full prompt"):
        qualify.validate_fixture(full_prompt_fixture, 10)


def test_full_prompt_fixture_requires_complete_forced_decode(full_prompt_fixture):
    full_prompt_fixture["scores"].pop()
    with pytest.raises(ValueError, match="every forced decode position"):
        qualify.validate_fixture(full_prompt_fixture, 10)


@pytest.fixture
def original_prompt_response():
    return {
        "meta_info": {
            "input_token_logprobs": [
                [None, 0, None],
                [-0.2, 4, None],
                [-3.5, 7, None],
            ],
            "output_token_ids_logprobs": [[[-0.4, 8, None]]],
        }
    }


def test_original_prompt_combines_input_and_final_output_target(
    full_prompt_fixture, original_prompt_response
):
    result = qualify.original_prompt_comparison(
        original_prompt_response, full_prompt_fixture, 0.08
    )
    assert result["passed"]
    assert result["count"] == 3
    assert result["source_positions"] == [0, 1, 2]
    assert result["actual_logprobs"] == full_prompt_fixture["prompt_logprobs"]
    assert result["scores_scope"] == "full_prompt_original_length"
    original_prompt_response["meta_info"]["output_token_ids_logprobs"][0][0][0] -= 0.09
    assert not qualify.original_prompt_comparison(
        original_prompt_response, full_prompt_fixture, 0.08
    )["passed"]


def test_single_token_original_prompt_uses_only_output_target(
    full_prompt_fixture, original_prompt_response
):
    full_prompt_fixture["prompt_ids"] = [0]
    full_prompt_fixture["prompt_logprobs"] = [-0.4]
    original_prompt_response["meta_info"]["input_token_logprobs"] = [[None, 0, None]]
    result = qualify.original_prompt_comparison(
        original_prompt_response, full_prompt_fixture, 0.08
    )
    assert result["passed"]
    assert result["count"] == 1
    assert result["source_positions"] == [0]


@pytest.mark.parametrize(
    "target", [None, [], [[]], [[[None, 8, None]]], [[[-0.4, 2, None]]]]
)
def test_original_prompt_rejects_missing_or_wrong_final_target(
    full_prompt_fixture, original_prompt_response, target
):
    original_prompt_response["meta_info"]["output_token_ids_logprobs"] = target
    with pytest.raises(ValueError, match="target"):
        qualify.original_prompt_comparison(
            original_prompt_response, full_prompt_fixture, 0.08
        )


def test_original_prompt_rejects_missing_input_target(
    full_prompt_fixture, original_prompt_response
):
    original_prompt_response["meta_info"]["input_token_logprobs"][1][0] = None
    with pytest.raises(ValueError, match="null positions"):
        qualify.original_prompt_comparison(
            original_prompt_response, full_prompt_fixture, 0.08
        )


def test_original_prompt_rejects_shifted_input_targets(
    full_prompt_fixture, original_prompt_response
):
    original_prompt_response["meta_info"]["input_token_logprobs"][1][1] = 7
    with pytest.raises(ValueError, match="input token scores"):
        qualify.original_prompt_comparison(
            original_prompt_response, full_prompt_fixture, 0.08
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
