"""Qualify the actual SGLang Engine against saved native xLLM fixtures.

Run after qualify.py on an independently allocated CUDA GPU. Forced-token
scoring checks prompt/output alignment; native greedy traces exercise real
decode, batching, reordered requests, and reuse of completed request slots.
The output directory must be new. Timing is diagnostic, not a benchmark claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

NLL_MEAN_LIMIT = 0.005
NLL_MAX_LIMIT = 0.08
LOGIT_RELATIVE_L2_LIMIT = 0.01
SUPPORTED_ARCHITECTURES = frozenset(
    {"XllmDenseForCausalLM", "XllmInklingForCausalLM", "XllmGekkoForCausalLM"}
)
PORT_SOURCE_PATHS = (
    "python/sglang/srt/models/xllm_hybrid.py",
    "python/sglang/srt/models/xllm_reference.py",
    "python/sglang/srt/models/xllm_gekko_reference.py",
    "python/sglang/srt/models/xllm_rmsnorm.py",
    "python/sglang/srt/models/xllm_tdn.py",
    "python/sglang/srt/models/xllm_sca.py",
    "python/sglang/srt/models/xllm_gekko_decode.py",
    "python/sglang/srt/models/xllm_rope.py",
    "python/sglang/srt/models/xllm_gemm.py",
    "python/sglang/srt/models/xllm_shortconv.py",
    "python/sglang/kernels/ops/attention/flash_attn/cute/interface.py",
    "python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py",
    "python/sglang/kernels/ops/attention/flash_attn/cute/softmax.py",
    "python/sglang/srt/configs/xllm_hybrid.py",
    "python/sglang/srt/configs/model_config.py",
    "python/sglang/srt/configs/__init__.py",
    "python/sglang/srt/layers/attention/attention_registry.py",
    "python/sglang/srt/layers/attention/linear/xllm_inkling_backend.py",
    "python/sglang/srt/layers/attention/linear/xllm_gekko_backend.py",
    "python/sglang/srt/layers/attention/base_attn_backend.py",
    "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
    "python/sglang/srt/layers/attention/linear/short_conv_backend.py",
    "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
    "python/sglang/srt/mem_cache/memory_pool.py",
    "python/sglang/srt/mem_cache/mamba_slot_fused.py",
    "python/sglang/srt/utils/hf_transformers/common.py",
)


def validate_architecture(config: dict) -> str:
    architectures = config.get("architectures", [])
    if len(architectures) != 1 or architectures[0] not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Expected an exported xLLM dense, Inkling, or Gekko model")
    return architectures[0]


def validate_thresholds(maximum: float, mean: float) -> None:
    if not math.isfinite(maximum) or not 0 < maximum <= NLL_MAX_LIMIT:
        raise ValueError(f"Maximum NLL error must be positive and <= {NLL_MAX_LIMIT}")
    if not math.isfinite(mean) or not 0 < mean <= NLL_MEAN_LIMIT:
        raise ValueError(f"Mean NLL error must be positive and <= {NLL_MEAN_LIMIT}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "sglang_commit": commit,
        "port_sources": {name: sha256_file(root / name) for name in PORT_SOURCE_PATHS},
        "qualifier_sha256": sha256_file(Path(__file__)),
    }


def write_json(path: Path, data: Any) -> None:
    with path.open("x") as destination:
        json.dump(data, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def token_identity(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_fixture(fixture: dict, vocabulary: int) -> None:
    prompt, continuation = fixture["prompt_ids"], fixture["continuation_ids"]
    greedy, greedy_logprobs = (
        fixture["greedy_continuation_ids"],
        fixture["greedy_logprobs"],
    )
    if (
        not prompt
        or not continuation
        or len(greedy) < 2
        or len(greedy) != len(greedy_logprobs)
    ):
        raise ValueError(
            "Fixture needs prompt, forced continuation, and aligned greedy trace"
        )
    stream = prompt + continuation
    if any(
        type(token) is not int or not 0 <= token < vocabulary
        for token in stream + greedy
    ):
        raise ValueError("Fixture contains an invalid token ID")
    if not fixture["scores"]:
        raise ValueError("Fixture has no native scores")
    positions = set()
    for score in fixture["scores"]:
        position = score["position"]
        if type(position) is not int or not 0 <= position < len(stream) - 1:
            raise ValueError("Fixture source position must predict a following token")
        if position in positions or score["token_id"] != stream[position + 1]:
            raise ValueError("Fixture next-token alignment is inconsistent")
        positions.add(position)
        if not math.isfinite(score["logprob"]) or score["logprob"] > 0:
            raise ValueError("Fixture native logprob is invalid")
        if type(score["argmax"]) is not int or not 0 <= score["argmax"] < vocabulary:
            raise ValueError("Fixture native argmax is invalid")
    if any(not math.isfinite(value) or value > 0 for value in greedy_logprobs):
        raise ValueError("Fixture greedy logprobs are invalid")
    first_greedy = next(
        (score for score in fixture["scores"] if score["position"] == len(prompt) - 1),
        None,
    )
    if first_greedy is None or first_greedy["argmax"] != greedy[0]:
        raise ValueError(
            "Fixture greedy trace does not match its prompt-final native argmax"
        )
    if first_greedy["token_id"] == greedy[0] and not math.isclose(
        first_greedy["logprob"], greedy_logprobs[0], abs_tol=1e-5, rel_tol=1e-5
    ):
        raise ValueError(
            "Fixture greedy trace does not match its prompt-final native score"
        )
    if "prompt_logprobs" in fixture:
        prompt_logprobs = fixture["prompt_logprobs"]
        if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(prompt):
            raise ValueError("Fixture prompt_logprobs must score every prompt logit")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value > 0
            for value in prompt_logprobs
        ):
            raise ValueError("Fixture prompt_logprobs must be finite and nonpositive")
        for score in fixture["scores"]:
            position = score["position"]
            if position < len(prompt) and not math.isclose(
                score["logprob"],
                prompt_logprobs[position],
                abs_tol=1e-5,
                rel_tol=1e-5,
            ):
                raise ValueError("Fixture selected and full prompt scores disagree")
        if {p for p in positions if p >= len(prompt)} != set(
            range(len(prompt), len(stream) - 1)
        ):
            raise ValueError("Fixture must score every forced decode position")


def fixture_score_scope(fixture: dict) -> str:
    return (
        "full_prompt_and_forced_decode"
        if "prompt_logprobs" in fixture
        else "selected_prompt_and_forced_decode"
    )


def error_metrics(
    expected: list[float],
    actual: list[float],
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> dict:
    """Absolute logprob error equals absolute NLL error on the same token."""
    if not expected or len(expected) != len(actual):
        raise ValueError("Logprob comparison requires equal nonempty vectors")
    finite = all(math.isfinite(value) for value in expected + actual)
    errors = [abs(a - b) for a, b in zip(actual, expected)] if finite else []
    maximum = max(errors) if errors else None
    mean = sum(errors) / len(errors) if errors else None
    return {
        "count": len(expected),
        "finite": finite,
        "mean_abs": mean,
        "max_abs": maximum,
        "signed_mean_nll_difference": sum(b - a for a, b in zip(actual, expected))
        / len(expected)
        if finite
        else None,
        "mean_tolerance": mean_tolerance,
        "max_tolerance": tolerance,
        "passed": finite and mean <= mean_tolerance and maximum <= tolerance,
    }


def prompt_comparison(
    response: dict,
    fixture: dict,
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> dict:
    """With logprob_start_len=0, row j scores input token j from logit j-1."""
    stream = fixture["prompt_ids"] + fixture["continuation_ids"]
    rows = response["meta_info"]["input_token_logprobs"]
    if len(rows) != len(stream):
        raise ValueError(
            f"Prompt logprob length {len(rows)} differs from {len(stream)} tokens"
        )
    if [row[1] for row in rows] != stream:
        raise ValueError("SGLang prompt token IDs differ from the native fixture")
    if rows[0][0] is not None:
        raise ValueError(
            "First prompt token unexpectedly has a preceding-token logprob"
        )
    # Full prompt values replace the sparse prompt subset. Keep the forced
    # continuation scores, with each source position counted exactly once.
    scored = {score["position"]: score["logprob"] for score in fixture["scores"]}
    if "prompt_logprobs" in fixture:
        scored.update(enumerate(fixture["prompt_logprobs"]))
    positions = sorted(scored)
    actual, expected = [], []
    for position in positions:
        value, token_id, *_ = rows[position + 1]
        if token_id != stream[position + 1] or value is None:
            raise ValueError("SGLang next-token prompt logprob is misaligned")
        actual.append(float(value))
        expected.append(float(scored[position]))
    prompt_count = sum(position < len(fixture["prompt_ids"]) for position in positions)
    prompt_nll = error_metrics(
        expected[:prompt_count], actual[:prompt_count], tolerance, mean_tolerance
    )
    decode_nll = (
        error_metrics(
            expected[prompt_count:], actual[prompt_count:], tolerance, mean_tolerance
        )
        if prompt_count < len(positions)
        else None
    )
    return {
        **error_metrics(expected, actual, tolerance, mean_tolerance),
        # Separate gates prevent one segment from diluting the other's error.
        "passed": prompt_nll["passed"] and (decode_nll is None or decode_nll["passed"]),
        "scores_scope": fixture_score_scope(fixture),
        "prompt_nll": prompt_nll,
        "forced_decode_nll": decode_nll,
        "source_positions": positions,
        "actual_logprobs": actual,
        "native_logprobs": expected,
    }


def original_prompt_comparison(
    response: dict,
    fixture: dict,
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> dict:
    """Score L prompt logits using L-1 input targets and one output target.

    The Engine receives exactly the native fixture's original prompt length;
    its last logit scores continuation_ids[0] through the output scoring API.
    """
    prompt = fixture["prompt_ids"]
    expected = fixture["prompt_logprobs"]
    if len(expected) != len(prompt):
        raise ValueError("Original-prompt comparison requires all prompt logprobs")
    rows = response["meta_info"]["input_token_logprobs"]
    if len(rows) != len(prompt) or [row[1] for row in rows] != prompt:
        raise ValueError("Original-prompt input token scores are misaligned")
    if rows[0][0] is not None or any(row[0] is None for row in rows[1:]):
        raise ValueError("Original-prompt input logprobs have invalid null positions")
    output_scores = response["meta_info"].get("output_token_ids_logprobs") or []
    if len(output_scores) != 1 or len(output_scores[0]) != 1:
        raise ValueError("Original-prompt comparison needs one output target score")
    value, token_id, *_ = output_scores[0][0]
    if token_id != fixture["continuation_ids"][0] or value is None:
        raise ValueError("Original-prompt final target logprob is misaligned")
    actual = [float(row[0]) for row in rows[1:]] + [float(value)]
    return {
        **error_metrics(expected, actual, tolerance, mean_tolerance),
        "scores_scope": "full_prompt_original_length",
        "source_positions": list(range(len(prompt))),
        "actual_logprobs": actual,
        "native_logprobs": expected,
    }


def output_comparison(
    response: dict,
    score: dict,
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> dict:
    """Cross-check the same forced target through output_token_ids_logprobs."""
    meta = response["meta_info"]
    steps = meta["output_token_ids_logprobs"]
    if len(steps) != 1 or len(steps[0]) != 1:
        raise ValueError("Expected one requested target logprob for one output step")
    value, token_id, *_ = steps[0][0]
    if token_id != score["token_id"] or value is None:
        raise ValueError("SGLang output target token/logprob is misaligned")
    generated = response["output_ids"]
    if len(generated) != 1:
        raise ValueError("Expected exactly one greedy output token")
    metrics = error_metrics(
        [score["logprob"]], [float(value)], tolerance, mean_tolerance
    )
    metrics.update(
        native_argmax=score["argmax"],
        actual_argmax=generated[0],
        greedy_agreement=generated[0] == score["argmax"],
    )
    metrics["passed"] &= metrics["greedy_agreement"]
    return metrics


def decode_comparison(
    response: dict,
    fixture: dict,
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> dict:
    expected_ids = fixture["greedy_continuation_ids"]
    actual_ids = response["output_ids"]
    rows = response["meta_info"]["output_token_logprobs"]
    if len(actual_ids) != len(expected_ids) or [row[1] for row in rows] != actual_ids:
        raise ValueError("Generated output token/logprob counts are misaligned")
    actual_logprobs = [float(row[0]) for row in rows]
    agreement = actual_ids == expected_ids
    # After the first greedy divergence, later logprobs condition on a different
    # stream and cannot meaningfully be compared against the native trace.
    common = next(
        (
            i
            for i, pair in enumerate(zip(actual_ids, expected_ids))
            if pair[0] != pair[1]
        ),
        len(expected_ids),
    )
    metrics = (
        error_metrics(
            fixture["greedy_logprobs"][:common],
            actual_logprobs[:common],
            tolerance,
            mean_tolerance,
        )
        if common
        else {"count": 0, "passed": False}
    )
    return {
        "passed": agreement and metrics["passed"],
        "greedy_agreement": agreement,
        "matching_prefix_tokens": common,
        "native_ids": expected_ids,
        "actual_ids": actual_ids,
        "native_logprobs": fixture["greedy_logprobs"],
        "actual_logprobs": actual_logprobs,
        "matching_prefix_logprob_error": metrics,
    }


def response_list(value: Any, expected: int) -> list[dict]:
    values = value if isinstance(value, list) else [value]
    if len(values) != expected or any(not isinstance(row, dict) for row in values):
        raise ValueError("Engine response count does not match submitted requests")
    return values


def validate_fixture_case(fixture: dict, native_report: dict) -> None:
    """Every prompt must have its own qualified native oracle execution."""
    length = len(fixture["prompt_ids"])
    cases = [
        case for case in native_report["checks"] if case["prompt_tokens"] == length
    ]
    if len(cases) != 1 or cases[0]["native_oracle"].get("passed") is not True:
        raise ValueError(
            f"No unique qualified native oracle for {length}-token fixture"
        )


def reference_selected_logit_metrics(native_report: dict) -> dict:
    """Check the paired reference report without calling logprobs raw logits.

    Engine.generate does not expose raw vocabulary logits. Its normalized
    logprobs cannot recover their additive constant, so the serving run cannot
    independently measure the raw-logit relative L2 gate.
    """
    checks = []
    for case in native_report["checks"]:
        named = [
            (name, case[name])
            for name in ("prefill", "partitioned_prefill")
            if name in case
        ]
        named.extend(
            (f"decode-{index}", row) for index, row in enumerate(case.get("decode", []))
        )
        for name, row in named:
            if "relative_l2" not in row:
                continue
            error = row["relative_l2"]
            checks.append(
                {
                    "prompt_tokens": case["prompt_tokens"],
                    "name": name,
                    "relative_l2": error,
                    "passed": bool(
                        row.get("finite", True)
                        and math.isfinite(error)
                        and 0 <= error <= LOGIT_RELATIVE_L2_LIMIT
                    ),
                }
            )
    return {
        "scope": "paired native-versus-reference-model report; not Engine raw logits",
        "applicable": bool(checks),
        "limit": LOGIT_RELATIVE_L2_LIMIT,
        "passed": all(row["passed"] for row in checks) if checks else None,
        "checks": checks,
    }


def qualification_status(
    checks: list[dict],
    native_report: dict,
    logits: dict,
    source_provenance_unchanged: bool = True,
) -> dict:
    """Keep Engine evidence usable even if the paired reference run failed.

    A trustworthy native oracle permits serving comparisons; failure of the
    port's reference implementation must still fail the combined qualification.
    """
    engine_passed = bool(checks) and all(check["passed"] for check in checks)
    native_passed = native_report["passed"] is True
    return {
        "passed": engine_passed
        and native_passed
        and logits["passed"] is not False
        and source_provenance_unchanged,
        "engine_checks_passed": engine_passed,
        "native_reference_report_passed": native_passed,
        "source_provenance_unchanged": source_provenance_unchanged,
    }


def finish_resources(run: Any, wandb: Any, engine: Any, exit_code: int) -> None:
    """Finish the run and its W&B service before Engine terminates children."""
    try:
        try:
            if run is not None:
                run.finish(exit_code=exit_code)
        finally:
            if wandb is not None:
                wandb.teardown(exit_code=exit_code)
    finally:
        if engine is not None:
            engine.shutdown()


def run_checks(
    engine: Any,
    fixtures: list[dict],
    output: Path,
    tolerance: float,
    mean_tolerance: float = NLL_MEAN_LIMIT,
) -> list[dict]:
    checks = []

    def record(name, started, results, responses):
        row = {
            "name": name,
            "passed": all(result["passed"] for result in results),
            "checks": results,
            "wall_seconds": time.perf_counter() - started,
        }
        checks.append(row)
        write_json(output / f"{name}.json", {**row, "responses": responses})
        print(json.dumps({"name": name, "passed": row["passed"]}), flush=True)

    # First prove target-token alignment with untouched model scores. Supplying
    # the predetermined continuation as input avoids sampling/logit constraints.
    for index, fixture in enumerate(fixtures):
        if "prompt_logprobs" in fixture:
            started = time.perf_counter()
            response = response_list(
                engine.generate(
                    input_ids=fixture["prompt_ids"],
                    sampling_params={
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                    logprob_start_len=0,
                    token_ids_logprob=[fixture["continuation_ids"][0]],
                ),
                1,
            )[0]
            record(
                f"original-prompt-{index}",
                started,
                [
                    original_prompt_comparison(
                        response, fixture, tolerance, mean_tolerance
                    )
                ],
                [response],
            )
        stream = fixture["prompt_ids"] + fixture["continuation_ids"]
        started = time.perf_counter()
        response = response_list(
            engine.generate(
                input_ids=stream,
                sampling_params={
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=0,
            ),
            1,
        )[0]
        record(
            f"forced-prefill-{index}",
            started,
            [prompt_comparison(response, fixture, tolerance, mean_tolerance)],
            [response],
        )
        for score_index, score in enumerate(fixture["scores"]):
            if score["position"] < len(fixture["prompt_ids"]) - 1:
                continue
            started = time.perf_counter()
            response = response_list(
                engine.generate(
                    input_ids=stream[: score["position"] + 1],
                    sampling_params={
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                    logprob_start_len=-1,
                    token_ids_logprob=[score["token_id"]],
                ),
                1,
            )[0]
            record(
                f"output-alignment-{index}-{score_index}",
                started,
                [output_comparison(response, score, tolerance, mean_tolerance)],
                [response],
            )

    # The first output token uses prefill; all remaining tokens exercise decode
    # with retained K/V and each family's convolution/recurrent request state.
    def generate_traces(indices, name, trim_last=False):
        chosen = [fixtures[i] for i in indices]
        if trim_last:
            chosen = [
                {
                    **fixture,
                    "greedy_continuation_ids": fixture["greedy_continuation_ids"][:1],
                    "greedy_logprobs": fixture["greedy_logprobs"][:1],
                }
                if j == len(chosen) - 1
                else fixture
                for j, fixture in enumerate(chosen)
            ]
        started = time.perf_counter()
        responses = response_list(
            engine.generate(
                input_ids=[fixture["prompt_ids"] for fixture in chosen],
                sampling_params=[
                    {
                        "max_new_tokens": len(fixture["greedy_continuation_ids"]),
                        "temperature": 0.0,
                        "ignore_eos": True,
                    }
                    for fixture in chosen
                ],
                return_logprob=True,
                logprob_start_len=-1,
            ),
            len(chosen),
        )
        results = [
            decode_comparison(response, fixture, tolerance, mean_tolerance)
            for response, fixture in zip(responses, chosen)
        ]
        record(name, started, results, responses)

    for index in range(len(fixtures)):
        generate_traces([index], f"decode-single-{index}")
    indices = list(range(len(fixtures)))
    generate_traces(indices, "decode-batch")
    generate_traces(indices[::-1], "decode-reordered")
    # Distinct completion lengths make one request finish while another remains
    # alive. The subsequent repeated batch reuses the freed scheduler resources.
    if len(fixtures) > 1:
        generate_traces(indices, "decode-unequal-completions", trim_last=True)
    generate_traces(indices, "decode-repeat-after-completion")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("fa3", "fa4", "flashinfer", "triton"),
        default="fa3",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=64)
    parser.add_argument(
        "--enable-deterministic-inference",
        action="store_true",
        help="Use SGLang's deterministic execution policy; qualify it separately",
    )
    parser.add_argument("--max-logprob-error", type=float, default=0.08)
    parser.add_argument("--mean-logprob-error", type=float, default=0.005)
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument("--wandb-project", default="sglang-hybrid-serving-port")
    parser.add_argument(
        "--wandb-mode", choices=("offline", "online", "disabled"), default="offline"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    engine, run, wandb, phase = None, None, None, "source_provenance"
    try:
        provenance_start = capture_source_provenance()
        write_json(args.output / "source_provenance_start.json", provenance_start)
        phase = "validate_inputs"
        validate_thresholds(args.max_logprob_error, args.mean_logprob_error)
        config = json.loads((args.export / "config.json").read_text())
        architecture = validate_architecture(config)
        manifest_hash = sha256_file(args.export / "manifest.json")
        complete = json.loads((args.export / "EXPORT_COMPLETE.json").read_text())
        if complete["manifest_sha256"] != manifest_hash:
            raise ValueError("Export completion hash is invalid")
        manifest = json.loads((args.export / "manifest.json").read_text())
        for filename, digest in manifest["files"].items():
            if sha256_file(args.export / filename) != digest:
                raise ValueError(f"Export artifact hash mismatch: {filename}")
        native_report = json.loads((args.fixtures / "report.json").read_text())
        if native_report["export_manifest_sha256"] != manifest_hash:
            raise ValueError("Native fixtures refer to a different model export")
        if not all(case["native_oracle"]["passed"] for case in native_report["checks"]):
            raise ValueError("The native fixture oracle was not qualified")
        selected_logit_metrics = reference_selected_logit_metrics(native_report)
        token_ids_path = args.fixtures / "token_ids.json"
        if sha256_file(token_ids_path) != native_report["token_ids_sha256"]:
            raise ValueError("Native fixture token-ID hash changed")
        all_ids = json.loads(token_ids_path.read_text())
        fixture_paths = sorted(args.fixtures.glob("serving_fixture-*.json"))
        if len(fixture_paths) < 2:
            raise ValueError(
                "At least two native fixtures are needed for batching/reorder checks"
            )
        fixtures = [json.loads(path.read_text()) for path in fixture_paths]
        for fixture in fixtures:
            validate_fixture(fixture, config["vocab_size"])
            validate_fixture_case(fixture, native_report)
            stream = fixture["prompt_ids"] + fixture["continuation_ids"]
            if stream != all_ids[: len(stream)]:
                raise ValueError(
                    "Fixture tokens do not match the saved native token stream"
                )
        if len({token_identity(fixture["prompt_ids"]) for fixture in fixtures}) != len(
            fixtures
        ):
            raise ValueError(
                "Batching fixtures must have distinct prompt token streams"
            )
        engine_args = {
            "model_path": str(args.export.resolve()),
            "tp_size": 1,
            "pp_size": 1,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "skip_tokenizer_init": True,
            "disable_radix_cache": True,
            "cuda_graph_config": {
                "decode": {"backend": "disabled"},
                "prefill": {"backend": "disabled"},
            },
            "speculative_algorithm": None,
            "attention_backend": args.attention_backend,
            "chunked_prefill_size": args.chunked_prefill_size,
            "enable_deterministic_inference": args.enable_deterministic_inference,
            "max_running_requests": max(2, len(fixtures)),
            "mem_fraction_static": args.mem_fraction_static,
            "random_seed": 20260919,
            "log_level": "warning",
        }
        write_json(args.output / "engine_args.json", engine_args)
        os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "true"
        # Set before importing SGLang, whose sampler reads this at import time.
        import torch
        import wandb
        from sglang import Engine
        from sglang.version import __version__ as sglang_version

        if not torch.cuda.is_available():
            raise RuntimeError("Serving qualification requires an allocated CUDA GPU")
        run = wandb.init(
            entity="mbzuai-llm",
            project=args.wandb_project,
            mode=args.wandb_mode,
            dir=str(args.output),
            job_type="serving-differential",
            config=engine_args,
        )
        phase = "engine_startup"
        started = time.perf_counter()
        engine = Engine(**engine_args)
        startup_seconds = time.perf_counter() - started
        phase = "engine_checks"
        checks = run_checks(
            engine,
            fixtures,
            args.output,
            args.max_logprob_error,
            args.mean_logprob_error,
        )
        provenance_end = capture_source_provenance()
        fixture_scopes = {fixture_score_scope(fixture) for fixture in fixtures}
        full_prompt_nll_evaluated = all(
            "prompt_logprobs" in fixture for fixture in fixtures
        )
        report = {
            **qualification_status(
                checks,
                native_report,
                selected_logit_metrics,
                source_provenance_unchanged=provenance_start == provenance_end,
            ),
            "qualification_scope": "actual SGLang Engine NLL/greedy checks plus paired reference selected-logit gate; batching, reorder, repeated slot allocation",
            "scores_scope": next(iter(fixture_scopes))
            if len(fixture_scopes) == 1
            else "mixed_full_and_selected_prompt",
            "full_prompt_nll_evaluated": full_prompt_nll_evaluated,
            "nll_scope": "per-fixture prompt coverage is identified by scores_scope; prompt and forced-decode NLL gates are separate; greedy traces compare matching conditionals",
            "architecture": architecture,
            "engine_selected_logits_relative_l2": {
                "evaluated": False,
                "reason": "Engine API returns normalized logprobs; fixtures contain target scores, not raw vocabulary logits",
            },
            "native_reference_selected_logits": selected_logit_metrics,
            "not_qualified": (
                [] if full_prompt_nll_evaluated else ["full-token Engine prompt NLL"]
            )
            + [
                "cancellation",
                "explicit eviction/retraction",
                "prefix reuse",
                "CUDA graphs",
                "speculative decoding",
                "TP greater than 1",
                "raw Engine selected-logits relative L2",
            ],
            "checks": checks,
            "engine_startup_seconds": startup_seconds,
            "timing_scope": "diagnostic synchronized API wall times; includes scoring and scheduling; not throughput benchmarks",
            "export_manifest_sha256": manifest_hash,
            "native_report_sha256": sha256_file(args.fixtures / "report.json"),
            "fixtures": [
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "prompt_ids_sha256": token_identity(fixture["prompt_ids"]),
                    "scores_scope": fixture_score_scope(fixture),
                }
                for path, fixture in zip(fixture_paths, fixtures)
            ],
            **provenance_start,
            "source_provenance_end": provenance_end,
            "torch": torch.__version__,
            "sglang": sglang_version,
            "device": torch.cuda.get_device_name(),
            "max_logprob_error": args.max_logprob_error,
            "mean_logprob_error": args.mean_logprob_error,
        }
        write_json(args.output / "report.json", report)
        run.summary["passed"] = report["passed"]
        if not report["passed"]:
            raise SystemExit(1)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {"phase": phase, "type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        finish_resources(
            run,
            wandb,
            engine,
            exit_code=0
            if (args.output / "report.json").exists()
            and json.loads((args.output / "report.json").read_text())["passed"]
            else 1,
        )


if __name__ == "__main__":
    main()
