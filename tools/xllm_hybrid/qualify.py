"""GPU differential checks against unchanged native xLLM operators and DCP.

Run one process on an independently allocated GPU. Writes a new directory only;
neither source checkpoints nor native repositories/environments are modified.
This qualifies the reference model; it does not measure SGLang scheduler speed.
"""

import argparse
import importlib
import itertools
import json
import sys
import time
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .export import (
        file_snapshot,
        git_identity,
        inventory_checkpoint,
        sha256_file,
        tensor_digest,
        write_json,
    )
    from .schema import canonical_name
except ImportError:
    from export import (
        file_snapshot,
        git_identity,
        inventory_checkpoint,
        sha256_file,
        tensor_digest,
        write_json,
    )
    from schema import canonical_name


def import_reference():
    """Keep the pure PyTorch oracle usable without SGLang's GPU dependencies."""
    directory = Path(__file__).resolve().parents[2] / "python/sglang/srt/models"
    name = "_xllm_qualification_models"
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.xllm_reference")


def capture_source_provenance(native_repo):
    """Capture source bytes before model imports, then verify them after checks."""
    directory = Path(__file__).resolve().parent
    models = directory.parents[1] / "python/sglang/srt/models"
    kernels = (
        directory.parents[1] / "python/sglang/kernels/ops/attention/flash_attn/cute"
    )
    native_paths = list((native_repo / "xllm").rglob("*.py"))
    native_paths.extend(
        path
        for path in (
            native_repo / "tools/prepare_gekko_conversion.py",
            native_repo / "tools/reference_hybrid_generation.py",
        )
        if path.is_file()
    )
    return {
        "qualifier_sha256": sha256_file(__file__),
        "reference_sources": {
            path.name: sha256_file(path) for path in sorted(models.glob("xllm*.py"))
        },
        "attention_sources": {
            name: sha256_file(kernels / name)
            for name in ("interface.py", "flash_fwd_sm90.py", "softmax.py")
        },
        "loader_sources": {
            name: sha256_file(directory / name) for name in ("export.py", "schema.py")
        },
        "port_repo": git_identity(directory),
        "native_repo": git_identity(native_repo),
        "native_sources": {
            str(path.relative_to(native_repo)): sha256_file(path)
            for path in sorted(native_paths)
        },
    }


def compare(reference, actual):
    reference, actual = reference.float(), actual.float()
    difference = actual - reference
    result = {
        "mean_abs": difference.abs().mean().item(),
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-12)).item(),
        "greedy_agreement": (actual.argmax(-1) == reference.argmax(-1))
        .float()
        .mean()
        .item(),
        "finite": bool(
            torch.isfinite(actual).all() and torch.isfinite(reference).all()
        ),
    }
    result["passed"] = bool(result["finite"] and result["relative_l2"] <= 0.01)
    return result


def compare_nll(reference, actual):
    difference = actual.float() - reference.float()
    absolute = difference.abs()
    mean, maximum = absolute.mean().item(), absolute.max().item()
    return {
        "signed_mean": difference.mean().item(),
        "mean_abs": mean,
        "max_abs": maximum,
        "absolute_percentiles": dict(
            zip(
                ("p50", "p95", "p99"),
                torch.quantile(
                    absolute.flatten(),
                    torch.tensor([0.5, 0.95, 0.99], device=absolute.device),
                ).tolist(),
            )
        ),
        "scored_tokens": difference.numel(),
        "passed": bool(
            torch.isfinite(difference).all() and mean <= 0.005 and maximum <= 0.08
        ),
    }


def load_export(directory, reference_module, device="cuda"):
    from safetensors import safe_open

    manifest = json.loads((directory / "manifest.json").read_text())
    complete = json.loads((directory / "EXPORT_COMPLETE.json").read_text())
    if complete["manifest_sha256"] != sha256_file(directory / "manifest.json"):
        raise ValueError("Invalid completed export manifest hash")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Export artifact hash mismatch: {name}")
    config = json.loads((directory / "config.json").read_text())
    with torch.device("meta"):
        model = reference_module.XllmReferenceModel(config)
    model.to_empty(device=device)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    destination = model.state_dict()
    expected = {spec["destination"] for spec in manifest["tensors"].values()}
    if set(destination) != expected:
        raise ValueError(
            f"Export/model keys differ: missing={sorted(set(destination) - expected)}, extra={sorted(expected - set(destination))}"
        )
    index = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    consumed = set()
    with torch.no_grad():
        for filename in sorted(set(index.values())):
            with safe_open(directory / filename, framework="pt", device="cpu") as shard:
                # safe_open provides keys() but is not a Python Mapping.
                for name in shard.keys():  # noqa: SIM118
                    if name in consumed or index.get(name) != filename:
                        raise ValueError(
                            f"Repeated or incorrectly indexed tensor: {name}"
                        )
                    tensor = shard.get_tensor(name)
                    if tensor.shape != destination[name].shape:
                        raise ValueError(f"Model tensor shape mismatch: {name}")
                    destination[name].copy_(tensor)
                    consumed.add(name)
    if consumed != expected:
        raise ValueError("Partially consumed export")
    return model.eval().requires_grad_(False), manifest


def load_native(args, manifest):
    """Independent native DCP restoration, bounded CPU memory, strict GPU copy.

    This uses the original module classes outside training FSDP wrappers at
    TP1. BF16 parameters match FSDP's forward mixed precision; persistent TDN
    buffers keep their native types and RoPE stays complex64.
    """
    import torch.distributed.checkpoint as dcp

    sys.path.insert(0, str(args.native_repo.resolve()))
    from xllm.config import ModelConf, TokenizerConf
    from xllm.data.dataset_streamer.tokenizer import build_tokenizer
    from xllm.models.build import get_model_cls

    report, saved, reader, metadata, files, snapshot = inventory_checkpoint(
        args.checkpoint
    )
    if report["source_hashes"] != {
        name: manifest["source_hashes"][name] for name in report["source_hashes"]
    }:
        raise ValueError("Native source configuration/metadata differs from the export")
    model_config = deepcopy(saved["model"])
    if model_config["arch"] == "transformer":
        from tools.prepare_gekko_conversion import migrate_source_model

        config = migrate_source_model(model_config)
    else:
        config = ModelConf.from_dict(model_config)
    config.init_mode = "none"
    config.ddp_backend = "fsdp2"
    config.fused_block = False
    tokenizer = build_tokenizer(TokenizerConf.from_dict(saved["tokenizer"]))
    with torch.device("meta"):
        model = get_model_cls(config.arch)(config, tokenizer)
    model.to_empty(device="cuda")
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    destination = model.state_dict()
    if set(destination) != {canonical_name(name) for name in report["tensors"]}:
        raise ValueError(
            "Native module state differs from complete checkpoint inventory"
        )
    # Use the export's deterministic grouping only to bound memory; values are
    # independently read from native DCP and checked against tensor-byte hashes.
    index = json.loads((args.export / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    loaded = set()
    with torch.no_grad():
        for filename in sorted(set(index.values())):
            names = [
                name
                for name in report["tensors"]
                if index[canonical_name(name)] == filename
            ]
            state = {
                name: torch.empty(
                    tuple(metadata.state_dict_metadata[name].size),
                    dtype=metadata.state_dict_metadata[name].properties.dtype,
                )
                for name in names
            }
            dcp.load(state, storage_reader=reader, no_dist=True)
            for name, tensor in state.items():
                if tensor_digest(tensor) != manifest["tensors"][name]["sha256"]:
                    raise ValueError(f"Native DCP/export tensor bytes differ: {name}")
                target = destination[canonical_name(name)]
                if target.shape != tensor.shape:
                    raise ValueError(f"Native model shape differs: {name}")
                target.copy_(tensor)
                if not torch.equal(
                    target, tensor.to(device=target.device, dtype=target.dtype)
                ):
                    raise ValueError(f"Native model tensor copy failed: {name}")
                loaded.add(name)
            del state
    if loaded != report["tensors"].keys():
        raise ValueError("Native DCP load left unused tensors")
    if file_snapshot(files) != snapshot:
        raise ValueError("Native checkpoint changed during qualification load")
    proof = {
        "repo": git_identity(args.native_repo),
        "checkpoint": str(args.checkpoint),
        "tensor_count": len(loaded),
        "all_source_tensor_hashes_match_export": True,
        "execution": "native TP1 modules without FSDP; BF16 parameters, native buffer dtypes",
        "config_overrides": {
            "init_mode": "none",
            "ddp_backend": "fsdp2",
            "fused_block": False,
        },
    }
    return model.eval().requires_grad_(False), tokenizer, proof


def native_logits(model, tokens, reference_module, positions=None):
    if model.arch == "hybrid_gekko":
        from tools.reference_hybrid_generation import _padded_prefix

        padded = _padded_prefix(
            tokens, model.chunk_size, model.full_cfg.max_position_embeddings
        )
        logits, aux, cache = model(
            padded, False, cache=None, logits_positions=positions
        )
        if aux is not None or cache is not None:
            raise ValueError(
                "Native full-prefix oracle unexpectedly returned recurrent state"
            )
        return logits if positions is not None else logits[:, : tokens.shape[1]]
    return reference_module.native_cache_free_logits(model, tokens, positions)


def qualify_oracle(model, tokens, reference_module):
    """Prove selected projection and (for Inkling) direct block-loop loss."""
    full = native_logits(model, tokens, reference_module)
    selected = native_logits(model, tokens, reference_module, [tokens.shape[1] - 1])
    selection = compare(full[:, -1:], selected)
    selection["passed"] = bool(
        selection["finite"]
        and selection["relative_l2"] <= 1e-3
        and selection["greedy_agreement"] == 1.0
    )
    result = {"selected_vs_full": selection}
    if model.arch != "hybrid_gekko":
        targets = tokens.roll(-1, dims=1)
        expected_loss = F.cross_entropy(
            full.flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        native_loss = model(tokens, False, targets=targets, cache=None)[0]
        difference = (expected_loss - native_loss.float()).abs()
        result["native_forward_loss"] = {
            "mean_abs": difference.mean().item(),
            "max_abs": difference.max().item(),
            "passed": bool(
                torch.allclose(expected_loss, native_loss.float(), atol=1e-5, rtol=1e-5)
            ),
        }
    result["passed"] = all(item["passed"] for item in result.values())
    return result, full


def run_cases(native, actual, tokenizer, args, reference_module):
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or any(length < 1 for length in lengths) or args.decode_steps < 0:
        raise ValueError(
            "Positive prompt lengths and nonnegative decode steps required"
        )
    prompt = (
        "Explain how a long sequence model retains earlier facts. The key is 48217. "
    )
    base = tokenizer.encode(prompt, bos=True, eos=False)
    count = max(lengths) + args.decode_steps + 1
    ids = (base * ((count + len(base) - 1) // len(base)))[:count]
    write_json(args.output / "token_ids.json", ids)
    rows = []
    for length in lengths:
        tokens = torch.tensor([ids[:length]], device="cuda", dtype=torch.long)
        positions = sorted({0, min(7, length - 1), length - 1})
        oracle, native_full = qualify_oracle(native, tokens, reference_module)
        reference = native_full[:, positions]
        start = time.perf_counter()
        prediction_full = actual(tokens).float()
        prediction = prediction_full[:, positions]
        torch.cuda.synchronize()
        row = {
            "prompt_tokens": length,
            "native_oracle": oracle,
            "prefill": compare(reference, prediction),
            "reference_prefill_seconds": time.perf_counter() - start,
        }
        targets = torch.tensor(
            [ids[position + 1] for position in positions], device="cuda"
        )
        ref_nll = F.cross_entropy(reference.flatten(0, 1), targets, reduction="none")
        actual_nll = F.cross_entropy(
            prediction.flatten(0, 1), targets, reduction="none"
        )
        row["selected_prompt_nll"] = {
            "native": ref_nll.tolist(),
            "port": actual_nll.tolist(),
            "mean_abs": (ref_nll - actual_nll).abs().mean().item(),
        }
        all_targets = torch.tensor(ids[1 : length + 1], device="cuda")
        native_nll = F.cross_entropy(
            native_full.flatten(0, 1), all_targets, reduction="none"
        )
        port_nll = F.cross_entropy(
            prediction_full.flatten(0, 1), all_targets, reduction="none"
        )
        row["prompt_nll"] = compare_nll(native_nll, port_nll)
        del native_full, prediction_full
        fixture = {
            "prompt_ids": ids[:length],
            "continuation_ids": ids[length : length + args.decode_steps + 1],
            # Entry p scores stream[p + 1] from source logit p. The final
            # entry predicts continuation_ids[0], so all L prompt logits are
            # covered and there is no entry for an unconditioned first token.
            "prompt_logprobs": (-native_nll).tolist(),
            "scores": [
                {
                    "position": position,
                    "token_id": ids[position + 1],
                    "logprob": -ref_nll[index].item(),
                    "argmax": reference[0, index].argmax().item(),
                }
                for index, position in enumerate(positions)
            ],
        }
        # Deliberately split off one token and at an odd interior position.
        cache = actual.new_cache()
        cuts = sorted({0, min(1, length), min(17, length), length})
        for begin, end in itertools.pairwise(cuts):
            cached = actual(
                tokens[:, begin:end], cache, selected_positions=[end - begin - 1]
            ).float()
        row["partitioned_prefill"] = compare(reference[:, -1:], cached)
        row["partitioned_prefill_nll"] = compare_nll(
            ref_nll[-1:],
            F.cross_entropy(cached.flatten(0, 1), targets[-1:], reduction="none"),
        )
        row["decode"] = []
        native_decode_nll, port_decode_nll = [], []
        for step in range(args.decode_steps):
            token = torch.tensor([[ids[length + step]]], device="cuda")
            tokens = torch.cat((tokens, token), dim=1)
            ref_step = native_logits(
                native, tokens, reference_module, [tokens.shape[1] - 1]
            )
            cached = actual(token, cache).float()
            result = compare(ref_step, cached)
            result["forced_input_id"] = int(token.item())
            row["decode"].append(result)
            target = ids[length + step + 1]
            native_decode_nll.append(
                -F.log_softmax(ref_step[0, -1].float(), -1)[target]
            )
            port_decode_nll.append(-F.log_softmax(cached[0, -1].float(), -1)[target])
            fixture["scores"].append(
                {
                    "position": length + step,
                    "token_id": target,
                    "logprob": F.log_softmax(ref_step[0, -1].float(), -1)[
                        target
                    ].item(),
                    "argmax": ref_step[0, -1].argmax().item(),
                }
            )
        if native_decode_nll:
            row["decode_nll"] = compare_nll(
                torch.stack(native_decode_nll), torch.stack(port_decode_nll)
            )
        # Native greedy continuations qualify actual scheduler decode as well
        # as forced-token prompt scoring. Keep generating after EOS so the
        # serving harness can explicitly request the same fixed budget.
        greedy_tokens = torch.tensor([ids[:length]], device="cuda")
        greedy_logits = reference[:, -1:]
        fixture["greedy_continuation_ids"] = []
        fixture["greedy_logprobs"] = []
        for step in range(args.decode_steps + 1):
            next_id = int(greedy_logits[0, -1].argmax())
            fixture["greedy_continuation_ids"].append(next_id)
            fixture["greedy_logprobs"].append(
                F.log_softmax(greedy_logits[0, -1].float(), -1)[next_id].item()
            )
            if step < args.decode_steps:
                greedy_tokens = torch.cat(
                    (greedy_tokens, torch.tensor([[next_id]], device="cuda")), dim=1
                )
                greedy_logits = native_logits(
                    native,
                    greedy_tokens,
                    reference_module,
                    [greedy_tokens.shape[1] - 1],
                )
        row["passed"] = bool(
            oracle["passed"]
            and row["prefill"]["passed"]
            and row["partitioned_prefill"]["passed"]
            and row["prompt_nll"]["passed"]
            and row["partitioned_prefill_nll"]["passed"]
            and row.get("decode_nll", {"passed": True})["passed"]
            and all(step["passed"] for step in row["decode"])
        )
        rows.append(row)
        write_json(args.output / f"serving_fixture-{length}.json", fixture)
        write_json(args.output / f"case-{length}.json", row)
        print(
            json.dumps(
                {
                    "prompt_tokens": length,
                    "passed": row["passed"],
                    "prefill": row["prefill"],
                }
            ),
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-repo", "checkpoint", "export", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lengths", default="31,127")
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--wandb-project", default="sglang-hybrid-serving-port")
    parser.add_argument(
        "--wandb-mode", choices=("offline", "online", "disabled"), default="offline"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    phase, run = "initialization", None
    try:
        provenance_start = capture_source_provenance(args.native_repo)
        write_json(args.output / "source_provenance_start.json", provenance_start)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qualification requires an independently allocated CUDA GPU"
            )
        torch.manual_seed(20260919)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference_module = import_reference()
        import wandb

        run = wandb.init(
            entity="mbzuai-llm",
            project=args.wandb_project,
            mode=args.wandb_mode,
            dir=str(args.output),
            job_type="native-differential",
            config={
                "checkpoint": str(args.checkpoint),
                "export": str(args.export),
                "lengths": args.lengths,
                "decode_steps": args.decode_steps,
            },
        )
        phase = "load_export"
        actual, manifest = load_export(args.export, reference_module)
        phase = "load_native_dcp"
        native, tokenizer, proof = load_native(args, manifest)
        write_json(args.output / "native-proof.json", proof)
        phase = "differential_checks"
        with torch.no_grad():
            rows = run_cases(native, actual, tokenizer, args, reference_module)
        provenance_end = capture_source_provenance(args.native_repo)
        source_provenance_unchanged = provenance_start == provenance_end
        report = {
            "passed": all(row["passed"] for row in rows)
            and source_provenance_unchanged,
            "source_provenance_unchanged": source_provenance_unchanged,
            "source_provenance_end": provenance_end,
            "checks": rows,
            "gates": {
                "nll_mean_abs": 0.005,
                "nll_max_abs": 0.08,
                "logits_relative_l2": 0.01,
            },
            "export_manifest_sha256": sha256_file(args.export / "manifest.json"),
            "token_ids_sha256": sha256_file(args.output / "token_ids.json"),
            **provenance_start,
            "native": proof,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "qualification_scope": "native versus port PyTorch reference; no serving throughput claim",
        }
        write_json(args.output / "report.json", report)
        run.summary["passed"] = report["passed"]
        run.finish(exit_code=0 if report["passed"] else 1)
        if not report["passed"]:
            raise SystemExit(1)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {"phase": phase, "type": type(error).__name__, "error": str(error)},
        )
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
