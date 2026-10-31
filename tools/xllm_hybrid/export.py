"""Strict DCP inventory and immutable safetensors export for xLLM hybrids.

Run ``python tools/xllm_hybrid/export.py --help``. Inventory reads metadata only;
the explicit ``export`` subcommand restores all tensors using PyTorch DCP.
Inputs must be trusted native checkpoints: DCP metadata is Python serialized.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .schema import canonical_name, resolve_config, validate_inventory
except ImportError:
    from schema import canonical_name, resolve_config, validate_inventory


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def write_json(path, value):
    with open(path, "xb") as stream:
        stream.write(json_bytes(value))


def tensor_digest(tensor):
    """Hash original storage bytes, including BF16 and scalar buffers."""
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def git_identity(path):
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "path": str(Path(path).resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _storage_files(reader, metadata):
    # Preserve the published logical checkpoint path: model-only checkpoints
    # can intentionally symlink the complete sharded_model.tp00 directory.
    root = Path(reader.path).absolute()
    files = {root / ".metadata"}
    for info in metadata.storage_data.values():
        path = root / info.relative_path
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid DCP payload {info.relative_path!r}")
        if (
            info.offset < 0
            or info.length <= 0
            or path.stat().st_size < info.offset + info.length
        ):
            raise ValueError(f"Truncated DCP payload {info.relative_path!r}")
        files.add(path)
    return sorted(files)


def file_snapshot(paths):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def inventory_checkpoint(checkpoint):
    """Read/validate every model metadata entry, never allocate full weights."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint / "config.json"
    saved = json.loads(config_path.read_text())
    config = resolve_config(saved)
    model_dirs = sorted(checkpoint.glob("sharded_model.tp*"))
    if [p.name for p in model_dirs] != ["sharded_model.tp00"]:
        raise ValueError(
            "Exporter currently requires exactly one TP1 sharded_model.tp00 directory"
        )
    reader = FileSystemReader(model_dirs[0])
    metadata = reader.read_metadata()
    if not metadata.storage_data:
        raise ValueError("Missing DCP storage index")
    files = _storage_files(reader, metadata) + [config_path]
    receipt = checkpoint / "graft_receipt.json"
    if receipt.exists():
        files.append(receipt)
    snapshot = file_snapshot(files)
    inventory = {}
    storage_indices = {
        (index.fqn, tuple(index.offset or ())): info
        for index, info in metadata.storage_data.items()
    }
    for name, entry in sorted(metadata.state_dict_metadata.items()):
        if not isinstance(entry, TensorStorageMetadata):
            raise TypeError(f"Unexpected non-tensor model metadata: {name}")
        shape = list(entry.size)
        total = math.prod(shape)
        covered = 0
        boxes = []
        for chunk in entry.chunks:
            offsets, sizes = list(chunk.offsets), list(chunk.sizes)
            if (
                len(offsets) != len(shape)
                or len(sizes) != len(shape)
                or any(
                    offset < 0 or size <= 0 or offset + size > limit
                    for offset, size, limit in zip(offsets, sizes, shape)
                )
            ):
                raise ValueError(f"Invalid DCP chunk for {name}")
            if (name, tuple(offsets)) not in storage_indices:
                raise ValueError(f"Missing DCP storage entry for {name} at {offsets}")
            for old_offsets, old_sizes in boxes:
                if all(
                    a < b + d and b < a + c
                    for a, b, c, d in zip(offsets, old_offsets, sizes, old_sizes)
                ):
                    raise ValueError(f"Overlapping DCP chunks for {name}")
            boxes.append((offsets, sizes))
            covered += math.prod(sizes)
        if covered != total:
            raise ValueError(
                f"Incomplete DCP chunks for {name}: {covered} of {total} elements"
            )
        inventory[name] = {
            "shape": shape,
            "dtype": str(entry.properties.dtype).removeprefix("torch."),
            "numel": total,
            "nbytes": total * entry.properties.dtype.itemsize,
            "destination": canonical_name(name),
            "transform": "identity"
            if canonical_name(name) == name
            else "rename_legacy_attention_norm",
        }
    expected = validate_inventory(config, inventory)
    for name, spec in inventory.items():
        spec["role"] = expected[canonical_name(name)]["role"]
    hashes = {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in files
        if path.name in {"config.json", ".metadata", "graft_receipt.json"}
    }
    if file_snapshot(files) != snapshot:
        raise ValueError("Checkpoint changed while reading metadata")
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "source_hashes": hashes,
        "architecture": config["architectures"][0],
        "tensor_count": len(inventory),
        "parameter_elements": sum(
            x["numel"] for x in inventory.values() if x["role"] == "parameter"
        ),
        "buffer_elements": sum(
            x["numel"] for x in inventory.values() if x["role"] == "buffer"
        ),
        "weight_bytes": sum(x["nbytes"] for x in inventory.values()),
        "tensors": inventory,
        "config": config,
    }
    return report, saved, reader, metadata, files, snapshot


def _tokenizer_files(saved, tokenizer):
    path = Path(tokenizer or saved.get("tokenizer", {}).get("path", "")).resolve()
    if not (path / "tokenizer.json").is_file():
        raise ValueError(
            "An explicit tokenizer directory containing tokenizer.json is required"
        )
    files = sorted(p for p in path.rglob("*") if p.is_file())
    # Root directories also occupy output names, even when only their children
    # appear in the file inventory. Reserve the shard namespace before copying.
    names = {p.relative_to(path).parts[0] for p in files}
    reserved = {
        "config.json",
        "native_config.json",
        "manifest.json",
        "model.safetensors.index.json",
        "EXPORT_COMPLETE.json",
        "graft_receipt.json",
    }
    if names & reserved or any(
        name.startswith("model-") and name.endswith(".safetensors") for name in names
    ):
        raise ValueError("Tokenizer contains names reserved by the export")
    return path, files


def export_checkpoint(
    checkpoint, output, source_repo, tokenizer=None, shard_bytes=1 << 30
):
    """Export exact native tensor values; no normalization or dtype conversion."""
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(min(torch.get_num_threads(), 4))

    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report, saved, reader, metadata, files, snapshot = inventory_checkpoint(checkpoint)
    tokenizer_root, tokenizer_files = _tokenizer_files(saved, tokenizer)
    token_snapshot = file_snapshot(tokenizer_files)
    provenance = {
        "source_repo": git_identity(source_repo),
        "exporter_repo": git_identity(Path(__file__).parent),
        "exporter_sources": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("schema.py", "export.py")
        },
        "torch_version": torch.__version__,
    }
    # Payload hashes make the selected source immutable and reproducible.
    source_root = Path(checkpoint).resolve()
    source_hashes = {str(p.relative_to(source_root)): sha256_file(p) for p in files}
    if file_snapshot(files) != snapshot:
        raise ValueError("Checkpoint changed while hashing payloads")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent)
    )
    try:
        write_json(staging / "config.json", report["config"])
        # Keep exact original bytes; their hash is already included in source_hashes.
        shutil.copyfile(source_root / "config.json", staging / "native_config.json")
        if (source_root / "graft_receipt.json").exists():
            shutil.copyfile(
                source_root / "graft_receipt.json", staging / "graft_receipt.json"
            )
        tokenizer_hashes = {}
        tokenizer_transforms = {}
        for path in tokenizer_files:
            relative = path.relative_to(tokenizer_root)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            tokenizer_hashes[str(relative)] = sha256_file(path)
            if relative.as_posix() == "generation_config.json":
                generation = json.loads(path.read_text())
                generation.update(bos_token_id=0, eos_token_id=1, pad_token_id=None)
                write_json(destination, generation)
                tokenizer_transforms[str(relative)] = (
                    "native BOS=0/EOS=1; no padding token"
                )
            else:
                shutil.copyfile(path, destination)
        if file_snapshot(tokenizer_files) != token_snapshot:
            raise ValueError("Tokenizer changed during export")
        # Use sorted keys and deterministic shard boundaries; values remain native.
        batches, batch, size = [], [], 0
        for name in sorted(report["tensors"]):
            nbytes = report["tensors"][name]["nbytes"]
            if batch and size + nbytes > shard_bytes:
                batches.append(batch)
                batch, size = [], 0
            batch.append(name)
            size += nbytes
        if batch:
            batches.append(batch)
        weight_map = {}
        for index, names in enumerate(batches, 1):
            filename = f"model-{index:05d}-of-{len(batches):05d}.safetensors"
            # DCP's planner restores a named subset correctly across all rank
            # payloads. Bound peak CPU memory by a shard, while visiting every
            # model tensor exactly once; optimizer state is never requested.
            state = {
                name: torch.empty(
                    tuple(metadata.state_dict_metadata[name].size),
                    dtype=metadata.state_dict_metadata[name].properties.dtype,
                    device="cpu",
                )
                for name in names
            }
            dcp.load(state, storage_reader=reader, no_dist=True)
            if file_snapshot(files) != snapshot:
                raise ValueError("Checkpoint changed during DCP restoration")
            tensors = {canonical_name(name): state[name].contiguous() for name in names}
            save_file(tensors, str(staging / filename))
            with safe_open(
                staging / filename, framework="pt", device="cpu"
            ) as restored:
                if set(restored.keys()) != set(tensors):
                    raise ValueError("Safetensors write/read inventory mismatch")
                for source in names:
                    name = canonical_name(source)
                    digest = tensor_digest(state[source])
                    value = restored.get_tensor(name)
                    if (
                        value.dtype != state[source].dtype
                        or tensor_digest(value) != digest
                    ):
                        raise ValueError(
                            f"Tensor failed exact export round-trip: {source}"
                        )
                    report["tensors"][source]["sha256"] = digest
                    weight_map[name] = filename
            del state, tensors, value
        write_json(
            staging / "model.safetensors.index.json",
            {
                "metadata": {"total_size": report["weight_bytes"]},
                "weight_map": weight_map,
            },
        )
        if set(weight_map) != {canonical_name(name) for name in report["tensors"]}:
            raise ValueError("Export did not consume every native tensor")
        if file_snapshot(files) != snapshot:
            raise ValueError("Checkpoint changed before export publication")
        manifest = {
            **report,
            "config": None,
            "source_hashes": source_hashes,
            "provenance": provenance,
            "tokenizer_source": str(tokenizer_root),
            "tokenizer_source_hashes": tokenizer_hashes,
            "tokenizer_transforms": tokenizer_transforms,
            "weight_transforms": "native values and dtypes preserved; legacy norm rename only",
            "validation": {"strict_inventory": True, "exact_tensor_roundtrip": True},
            "files": {
                str(p.relative_to(staging)): sha256_file(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        }
        manifest.pop("config")
        write_json(staging / "manifest.json", manifest)
        # Reserve destination exclusively, then publish complete files into it. A
        # completion marker is written last; no existing output is ever replaced.
        output.mkdir()
        for path in staging.iterdir():
            shutil.move(str(path), output / path.name)
        write_json(
            output / "EXPORT_COMPLETE.json",
            {"manifest_sha256": sha256_file(output / "manifest.json")},
        )
        return manifest
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "export"):
        command = subparsers.add_parser(name)
        command.add_argument("checkpoint", type=Path)
        command.add_argument(
            "--output",
            required=True,
            type=Path,
            help="New JSON file (inventory) or new directory (export)",
        )
        if name == "export":
            command.add_argument("--source-repo", required=True, type=Path)
            command.add_argument("--tokenizer", type=Path)
            command.add_argument("--shard-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    if args.command == "inventory":
        report = inventory_checkpoint(args.checkpoint)[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    else:
        report = export_checkpoint(
            args.checkpoint,
            args.output,
            args.source_repo,
            args.tokenizer,
            args.shard_bytes,
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "architecture",
                    "tensor_count",
                    "parameter_elements",
                    "buffer_elements",
                    "weight_bytes",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
