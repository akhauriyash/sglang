"""CPU regression tests for strict metadata handling and exact DCP restoration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

try:
    from .export import export_checkpoint, inventory_checkpoint, sha256_file
    from .schema import expected_tensors, resolve_config, validate_inventory
except ImportError:
    from export import export_checkpoint, inventory_checkpoint, sha256_file
    from schema import expected_tensors, resolve_config, validate_inventory


def native_config(kind="full"):
    family = {
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "v_head_dim": 4,
        "rope_head_dim": 4,
        "rope_base": 500000.0,
        "rope_scaling": None,
        "causal_conv_width": 4,
        "apply_bias_term": True,
        "timenorm_num_groups": 2,
        "local_attention_window": 8,
        "causal_conv_weight_normalization": kind != "inkling",
    }
    model = {
        "arch": "transformer" if kind == "full" else f"hybrid_{kind}",
        "model_dim": 6,
        "ffn_hidden_dim": 10,
        "num_layers": 2,
        "vocab_size": 16,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "v_head_dim": None,
        "norm_eps": 1e-6,
        "rope_base": 1000000.0,
        "rope_head_dim": 4,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": None,
        },
        "max_position_embeddings": 128,
        "gekko_value_activation": "silu",
    }
    if kind != "full":
        model.update(
            layer_kinds=["full", kind],
            full_attention={**family, "rope_base": 1000000.0},
            **{f"{kind}_attention": {**family, "gekko_value_activation": "identity"}},
        )
    return {"model": model, "dtype": "bf16"}


class ConfigTests(unittest.TestCase):
    def test_explicit_head_dimension_and_native_offsets(self):
        config = resolve_config(native_config())
        self.assertEqual(config["head_dim"], 4)
        self.assertNotEqual(
            config["head_dim"], config["hidden_size"] // config["num_attention_heads"]
        )
        self.assertEqual(config["rms_norm_eps"], 1e-6)
        self.assertTrue(config["native_norm_offset"])
        self.assertIsNone(
            config["full_attention"]["rope_scaling"]["original_max_position_embeddings"]
        )

    def test_resolved_family_overrides_take_precedence(self):
        saved = native_config("gekko")
        config = resolve_config(saved)
        self.assertEqual(
            config["gekko_attention"]["gekko_value_activation"], "identity"
        )
        self.assertEqual(
            config["native_model_config"]["gekko_value_activation"], "silu"
        )
        self.assertEqual(config["layer_kinds"], ["full", "gekko"])
        self.assertEqual(saved, native_config("gekko"))

    def test_unknown_operator_fails(self):
        saved = native_config()
        saved["model"]["head_dim"] = None
        with self.assertRaisesRegex(ValueError, "explicit positive integer"):
            resolve_config(saved)
        saved = native_config("inkling")
        saved["model"]["inkling_attention"]["causal_conv_weight_normalization"] = True
        with self.assertRaisesRegex(ValueError, "raw convolution"):
            resolve_config(saved)

    def test_missing_extra_collision_and_shape_fail(self):
        config = resolve_config(native_config())
        inventory = {
            name: {**spec, "dtype": "float32"}
            for name, spec in expected_tensors(config).items()
        }
        validate_inventory(config, inventory)
        missing = dict(inventory)
        del missing["embed.weight"]
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_inventory(config, missing)
        with self.assertRaisesRegex(ValueError, "unexpected="):
            validate_inventory(
                config,
                {**inventory, "optimizer.moment": {"shape": [1], "dtype": "float32"}},
            )
        with self.assertRaisesRegex(ValueError, "Duplicate mapping"):
            validate_inventory(
                config,
                {
                    **inventory,
                    "layers.0.norm.weight": inventory["layers.0.attention.norm.weight"],
                },
            )
        inventory["embed.weight"] = {"shape": [1], "dtype": "float32"}
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            validate_inventory(config, inventory)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text('{"model": {"vocab": {"a": 0}}}')
        (self.tokenizer / "generation_config.json").write_text(
            '{"eos_token_id": [1,250019]}'
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_checkpoint(self, kind="full"):
        checkpoint = self.root / kind
        checkpoint.mkdir()
        saved = native_config(kind)
        (checkpoint / "config.json").write_text(json.dumps(saved))
        state = {}
        for index, (name, spec) in enumerate(
            expected_tensors(resolve_config(saved)).items()
        ):
            source = (
                name.replace(".attention.norm.weight", ".norm.weight")
                if kind == "full"
                else name
            )
            dtype = torch.int64 if source.endswith("prior_count") else torch.bfloat16
            state[source] = torch.full(
                spec["shape"],
                index if dtype == torch.int64 else index / 100,
                dtype=dtype,
            )
        dcp.save(state, checkpoint_id=checkpoint / "sharded_model.tp00", no_dist=True)
        return checkpoint, state

    def export(self, checkpoint, target):
        # A fixture need not be a Git repository; production requires identities.
        with patch(
            "export.git_identity"
            if __package__ in (None, "")
            else f"{__package__}.export.git_identity",
            return_value={"commit": "test", "dirty": False},
        ):
            return export_checkpoint(
                checkpoint, target, self.root, self.tokenizer, shard_bytes=512
            )

    def test_dcp_roundtrip_legacy_names_and_immutable_destination(self):
        checkpoint, state = self.make_checkpoint()
        target = self.root / "export"
        manifest = self.export(checkpoint, target)
        self.assertEqual(len(state), manifest["tensor_count"])
        self.assertEqual(
            manifest["tensors"]["layers.0.norm.weight"]["transform"],
            "rename_legacy_attention_norm",
        )
        weights = json.loads((target / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        name = "layers.0.attention.norm.weight"
        with safe_open(target / weights[name], framework="pt") as restored:
            self.assertTrue(
                torch.equal(restored.get_tensor(name), state["layers.0.norm.weight"])
            )
        self.assertEqual(
            json.loads((target / "generation_config.json").read_text())["eos_token_id"],
            1,
        )
        self.assertTrue((target / "EXPORT_COMPLETE.json").is_file())
        with self.assertRaises(FileExistsError):
            self.export(checkpoint, target)

    def test_gekko_persistent_priors_and_deterministic_shards(self):
        checkpoint, _ = self.make_checkpoint("gekko")
        first, second = self.root / "one", self.root / "two"
        manifest = self.export(checkpoint, first)
        self.export(checkpoint, second)
        self.assertEqual(manifest["buffer_elements"], 5)
        for name in ("prior_count", "prior_mean", "prior_var"):
            self.assertEqual(
                manifest["tensors"][f"layers.1.gda.timenorm.{name}"]["role"], "buffer"
            )
        self.assertEqual(
            {p.name: sha256_file(p) for p in first.glob("*.safetensors")},
            {p.name: sha256_file(p) for p in second.glob("*.safetensors")},
        )

    def test_inkling_all_convolutions_preserved(self):
        checkpoint, _ = self.make_checkpoint("inkling")
        report = inventory_checkpoint(checkpoint)[0]
        self.assertEqual(sum("shortconv" in name for name in report["tensors"]), 4)

    def test_tokenizer_cannot_overwrite_export_metadata_or_shards(self):
        checkpoint, _ = self.make_checkpoint()
        target = self.root / "export"
        for name in (
            "config.json",
            "native_config.json",
            "manifest.json",
            "model.safetensors.index.json",
            "EXPORT_COMPLETE.json",
            "graft_receipt.json",
            "model-00001-of-00001.safetensors",
        ):
            with self.subTest(name=name):
                collision = self.tokenizer / name
                collision.write_bytes(b"tokenizer source must remain intact")
                try:
                    with self.assertRaisesRegex(ValueError, "names reserved"):
                        self.export(checkpoint, target)
                    self.assertFalse(target.exists())
                    self.assertEqual(
                        collision.read_bytes(), b"tokenizer source must remain intact"
                    )
                finally:
                    collision.unlink()

    def test_tokenizer_directory_cannot_occupy_reserved_output_name(self):
        checkpoint, _ = self.make_checkpoint()
        collision = self.tokenizer / "EXPORT_COMPLETE.json"
        collision.mkdir()
        (collision / "data.json").write_text("{}")
        target = self.root / "export"
        with self.assertRaisesRegex(ValueError, "names reserved"):
            self.export(checkpoint, target)
        self.assertFalse(target.exists())

    def test_truncated_checkpoint_fails_before_allocation(self):
        checkpoint, _ = self.make_checkpoint()
        payload = next((checkpoint / "sharded_model.tp00").glob("*.distcp"))
        payload.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Truncated DCP"):
            inventory_checkpoint(checkpoint)

    def test_published_model_directory_symlink(self):
        checkpoint, _ = self.make_checkpoint()
        published = self.root / "published"
        published.mkdir()
        (published / "config.json").write_bytes(
            (checkpoint / "config.json").read_bytes()
        )
        (published / "sharded_model.tp00").symlink_to(
            checkpoint / "sharded_model.tp00", target_is_directory=True
        )
        report = inventory_checkpoint(published)[0]
        self.assertIn("sharded_model.tp00/.metadata", report["source_hashes"])

    def test_qualification_loader_reads_safetensors_keys_on_cpu(self):
        try:
            from .qualify import import_reference, load_export
        except ImportError:
            from qualify import import_reference, load_export

        checkpoint, state = self.make_checkpoint("gekko")
        destination = self.root / "qualified"
        self.export(checkpoint, destination)
        model, manifest = load_export(destination, import_reference(), device="cpu")
        self.assertEqual(set(model.state_dict()), set(state))
        self.assertEqual(manifest["tensor_count"], len(state))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name].to(tensor.dtype)), name)


if __name__ == "__main__":
    unittest.main()
