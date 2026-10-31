"""Historical gates apply to NLL differences, not raw logit magnitudes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn.functional as F

try:
    from .qualify import capture_source_provenance, compare, compare_nll
except ImportError:
    from qualify import capture_source_provenance, compare, compare_nll


class MetricTests(unittest.TestCase):
    def test_provenance_retains_initial_native_bytes_after_source_changes(self):
        with TemporaryDirectory() as temporary:
            native = Path(temporary)
            (native / "xllm").mkdir()
            source = native / "xllm/operator.py"
            source.write_text("VERSION = 1\n")
            with patch.dict(
                capture_source_provenance.__globals__,
                {"git_identity": lambda path: {"commit": "fixed", "dirty": False}},
            ):
                before = capture_source_provenance(native)
                self.assertTrue(before["reference_sources"])
                self.assertIn("xllm_rmsnorm.py", before["reference_sources"])
                self.assertIn("xllm_tdn.py", before["reference_sources"])
                self.assertIn("xllm_sca.py", before["reference_sources"])
                self.assertEqual(
                    set(before["attention_sources"]),
                    {"interface.py", "flash_fwd_sm90.py", "softmax.py"},
                )
                original_hash = before["native_sources"]["xllm/operator.py"]
                source.write_text("VERSION = 2\n")
                after = capture_source_provenance(native)
            self.assertEqual(
                before["native_sources"]["xllm/operator.py"], original_hash
            )
            self.assertNotEqual(before, after)

    def test_logit_offset_is_not_a_nll_error(self):
        logits = torch.tensor([[100.0, 101.0, 99.0]])
        shifted = logits + 0.1
        self.assertGreater(compare(logits, shifted)["max_abs"], 0.08)
        self.assertTrue(compare(logits, shifted)["passed"])
        target = torch.tensor([1])
        self.assertTrue(
            compare_nll(
                F.cross_entropy(logits, target, reduction="none"),
                F.cross_entropy(shifted, target, reduction="none"),
            )["passed"]
        )

    def test_nll_mean_and_maximum_have_independent_gates(self):
        reference = torch.ones(100)
        mean_failure = reference + 0.006
        maximum_failure = reference.clone()
        maximum_failure[-1] += 0.09
        self.assertFalse(compare_nll(reference, mean_failure)["passed"])
        self.assertLess(compare_nll(reference, maximum_failure)["mean_abs"], 0.005)
        self.assertFalse(compare_nll(reference, maximum_failure)["passed"])
        self.assertTrue(compare_nll(reference, reference + 0.004)["passed"])

    def test_nonfinite_values_cannot_pass(self):
        reference = torch.ones(3)
        actual = torch.tensor([1.0, float("nan"), 1.0])
        self.assertFalse(compare(reference, actual)["passed"])
        self.assertFalse(compare_nll(reference, actual)["passed"])


if __name__ == "__main__":
    unittest.main()
