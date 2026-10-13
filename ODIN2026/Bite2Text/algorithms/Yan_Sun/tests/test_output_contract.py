"""Check the public output validator without loading any inference models."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "task2_bite2text/hybrid_submission_v9_final/verify_output.py"


class OutputContractTests(unittest.TestCase):
    def verify(self, payload, optimized=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.run(
                [sys.executable, *(["-O"] if optimized else []), str(VERIFIER), str(path)],
                capture_output=True, text=True,
            )

    def test_valid_report(self):
        for optimized in (False, True):
            result = self.verify({"report": "Synthetic test report."}, optimized)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["characters"], 22)

    def test_invalid_report(self):
        for payload in ({}, [], {"report": 1}, {"report": " "},
                        {"report": "test", "extra": True}):
            with self.subTest(payload=payload):
                self.assertNotEqual(self.verify(payload).returncode, 0)

    def test_optimization_cannot_disable_validation(self):
        for payload in ({"report": ""}, {"report": "test", "extra": True}):
            with self.subTest(payload=payload):
                self.assertNotEqual(self.verify(payload, optimized=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
