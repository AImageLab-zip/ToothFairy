"""Exercise the evaluator's dry-run configuration without any API dependencies."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RadfactResumeTests(unittest.TestCase):
    def dry_run_signature(self, rows):
        path = ROOT / "task2_bite2text/radfact_glm_eval/run_radfact_glm.py"
        # Execute the actual dry-run branch of main(), with synthetic rows and
        # fake arguments. No imports of OpenAI/radfact_lite, files or API keys.
        tree = ast.parse(path.read_text())
        names = {"canonical_json", "select_rows", "main"}
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        args = SimpleNamespace(
            input=ROOT / "synthetic.jsonl", run_dir=ROOT / "synthetic-run",
            cache_dir=None, report_type="Bite2Text", sample_size=None, seed=1,
            limit=None, model="synthetic-model", base_url="https://example.invalid/",
            api_key_env="UNUSED_TEST_KEY", filter_normal=True, dry_run=True,
        )
        namespace = {
            "json": json, "hashlib": hashlib, "random": random,
            "UPSTREAM_COMMIT": "synthetic-test",
            "build_parser": lambda: SimpleNamespace(parse_args=lambda: args),
            "ReportType": lambda value: SimpleNamespace(value=value),
            "load_jsonl": lambda path: rows,
        }
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(namespace["main"](), 0)
        return json.loads(output.getvalue())["signature"]

    def test_same_selected_content_can_resume(self):
        rows = [{"patient_id": "synthetic", "prediction": "Candidate.", "reference": "Reference."}]
        self.assertEqual(self.dry_run_signature(rows), self.dry_run_signature(rows))

    def test_changed_content_cannot_reuse_completed_results(self):
        row = {"patient_id": "synthetic", "prediction": "Candidate.", "reference": "Reference."}
        original = self.dry_run_signature([row])
        for field in ("prediction", "reference"):
            with self.subTest(field=field):
                self.assertNotEqual(original, self.dry_run_signature([{**row, field: "Changed."}]))


if __name__ == "__main__":
    unittest.main()
