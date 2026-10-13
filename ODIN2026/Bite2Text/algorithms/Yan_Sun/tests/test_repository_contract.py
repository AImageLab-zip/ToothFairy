"""Lightweight repository-wide checks; never import training or inference code."""
import ast
from pathlib import Path
import re
import subprocess
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        cls.names = set(output.decode().rstrip("\0").split("\0"))

    def test_all_tracked_python_syntax(self):
        for name in sorted(self.names):
            if name.endswith(".py"):
                with self.subTest(file=name):
                    ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_all_tracked_shell_syntax(self):
        for name in sorted(self.names):
            if name.endswith((".sh", ".command")):
                with self.subTest(file=name):
                    result = subprocess.run(["bash", "-n", str(ROOT / name)],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_markdown_has_no_questionnaire_numbering(self):
        for name in sorted(self.names):
            if name.endswith(".md"):
                with self.subTest(file=name):
                    self.assertNotRegex((ROOT / name).read_text(), r"\bQ\d{2,}\b")

    def test_local_inference_imports_are_staged_and_copied_into_image(self):
        source_dir = ROOT / "task2_bite2text/ptv3_finetune"
        tree = ast.parse((source_dir / "inference.py").read_text())
        local_imports = {
            node.module + ".py" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            and (source_dir / (node.module + ".py")).is_file()
        }
        self.assertIn("report_renderer.py", local_imports)
        dockerfile = (ROOT / "task2_bite2text/ptv3_submission/Dockerfile").read_text()
        copied = set()
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                copied.update(line.split()[1:-1])
        bootstrap = (ROOT / "scripts/bootstrap_upstreams.sh").read_text()
        staged = set(re.search(r"for source_name in (.+); do", bootstrap).group(1).split())
        for module in sorted(local_imports):
            with self.subTest(module=module):
                self.assertIn(module, staged, "Local inference dependency is not staged")
                self.assertIn(module, copied, "Local inference dependency is absent from image")

    def test_local_markdown_file_links_exist_in_repository(self):
        for name in sorted(self.names):
            if not name.endswith(".md"):
                continue
            for link in re.findall(r"\]\(([^\s)]+)\)", (ROOT / name).read_text()):
                parts = urlsplit(link)
                if parts.scheme or parts.netloc or not parts.path:
                    continue
                with self.subTest(file=name, link=link):
                    target = ((ROOT / name).parent / unquote(parts.path)).resolve()
                    relative = str(target.relative_to(ROOT))
                    self.assertTrue(relative in self.names or any(
                        path.startswith(relative + "/") for path in self.names
                    ), "Link must point to a tracked file/directory, not a local-only artifact")


if __name__ == "__main__":
    unittest.main()
