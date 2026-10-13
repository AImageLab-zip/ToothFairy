"""Public-data-free checks for asset handling and reproduction entry points."""
import hashlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = load("assets", ROOT / "scripts/v9_assets.py")
prepare = load("prepare", ROOT / "scripts/prepare_reproduction.py")
geometry = load("geometry", ROOT / "task2_bite2text/ptv3_finetune/evaluate_geometry_retrieval.py")
cohort = load("cohort", ROOT / "scripts/check_reproduction_data.py")


class AssetsTests(unittest.TestCase):
    def test_model_license_matches_published_checkpoints(self):
        manifest = json.loads(assets.MANIFEST.read_text())
        notice = (ROOT / "MODEL_LICENSE").read_text()
        for name in ("model_final.pth", "photo_model_final.pt", "photo_view_classifier.pt"):
            with self.subTest(checkpoint=name):
                self.assertIn(name, notice)
                self.assertIn(manifest["model_files"][name]["sha256"], notice)
        self.assertIn(manifest["archives"]["weights"]["sha256"], notice)
        self.assertIn("SPDX-License-Identifier: MIT", notice)
        # Preserve the standard grant and disclaimer, not a shortened license.
        mit_text = (ROOT / "LICENSE").read_text().split("\n\n", 1)[1]
        self.assertIn(mit_text.strip(), notice)
        self.assertIn("does not relicense IOS-Normalizer", notice)

    def test_release_manifest(self):
        manifest = json.loads(assets.MANIFEST.read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(set(manifest["archives"]), {"submission", "weights"})
        self.assertEqual(len(manifest["archives"]["submission"]["metadata_files"]), 2)
        self.assertEqual(len(manifest["model_files"]), 9)
        for entry in [*manifest["archives"].values(), manifest["image_archive"],
                      manifest["model_archive"], *manifest["model_files"].values()]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)

    def test_invalid_member_names(self):
        for name in ("../a", "/a", "a/b", "a\\b", ".", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                assets.safe_flat_name(name)
        self.assertEqual(assets.safe_flat_name("./weights.pt"), "weights.pt")

    def make_archive(self, root, name="weights.pt", symlink=False):
        payload = b"test-only-not-a-checkpoint"
        archive = root / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/escape"
            tar.addfile(member, io.BytesIO(payload))
        manifest = {"model_archive": {"bytes": archive.stat().st_size, "sha256": assets.sha256(archive)},
                    "model_files": {"weights.pt": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
        return archive, manifest

    def test_verified_extract_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.make_archive(root)
            assets.extract_model(archive, root / "model", manifest)
            assets.verify_model(root / "model", manifest)
            with self.assertRaises(ValueError):
                assets.extract_model(archive, root / "model", manifest)
            (root / "model/weights.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                assets.verify_model(root / "model", manifest)

    def test_no_extraction_on_checksum_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.make_archive(root)
            manifest["model_archive"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                assets.extract_model(archive, root / "model", manifest)
            self.assertFalse((root / "model").exists())

    def test_reject_traversal_and_symlinks(self):
        for name, symlink in (("../weights.pt", False), ("weights.pt", True)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive, manifest = self.make_archive(root, name, symlink)
                with self.assertRaises(ValueError):
                    assets.extract_model(archive, root / "model", manifest)
                self.assertFalse((root / "model").exists())

    def make_submission(self, root, extra_member=None):
        model_archive, manifest = self.make_archive(root)
        image_bytes = b"synthetic-docker-export"
        image_name = "image.tar.gz"
        manifest["model_archive"]["filename"] = model_archive.name
        manifest["image_archive"] = {"filename": image_name, "bytes": len(image_bytes),
                                     "sha256": hashlib.sha256(image_bytes).hexdigest()}
        # Keep compatibility with metadata filenames in the already-published ZIP.
        metadata = json.loads(assets.MANIFEST.read_text())["archives"]["submission"]["metadata_files"]
        archive = root / "submission.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.write(model_archive, model_archive.name)
            bundle.writestr(image_name, image_bytes)
            for name in metadata:
                bundle.writestr(name, b"synthetic metadata")
            if extra_member:
                bundle.writestr(extra_member, b"unexpected")
        manifest["archives"] = {"submission": {"bytes": archive.stat().st_size,
            "sha256": assets.sha256(archive), "metadata_files": metadata}}
        return archive, manifest

    def test_unpack_submission_preserves_published_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.make_submission(root)
            assets.unpack_submission(archive, root / "release", manifest)
            assets.verify_model(root / "release/model", manifest)
            for name in manifest["archives"]["submission"]["metadata_files"]:
                self.assertTrue((root / "release" / name).is_file())
            with self.assertRaises(ValueError):
                assets.unpack_submission(archive, root / "release", manifest)

    def test_submission_rejects_unexpected_members_before_extraction(self):
        for member in ("unexpected.txt", "../escape.txt"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive, manifest = self.make_submission(root, member)
                with self.assertRaises(ValueError):
                    assets.unpack_submission(archive, root / "release", manifest)
                self.assertFalse((root / "release").exists())

    def test_semantic_command_line_names(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/v9_assets.py"),
                                 "--help"], capture_output=True, text=True, check=True)
        self.assertIn("unpack-submission", result.stdout)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/v9_assets.py"),
                                 "verify-archive", "--help"], capture_output=True, text=True, check=True)
        self.assertIn("{submission,weights}", result.stdout)


class InputTests(unittest.TestCase):
    def test_assembly_rejects_stale_retrieval_hashes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptv3, data, retrieval = (root / name for name in ("ptv3", "data", "retrieval"))
            (ptv3 / "model").mkdir(parents=True)
            data.mkdir()
            retrieval.mkdir()
            for file in (ptv3 / "model/model_last.pth", ptv3 / "config.py",
                         root / "photo.pt", root / "view.pt", root / "normalizer.pt"):
                file.write_bytes(b"synthetic-not-a-checkpoint")
            (data / "head_vocabs.json").write_text('{}')
            (data / "full_dataset_audit.json").write_text('{"train_cases": 867}')
            patients = [f"synthetic{i}" for i in range(867)]
            np.savez(retrieval / "retrieval_index.npz", patient_ids=patients,
                     descriptors=np.zeros((867, 3720), dtype=np.float32),
                     mean=np.zeros(3720), scale=np.ones(3720))
            (retrieval / "retrieval_reports.json").write_text(json.dumps({
                "patient_ids": patients, "reports": ["Synthetic report."] * 867,
                "index_sha256": "0" * 64}))
            (retrieval / "retrieval_labels.json").write_text(json.dumps({
                "version": "bite2text-hybrid-labels-v1", "patient_ids": patients,
                "target_values": [{} for _ in patients], "retrieval_reports_sha256": "0" * 64}))
            def assemble():
                return prepare.assemble(root / "model", ptv3, data, root / "photo.pt",
                                        root / "view.pt", root / "normalizer.pt", retrieval)
            with self.assertRaisesRegex(ValueError, "index checksum"):
                assemble()
            self.assertFalse((root / "model").exists())
            report_file = retrieval / "retrieval_reports.json"
            reports = json.loads(report_file.read_text())
            reports["index_sha256"] = assets.sha256(retrieval / "retrieval_index.npz")
            report_file.write_text(json.dumps(reports))
            with self.assertRaisesRegex(ValueError, "report checksum"):
                assemble()
            self.assertFalse((root / "model").exists())
            label_file = retrieval / "retrieval_labels.json"
            labels = json.loads(label_file.read_text())
            labels["retrieval_reports_sha256"] = assets.sha256(report_file)
            label_file.write_text(json.dumps(labels))
            self.assertEqual(assemble()["files"], 9)

    def test_cohort_rejects_leaking_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, cv, full = (root / name for name in ("data", "cv", "full"))
            observed = {f"synthetic{i}" for i in range(5)}
            for path, selected in ((data, observed | {"unlabeled"}), (full, observed)):
                self.make_cohort(path, selected)
            cv.mkdir()
            with (cv / "fold_assignments.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["patient_id", "fold"])
                writer.writerows((f"synthetic{i}", i + 1) for i in range(5))
            for i in range(5):
                fold = cv / f"fold{i + 1}"
                self.make_cohort(fold, observed)
                sample = fold / "train" / f"dental_synthetic{i}.npz"
                sample.rename(fold / "val" / sample.name)
                (fold / "test" / sample.name).touch()
            self.assertEqual(cohort.check(data, cv, full, 6, 5)["supervised_cases"], 5)
            (cv / "fold1/train/dental_synthetic0.npz").touch()
            with self.assertRaises(ValueError):
                cohort.check(data, cv, full, 6, 5)

    @staticmethod
    def make_cohort(root, selected):
        for split in ("train", "val", "test"):
            (root / split).mkdir(parents=True)
        for pid in selected:
            (root / "train" / f"dental_{pid}.npz").touch()
        with (root / "labels.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", *[f"label_{i}" for i in range(12)]])
            writer.writerows([pid, *([-1 if pid == "unlabeled" else 0] * 12)] for pid in sorted(selected))
        (root / "head_vocabs.json").write_text(json.dumps({f"head{i}": ["value"] for i in range(12)}))

    def test_stage_read_only_pair_and_skip_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            for patient in ("synthetic1", "synthetic2"):
                (raw / patient / "ios").mkdir(parents=True)
                (raw / patient / "ios/ios_lower.stl").write_bytes(b"synthetic")
            (raw / "synthetic1/ios/ios_upper.stl").write_bytes(b"synthetic")
            result = prepare.stage_ios(raw, root / "staged")
            self.assertEqual(result, {"pairs": 1, "skipped_missing_pair": 1})
            self.assertTrue((root / "staged/synthetic1/ios_lower.stl").is_symlink())
            with self.assertRaises(ValueError):
                prepare.stage_ios(raw, root / "staged")

    def test_geometry_validation_and_shape(self):
        import numpy as np
        for data in (np.zeros((4, 3)), np.ones((5, 3)), np.ones((2, 3)), np.full((4, 3), np.nan)):
            with self.assertRaises(ValueError):
                geometry.descriptor_from_coord(data)
        self.assertEqual(geometry.descriptor_from_coord(np.random.default_rng(1).normal(size=(20, 3))).shape, (3720,))


if __name__ == "__main__":
    unittest.main()
