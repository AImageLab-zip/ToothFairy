"""Download, verify and unpack the public inference-only weight release."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    release = json.loads((Path(__file__).resolve().parents[1] / "configs/release.json").read_text())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Choose an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    archive = output.parent / (output.name + ".download.tar.gz")
    url = release["weights"]["url"]
    print("Downloading inference weights from the public release...")
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as handle:
        while block := response.read(8 * 1024**2):
            handle.write(block)
    if sha256(archive) != release["weights"]["sha256"]:
        raise SystemExit("Archive checksum mismatch; extraction cancelled.")
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (output / member.name).resolve()
            if not target.is_relative_to(output) or not (member.isfile() or member.isdir()):
                raise SystemExit("Unsafe archive member; extraction cancelled.")
        tar.extractall(output, filter="data")
    manifest = json.loads((output / "weights_manifest.json").read_text())
    for item in manifest["files"]:
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(output) or sha256(path) != item["sha256"]:
            raise SystemExit("Extracted file checksum mismatch.")
    archive.unlink()
    print("Weights verified. Run: python scripts/run_case.py --help")


if __name__ == "__main__":
    main()
