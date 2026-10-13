"""Run a local CBCT through the container without network access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One local .mha CBCT volume")
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--image", default="repgen:1.0.0")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    source = args.input.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not source.is_file() or source.suffix.lower() != ".mha":
        parser.error("Input must be an existing .mha file.")
    if not (weights / "model_manifest.json").is_file():
        parser.error("Download and verify the model weights first.")
    if os.getuid() == 0:
        parser.error("Run from a non-root user account.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repgen-") as temporary:
        inputs = Path(temporary) / "input"
        images = inputs / "images/cbct"
        images.mkdir(parents=True)
        shutil.copyfile(source, images / "case.mha")
        (inputs / "inputs.json").write_text(json.dumps([{"socket": {"slug": "cbct-image"}}]))
        command = ["docker", "run", "--rm", "--platform", "linux/amd64",
                   "--gpus", "device=" + args.gpu, "--network", "none", "--read-only",
                   "--memory", "32g", "--shm-size", "16g",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "--tmpfs", "/tmp:rw,exec,size=8g",
                   "--mount", f"type=bind,source={inputs},target=/input,readonly",
                   "--mount", f"type=bind,source={weights},target=/opt/ml/model,readonly",
                   "--mount", f"type=bind,source={output},target=/output", args.image]
        subprocess.run(command, check=True)
    report = output / "diagnostic-imaging-report.json"
    value = json.loads(report.read_text())
    if not isinstance(value.get("report"), str) or not value["report"].strip():
        raise SystemExit("The algorithm did not produce a valid report.")
    print(report)


if __name__ == "__main__":
    main()
