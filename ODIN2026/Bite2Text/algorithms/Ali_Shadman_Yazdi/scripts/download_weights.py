#!/usr/bin/env python
"""Download the CLIK crown-landmark checkpoints and verify every weight file.

The four ``[Crown]*.pt`` detectors (618 MB each) are published by the
CLIK-Diffusion authors (https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion)
on Google Drive and are fetched from there rather than redistributed:
https://drive.google.com/drive/folders/1o9tVJ6p8Jbad3gu0ZUkX0tE5dp7Vkh9g

    python scripts/download_weights.py            # download what is missing, verify all
    python scripts/download_weights.py --check    # verify only, exit 1 if anything is wrong
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("BITE2TEXT_WEIGHTS", ROOT / "weights"))

DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

CLIK = {  # file name -> (Google Drive file id, size in bytes, sha256)
    "[Crown]incisor-e837.pt":  ("1_63awewRKDtX31EanS0rR7WrcKvzBZoU", 648252378,
                                "fdedcc05daa03c2d962e113f5cf0a0d5dfcc386f62f4065a19b6a4ed1f7fec19"),
    "[Crown]cuspid-e879.pt":   ("1yvbrDQoOaSv4yto4mXIDf-m6rOfnWYZv", 648252378,
                                "f2bb34e9a69b5f279fbcde9d3885d0c433cd45172ab6aebf3f47d02e0e5e8b67"),
    "[Crown]premolar-e443.pt": ("1_242qBP62YVeoFtlsaUTAzJ7sMetz8tG", 648253914,
                                "429bd61e191c083a974d44b845dd5a746880592c18288ca0ce16e32522c10cf6"),
    "[Crown]molar-e178.pt":    ("1Y7UYoSfzoFzSqevlmmFL630GJW6hz9wY", 648256922,
                                "0459fcb588cc2dc0b8f56e6188f3df25517b240d8308c4af5758ebd6f6c477a5"),
}

BUNDLED = {  # shipped in the repository
    "tsgcnet/upper.pth":  "16d0efaa223d82386da67f9077c2500635f201cc196c43e09191b1170761e1cd",
    "tsgcnet/lower.pth":  "62e36d6579a53706c837cbfb0e0e3d8ce645bf2f1a43f444805b1748f8838d87",
    "photo_cnn/best.pt":  "3b2abc64f2d4d153f48d81718e53836d38208a3ede7d1450aa3ad926d0885117",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(file_id: str, size: int, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(DRIVE_URL.format(id=file_id), headers={"User-Agent": "bite2text"})
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Google Drive returned a web page instead of the file "
                               "(quota exceeded?). Download it manually from the folder "
                               "linked in weights/README.md.")
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            print(f"\r  {dest.name}: {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    part.rename(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only; do not download")
    args = ap.parse_args(argv)

    ok = True
    clik_dir = WEIGHTS / "clik"
    clik_dir.mkdir(parents=True, exist_ok=True)

    for name, (file_id, size, digest) in CLIK.items():
        path = clik_dir / name
        if not path.is_file() and not args.check:
            print(f"downloading {name}")
            download(file_id, size, path)
        ok &= verify(path, digest)

    for name, digest in BUNDLED.items():
        ok &= verify(WEIGHTS / name, digest)

    print("all weights present and verified" if ok else "some weights are missing or corrupt")
    return 0 if ok else 1


def verify(path: Path, digest: str) -> bool:
    if not path.is_file():
        print(f"  MISSING  {path}")
        return False
    if sha256(path) != digest:
        print(f"  BAD HASH {path}")
        return False
    print(f"  ok       {path.relative_to(WEIGHTS)}")
    return True


if __name__ == "__main__":
    sys.exit(main())
