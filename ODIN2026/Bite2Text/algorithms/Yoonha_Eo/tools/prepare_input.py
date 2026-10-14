"""Stage one paired IOS + IOP case into the submission's input socket layout."""
import argparse
import json
import shutil
from pathlib import Path


def prepare(upper: Path, lower: Path, photos: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for scan in (upper, lower):
        if not scan.is_file() or scan.suffix.lower() not in {".stl", ".obj"}:
            raise ValueError(f"Expected an STL or OBJ scan: {scan}")
    if not photos or any(not p.is_file() for p in photos):
        raise ValueError("Provide existing intraoral photograph files")
    if len({p.name for p in photos}) != len(photos):
        raise ValueError("Photograph basenames must be unique")
    allowed = {".mha", ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp"}
    if any(p.suffix.lower() not in allowed for p in photos):
        raise ValueError("Unsupported photograph format")
    for arch, scan in (("upper", upper), ("lower", lower)):
        target = output / "files" / f"ios-{arch}"
        target.mkdir(parents=True)
        shutil.copyfile(scan, target / f"ios_{arch}{scan.suffix.lower()}")
    target = output / "images/intraoral-photo"
    target.mkdir(parents=True)
    for photo in photos:
        shutil.copyfile(photo, target / photo.name)
    manifest = [{"socket": {"slug": slug}} for slug in
                ("3d-upper-teeth-scan", "3d-lower-teeth-scan", "2d-intraoral-photographs")]
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper", required=True, type=Path)
    parser.add_argument("--lower", required=True, type=Path)
    parser.add_argument("--photos", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.upper, args.lower, args.photos, args.output)
    print(f"Input staged at {args.output}")
