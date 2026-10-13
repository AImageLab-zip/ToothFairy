#!/usr/bin/env python
"""Generate an orthodontic report for one patient.

    python predict.py \
        --upper  sample_data/F5535/ios_upper.stl \
        --lower  sample_data/F5535/ios_lower.stl \
        --photos sample_data/F5535 \
        --output outputs/F5535.json

``--photos`` takes image files and/or directories. Files named
``intraoral_1`` ... ``intraoral_5`` are mapped to front / right / left / lower /
upper views; anything else is read in name order. Without photos, the photo
section of the report comes from the cohort prior.
"""

import os

# Thread pools are sized at first import of torch / numpy / sklearn.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upper", required=True, type=Path, help="upper-jaw scan (.stl/.obj/.ply)")
    ap.add_argument("--lower", required=True, type=Path, help="lower-jaw scan (.stl/.obj/.ply)")
    ap.add_argument("--photos", nargs="*", default=[], type=Path,
                    help="intraoral photographs: files and/or directories")
    ap.add_argument("--output", type=Path, help="write the report and predicted slots as JSON")
    ap.add_argument("--landmarks", type=Path, help="also write the per-tooth landmarks as JSON")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0, help="seed for landmark point sampling")
    ap.add_argument("--quiet", action="store_true", help="only print the report")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="[bite2text] %(message)s")

    from bite2text import ReportGenerator, config
    from bite2text.photo_io import list_photos

    missing = config.missing_weights()
    if missing:
        print("Missing model weights:\n  " + "\n  ".join(map(str, missing))
              + "\nRun `python scripts/download_weights.py` first.", file=sys.stderr)
        return 1

    photos = list_photos(args.photos)
    generator = ReportGenerator(device=args.device, seed=args.seed)

    t0 = time.perf_counter()
    report = generator(upper=args.upper, lower=args.lower, photos=photos)
    report.timings["total"] = round(time.perf_counter() - t0, 2)

    if not args.quiet:
        logging.getLogger("bite2text").info("done in %.1f s on %s", report.timings["total"], generator.device)
    print(report.text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    if args.landmarks:
        args.landmarks.parent.mkdir(parents=True, exist_ok=True)
        args.landmarks.write_text(json.dumps(report.landmarks, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
