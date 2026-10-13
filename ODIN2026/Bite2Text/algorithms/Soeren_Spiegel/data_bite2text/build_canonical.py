#!/usr/bin/env python3
"""Assemble the canonical training targets: splits/canonical/<split>.jsonl,
one line per case: {"case_id", "record": <raw_cases JSON>, "prose": render_prose(record)}.
Consumed by bite2text_pipeline.dataset.load_split_records. Re-runnable; overwrites."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_bite2text.chart_render import render_prose  # noqa: E402

splits_dir = REPO / "data_bite2text" / "splits"
out_dir = splits_dir / "canonical"
out_dir.mkdir(exist_ok=True)
for split in ("train", "val", "test"):
    ids = json.loads((splits_dir / f"{split}.json").read_text())
    n, lens = 0, []
    with (out_dir / f"{split}.jsonl").open("w") as f:
        for cid in ids:
            rec = json.loads((REPO / "data_bite2text" / "raw_cases" / f"{cid}.json").read_text())
            prose = render_prose(rec)
            f.write(json.dumps({"case_id": cid, "record": rec, "prose": prose}) + "\n")
            n += 1
            lens.append(len(prose))
    lens.sort()
    print(f"{split}: {n} targets, prose chars min/med/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
