#!/usr/bin/env python3
"""
code/train/dataset.py -- the rows an SFT arm trains on, and how they are read.

Split out of trainer.py (was train_vision_lora.py) on 2026-08-21. Both halves
are about WHICH rows exist and when they are encoded, not about the training
step, and keeping them here is what lets a change to row selection be read
without opening the trainer.

The encoding itself is sft_collator.ToothCallCollator; this module decides
when it runs (lazily, in dataloader workers) and what happens when it fails.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(
    p for p in _Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import add_code_paths  # noqa: E402

add_code_paths()

import sft_collator as SC   # noqa: E402
import sft_prompt as SP     # noqa: E402


class RowDataset:
    """Rows, encoded lazily so dataloader workers do the image preprocessing.

    Encoding is ~0.4 s of CPU per row (resize to 1344^2, patchify), which over
    3,500 rows x 2 epochs is ~45 min. In the main process that is 45 min the
    GPU spends idle; in workers it overlaps the step it belongs to.

    A row that cannot be encoded is dropped ON FIRST TOUCH and counted, and the
    epoch is short by one rather than failing -- but the count is printed at the
    end, because a silent coverage difference between arms is exactly what the
    plan's conventions forbid ("a lost call is a difference in coverage between
    the arms, not a finding about them").
    """

    def __init__(self, rows: list[dict], collator: SC.ToothCallCollator):
        self.rows = rows
        self.collator = collator
        self.dropped: collections.Counter = collections.Counter()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        try:
            return self.collator.encode(self.rows[i])
        except (SC.TooLong, SP.NoImage, ValueError) as e:
            self.dropped[type(e).__name__] += 1
            print(f"[DROP] {type(e).__name__}: {e}", file=sys.stderr)
            # Fall forward to the next usable row rather than returning None:
            # a None would have to be handled in the collator, and a batch of
            # one cannot be made empty.
            return self[(i + 1) % len(self.rows)]



def load_rows(path: Path, case_list: Path | None, limit: int | None,
              seed: int) -> list[dict]:
    import random

    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if case_list:
        keep = {l.strip() for l in case_list.read_text().splitlines()
                if l.strip() and not l.startswith("#")}
        rows = [r for r in rows if r["case_id"] in keep]
        print(f"[INFO] {case_list.name}: {len(keep)} case(s) -> {len(rows)} row(s)")
    if limit:
        # Seeded and case-shuffled: taking the first N rows of the file takes
        # the first few CASES, and a 200-row sanity check that saw 10 cases is
        # not a sanity check on the pool.
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows
