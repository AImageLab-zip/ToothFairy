#!/usr/bin/env python3
"""
sft_collator.py -- an SFT row -> input_ids, per-token labels, per-token weights.

docs/vision_sft_plan.md open item 9. Everything §3 produces is a FIELD-level
decision; this is what turns it into `-100` spans and a weight vector, which is
the only place those decisions become gradient.

Split out of trainer.py so it can be audited with no GPU and no model
load: `python code/train/sft_collator.py --rows ... --show` prints, per row, exactly
which tokens carry loss and at what weight. A mask that swallows a field should
be visible before an 8-hour job, not inferred from a loss curve.

WHAT CARRIES LOSS, AND WHAT DOES NOT
────────────────────────────────────
Only the VALUES of supervised fields. Not the braces, not the field names, not
the commas, not the closing `<|im_end|>`.

That is a deliberate boundary and it follows from the arithmetic in code/train/visual_evidence/README.md. The
evidence weight of 0.04 was set against "~230 prose tokens per row against ~10
tokens of supervised decision content" -- and ~10 is what the VALUES of ~8.3
supervised fields per row (15,407 slots / 1,850 rows) tokenize to when they are
mostly `false`, `true`, `null` and a short enum. Supervising the scaffolding
would multiply the decision side by the length of the field names and make that
ratio, and therefore the weight, wrong.

It is also right on its own terms: the shape of the answer is not what this
experiment is trying to change. At inference the object is decoded under
`response_format` (R2), so the shape is enforced by the sampler rather than
learned, and teaching it again here would spend gradient -- most of it on arm
3's side of a 1.5%-matched contrast -- on the one thing already guaranteed.

THE TWO WEIGHTS
───────────────
  1.00  a supervised decision field's value
  0.04  a supervised `visual_evidence` string (code/train/visual_evidence/README.md, --evidence-weight)
  0     everything else, and those positions are `-100` as well

Anything at weight 0 is still IN the target -- the span is tokenized and the
fields after it are conditioned on it exactly as at inference; it simply
arrives with `mask[fact][visual_evidence] == "evidence"` and carries nothing.
Which is why the CONTENT of that field is decided in build_sft_targets.py and
not here, and why it has to be uniform across tooth and arch rows: see the
`evidence` entry of its mask table. The rows this repo builds carry an empty
string in every one, so the 0.04 row above is currently unreachable -- it
applies again the moment --evidence-from supplies prose.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

ROOT = REPO_ROOT

import sft_prompt as SP                   # noqa: E402

EVIDENCE_FIELD = "visual_evidence"
IGNORE = -100


def serialize_target(target: dict, mask: dict, indent: int = 2
                     ) -> tuple[str, list[tuple[int, int, str, str]]]:
    """The target JSON, plus a char span per field VALUE.

    Hand-rolled rather than json.dumps + a search, because the spans have to be
    exact and several fields share a value: `false` appears a dozen times in a
    row and locating it by string search would attach the loss to whichever one
    came first. Emitting and recording position as it goes cannot make that
    mistake.

    Field order is the schema's own -- build_sft_targets.py wrote the dict in
    `object_fields` order and Python preserves it -- which is load-bearing:
    vLLM decodes under a grammar that follows the schema, so a target in any
    other order trains a sequence the server cannot produce.
    """
    sp1, sp2 = " " * indent, " " * (2 * indent)
    out: list[str] = []
    spans: list[tuple[int, int, str, str]] = []
    pos = 0

    def emit(s: str) -> None:
        nonlocal pos
        out.append(s)
        pos += len(s)

    emit("{\n")
    facts = list(target.items())
    for i, (fact_id, fields) in enumerate(facts):
        emit(f'{sp1}"{fact_id}": {{\n')
        items = list(fields.items())
        for j, (field, value) in enumerate(items):
            emit(f'{sp2}"{field}": ')
            start = pos
            emit(json.dumps(value, ensure_ascii=False))
            if mask.get(fact_id, {}).get(field) is None:
                spans.append((start, pos, fact_id, field))
            emit(",\n" if j < len(items) - 1 else "\n")
        emit(sp1 + ("},\n" if i < len(facts) - 1 else "}\n"))
    emit("}")
    return "".join(out), spans


class TooLong(ValueError):
    """The call does not fit --max-length. Dropped and counted, never truncated.

    §4: a truncated call loses the tail of its target -- and with the fields
    emitted in schema order, the tail is the fields the schema put last. It
    would train a systematically incomplete answer.
    """


class ToothCallCollator:
    """Rows -> a batch. Batch size 1 by design; see __call__."""

    def __init__(self, processor, evidence_weight: float = 0.04,
                 max_length: int = 8192, content_format: str | None = None,
                 root: Path = ROOT):
        self.proc = processor
        self.tok = processor.tokenizer
        self.evidence_weight = evidence_weight
        self.max_length = max_length
        self.content_format = content_format or SP.CONTENT_FORMAT
        self.root = root
        self.stats = collections.Counter()

    # ── one row ───────────────────────────────────────────────────────────
    def encode(self, row: dict) -> dict:
        import torch

        from PIL import Image

        messages, _ = SP.build_row_messages(row, self.root)
        text = SP.render_chat_text(self.proc, messages,
                                   content_format=self.content_format)
        images = [Image.open(p).convert("RGB")
                  for p in SP.image_paths(row, self.root)]
        enc = self.proc(text=[text], images=images, return_tensors="pt")

        prompt_ids = enc["input_ids"][0]
        n_prompt = int(prompt_ids.shape[0])

        target_text, spans = serialize_target(row["target"], row["mask"])
        # The assistant turn ends here. `<|im_end|>` is in the sequence so the
        # context is what the server sees, and unsupervised for the same reason
        # the braces are: stopping is not what this is teaching.
        tgt = self.tok(target_text + "<|im_end|>", add_special_tokens=False,
                       return_offsets_mapping=True)
        target_ids = tgt["input_ids"]
        offsets = tgt["offset_mapping"]
        n_target = len(target_ids)

        if n_prompt + n_target > self.max_length:
            raise TooLong(f"{SP.row_label(row)}: "
                          f"{n_prompt} + {n_target} > {self.max_length}")

        # Char spans -> token positions. A token counts as supervised when it
        # OVERLAPS a supervised value, which is what makes this robust to the
        # tokenizer merging a value with the punctuation beside it: `false,`
        # can be one token, and dropping it would silently unsupervise the
        # field. Overlap keeps it; the comma comes along, one token of noise
        # against losing the answer entirely.
        labels = [IGNORE] * n_target
        weights = [0.0] * n_target
        supervised = collections.Counter()
        for start, end, fact_id, field in spans:
            w = self.evidence_weight if field == EVIDENCE_FIELD else 1.0
            if w <= 0:
                continue
            hit = 0
            for k, (a, b) in enumerate(offsets):
                if a < end and b > start:          # overlap, not containment
                    labels[k] = target_ids[k]
                    weights[k] = max(weights[k], w)
                    hit += 1
            if not hit:
                raise ValueError(
                    f"{SP.row_label(row)}: no token covers "
                    f"{fact_id}.{field} at chars [{start},{end}) -- the offset "
                    f"mapping and the serialization disagree")
            supervised[field] += 1

        ids = torch.cat([prompt_ids, torch.tensor(target_ids)])
        out = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels),        # TARGET SPAN ONLY, length n_target
            "weights": torch.tensor(weights, dtype=torch.float32),
            "n_prompt": n_prompt,
            "n_target": n_target,
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }
        if "mm_token_type_ids" in enc:
            # The model uses this to route image positions; it is built for the
            # prompt, so the target span is plain text (0).
            mm = enc["mm_token_type_ids"][0]
            out["mm_token_type_ids"] = torch.cat(
                [mm, torch.zeros(n_target, dtype=mm.dtype)])

        self.stats["rows"] += 1
        self.stats["supervised_fields"] += sum(supervised.values())
        self.stats["supervised_tokens"] += sum(1 for w in weights if w >= 1.0)
        self.stats["evidence_tokens"] += sum(
            1 for w in weights if 0 < w < 1.0)
        return out

    # ── a batch ───────────────────────────────────────────────────────────
    def __call__(self, rows: list[dict]) -> dict:
        """Batch size 1, asserted rather than assumed.

        §4 sets batch_size=1 / grad_accum=16, and the alignment below depends
        on it: `logits_to_keep` is an int count of TRAILING positions, so it is
        only the target span when the target is at the end of the sequence --
        true for one row, false for any padded batch. Samples here are uniform
        at ~6.2k tokens, so batching buys nothing that grad accumulation does
        not, and the memory term that actually binds is the logits (§4), which
        batching makes worse.
        """
        if len(rows) != 1:
            raise ValueError(
                f"batch size {len(rows)}: this collator is batch-size-1 by "
                f"design (see the docstring); raise --grad-accum instead")
        row = rows[0]
        item = row if "input_ids" in row else self.encode(row)
        return {k: (v.unsqueeze(0) if hasattr(v, "dim") and k not in
                    ("pixel_values", "image_grid_thw") else v)
                for k, v in item.items() if k not in ("n_prompt", "n_target")} | {
            "n_prompt": item["n_prompt"], "n_target": item["n_target"]}


# ── standalone audit: no GPU, no model ────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path,
                    default=ROOT / "outputs/training_results/vsft_pool_training/sft_targets.jsonl")
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "models/Qwen3.5-9B-AWQ")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--evidence-weight", type=float, default=0.04)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--show", action="store_true",
                    help="print every supervised token of the first row")
    args = ap.parse_args()

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    col = ToothCallCollator(proc, args.evidence_weight, args.max_length)

    rows = [json.loads(l) for l in
            args.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[INFO] {len(rows)} row(s) in {args.rows.name}; "
          f"encoding {min(args.limit, len(rows))}")

    lens, dropped = [], collections.Counter()
    for i, row in enumerate(rows[:args.limit]):
        try:
            item = col.encode(row)
        except (TooLong, SP.NoImage) as e:
            dropped[type(e).__name__] += 1
            print(f"  [DROP] {type(e).__name__}: {e}")
            continue
        lens.append((item["n_prompt"], item["n_target"]))

        if args.show and i == 0:
            print(f"\n── {SP.row_label(row)} "
                  f"── prompt {item['n_prompt']} + target {item['n_target']}")
            for k, (lab, w) in enumerate(zip(item["labels"].tolist(),
                                             item["weights"].tolist())):
                if lab == IGNORE:
                    continue
                print(f"   {k:>4} w={w:<5.2f} {proc.tokenizer.decode([lab])!r}")
            print()

    if not lens:
        print("[FAIL] nothing encoded")
        return 1
    p = [a for a, _ in lens]
    t = [b for _, b in lens]
    print(f"\n[INFO] prompt tokens  min {min(p):,}  mean {sum(p)/len(p):,.0f}  max {max(p):,}")
    print(f"[INFO] target tokens  min {min(t):,}  mean {sum(t)/len(t):,.0f}  max {max(t):,}")
    print(f"[INFO] total          max {max(a + b for a, b in lens):,} "
          f"(--max-length {args.max_length:,})")
    print(f"[INFO] dropped: {dict(dropped) or 'none'}")

    s = col.stats
    n = s["rows"]
    print(f"\n[INFO] per row: {s['supervised_fields']/n:.1f} supervised field(s), "
          f"{s['supervised_tokens']/n:.1f} decision token(s) at w=1.0, "
          f"{s['evidence_tokens']/n:.1f} evidence token(s) at w={args.evidence_weight:g}")
    if s["supervised_tokens"]:
        ratio = s["evidence_tokens"] / s["supervised_tokens"]
        mass = args.evidence_weight * ratio
        print(f"[INFO] evidence:decision token ratio {ratio:.1f}:1 -> loss mass "
              f"{mass:.2f}:1 at w={args.evidence_weight:g} "
              f"(the evidence README sized this at ~23:1 -> ~0.9:1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
