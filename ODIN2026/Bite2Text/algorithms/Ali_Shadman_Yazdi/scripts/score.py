#!/usr/bin/env python
"""Score generated reports with the challenge's on-platform text metrics.

BLEU-4 (nltk ``corpus_bleu``, method1 smoothing) and METEOR-lite (greedy
exact-token matching, no WordNet), ported from the organisers' evaluation code
(github.com/AImageLab-zip/ToothFairy, ODIN2026/Bite2Text/evaluation/evaluate.py,
MIT License). On Grand Challenge these are the only live metrics; they make up
20% of the final score. RadFact (80%) needs an LLM judge and is not computed.

    python scripts/score.py \
        --prediction outputs/F5535.json \
        --reference  sample_data/F5535/reference_report_photo_en.txt

Several pairs can be passed; BLEU-4 is corpus-level, METEOR-lite is averaged.
A prediction may be a .json with a "report" key or a plain .txt file.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"\w+|[^\w\s]", text.lower()) if t.strip()]


def bleu_4(predictions: list, references: list) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    if not predictions:
        return 0.0
    return float(corpus_bleu(
        list_of_references=[[tokenize(r)] for r in references],
        hypotheses=[tokenize(p) for p in predictions],
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    ))


def _greedy_match_indices(pred: list, ref: list) -> list:
    used = [False] * len(ref)
    indices = []
    for token in pred:
        for i, ref_token in enumerate(ref):
            if not used[i] and token == ref_token:
                used[i] = True
                indices.append(i)
                break
    return indices


def _chunk_count(indices: list) -> int:
    if not indices:
        return 0
    return 1 + sum(1 for cur, prev in zip(indices[1:], indices[:-1]) if cur != prev + 1)


def meteor_lite_score(pred: list, ref: list) -> float:
    matched = _greedy_match_indices(pred, ref)
    matches = len(matched)
    if matches == 0:
        return 0.0
    precision, recall = matches / len(pred), matches / len(ref)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)
    penalty = 0.5 * (_chunk_count(matched) / matches) ** 3
    return float((1.0 - penalty) * f_mean)


def meteor_lite(predictions: list, references: list) -> float:
    scores = []
    for prediction, reference in zip(predictions, references):
        p, r = tokenize(prediction), tokenize(reference)
        if not p and not r:
            scores.append(1.0)
        elif not p or not r:
            scores.append(0.0)
        else:
            scores.append(meteor_lite_score(p, r))
    return float(sum(scores) / len(scores)) if scores else 0.0


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)["report"] if path.suffix.lower() == ".json" else text.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--reference", nargs="+", required=True, type=Path)
    args = ap.parse_args(argv)
    if len(args.prediction) != len(args.reference):
        ap.error("pass the same number of --prediction and --reference files")

    predictions = [read_text(p) for p in args.prediction]
    references = [read_text(r) for r in args.reference]
    bleu, meteor = bleu_4(predictions, references), meteor_lite(predictions, references)
    print(json.dumps({"n": len(predictions), "bleu_4": round(bleu, 4), "meteor_lite": round(meteor, 4),
                      "mean": round((bleu + meteor) / 2, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
