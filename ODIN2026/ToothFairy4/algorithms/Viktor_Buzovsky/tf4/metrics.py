"""Reimplementation of the ODIN2026 offline metrics (copied from evaluation/evaluate.py
local-fallback paths, which is what runs on Grand Challenge since
RUNNING_ON_GRAND_CHALLENGE forces use_local_metrics()). Pure stdlib."""
import re, math
from collections import Counter


def tokenize(text):
    return [t for t in re.findall(r"\w+|[^\w\s]", text.lower()) if t.strip()]


# ---------- BLEU-4 (nltk corpus_bleu, weights .25x4, SmoothingFunction().method1) ----------
def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu4(predictions, references):
    """references: list[str] (single reference each)."""
    num = [0] * 5
    den = [0] * 5
    hyp_len = ref_len = 0
    for p, r in zip(predictions, references):
        pt, rt = tokenize(p), tokenize(r)
        hyp_len += len(pt)
        ref_len += len(rt)
        for n in range(1, 5):
            pn, rn = _ngrams(pt, n), _ngrams(rt, n)
            num[n] += sum(min(c, rn[g]) for g, c in pn.items())
            den[n] += max(0, len(pt) - n + 1)
    s = 0.0
    for n in range(1, 5):
        if den[n] == 0:
            return 0.0
        p_n = num[n] / den[n] if num[n] else 0.1 / den[n]  # method1 smoothing
        s += 0.25 * math.log(p_n)
    bp = 1.0 if hyp_len > ref_len else (math.exp(1 - ref_len / hyp_len) if hyp_len else 0.0)
    return bp * math.exp(s)


# ---------- METEOR-lite (verbatim from evaluate.py) ----------
def greedy_match_indices(pred, ref):
    used = [False] * len(ref)
    idx = []
    for tok in pred:
        for i, rt in enumerate(ref):
            if used[i] or tok != rt:
                continue
            used[i] = True
            idx.append(i)
            break
    return idx


def chunk_count(indices):
    if not indices:
        return 0
    ch = 1
    for cur, prev in zip(indices[1:], indices[:-1]):
        if cur != prev + 1:
            ch += 1
    return ch


def meteor_lite_score(pred, ref):
    m = greedy_match_indices(pred, ref)
    matches = len(m)
    if matches == 0:
        return 0.0
    precision = matches / len(pred)
    recall = matches / len(ref)
    den = recall + 9.0 * precision
    if den == 0.0:
        return 0.0
    f_mean = (10.0 * precision * recall) / den
    penalty = 0.5 * (chunk_count(m) / matches) ** 3
    return (1.0 - penalty) * f_mean


def meteor(predictions, references):
    if not predictions:
        return 0.0
    sc = []
    for p, r in zip(predictions, references):
        pt, rt = tokenize(p), tokenize(r)
        if not pt and not rt:
            sc.append(1.0)
        elif not pt or not rt:
            sc.append(0.0)
        else:
            sc.append(meteor_lite_score(pt, rt))
    return sum(sc) / len(sc)


def score(predictions, references):
    b = corpus_bleu4(predictions, references)
    m = meteor(predictions, references)
    return b, m, 0.2 * (b + m) / 2  # captioning contribution to final score
