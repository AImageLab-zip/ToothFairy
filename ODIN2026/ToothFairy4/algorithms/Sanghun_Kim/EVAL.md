# Evaluation recipe

## 1. Generate

```bash
python -m toothfairy.cli.eval generate --run-dir models/sinus_rft_eval
```

Writes `predictions.csv` (example_id, prediction, target) and `per_region.jsonl` (the raw
slot outputs) into the run directory. `predictions.csv` is what everything downstream reads.

## 2. Rewrite into a narrative

The submitted model reports the rewritten text, not the bullets:

```bash
python -m toothfairy.cli.rewrite pairs-from-run --run-dir models/sinus_rft_eval --no-cls \
    --out experiments/analysis/rewrite/pairs_val.jsonl
python -m toothfairy.cli.rewrite apply --pairs experiments/analysis/rewrite/pairs_val.jsonl \
    --adapter models/rewrite_qlora_nocls/adapter_best \
    --out-dir experiments/analysis/rewrite/sinus_rft_eval
```

Note the two different pair builders: `pairs` builds *training* pairs from the reference slots,
`pairs-from-run` builds the pairs to rewrite from a run's own slot outputs. Feeding the first to
the rewriter at evaluation time means rewriting the ground truth.

## 3. Captioning metrics

```bash
python -m toothfairy.cli.eval captioning --csv models/sinus_rft_eval/predictions.csv
```

BLEU-4 and METEOR, no GPU.

## 4. Clinical metric

The challenge scores the clinical half with the organisers' own pipeline
([radfact_lite](https://github.com/AImageLab-zip/radfact_lite)), whose judge is a hosted
model. That pipeline is not vendored here; run it on `predictions.csv` and keep its
per-sample output.

Two warnings, both measured rather than assumed:

- **Run it with the normal-finding filter on**, as the organisers' usage does. The filter
  drops normal-status confirmations, tooth-presence statements and field-of-view sentences
  from *both* sides. On our generations it removes 78% of the phrases (48.2 -> 10.8). Scoring
  without it measures a different task.
- **A locally served judge is a development tool, not a substitute.** Swapping the hosted
  judge for a local `Qwen3-32B-AWQ` changes the ranking, not just the scale: on the same arms
  the two disagreed about which model was first. A local judge is fine for exploring, but a
  number meant to be compared with the challenge's must come from the challenge's pipeline.
  The judges differ in strictness mostly on recall (0.3806 -> 0.3132 for the hosted one, with
  precision nearly unchanged), so the gap is judgement, not a different phrase count.

## 5. Official score

```bash
python -m toothfairy.cli.eval score --csv models/sinus_rft_eval/predictions.csv \
    --radfact <radfact-output>/per_sample_scores.json
```

`Final = 0.8 x RadFact logical F1 + 0.2 x mean(BLEU-4, METEOR)`.

## Comparing two models

RadFact drops a case when the judge cannot quote its evidence verbatim, and the set of
dropped cases differs per model — 48 to 52 out of 62 in our runs. **Compare per-patient
scores over the cases both models scored**, not the two averages, or the comparison is
between different patients. `toothfairy.cli.eval score --common-with <other per_sample_scores.json>`
restricts to that intersection.

The same effect gets worse the more arms are compared at once: requiring five arms to have
scored the same patient left 28 in common and widened the interval to about ±0.05, enough to
hide a real difference. Put only the arms that need comparing into one table.
