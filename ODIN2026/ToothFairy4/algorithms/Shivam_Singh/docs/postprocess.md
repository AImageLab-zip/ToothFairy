# Post-processing and report synthesis

Everything downstream of the VLM: `normalize_pred.py` → `postprocess_pred.py` →
`source_rules.py` → `synthesize_report.py`. Per-fact JSON answers in, one
free-text radiology report out.

**Two parts, and they answer different questions.**

| | | |
|---|---|---|
| **1** | *What is applied* | the arm that ships, the rules it runs, what reaches the report, and how to run and verify it. Read this to know what the code does today. |
| **2** | *Why* | the defence. **2.1** is the current strategy — the per-source table every rule was chosen off, and what the rules changed. **2.2** is one question per decision, each with the measurement that settled it. Read this before changing anything. |

> **Validate-40 only.** Arm 6's inference on the 582-case training split **has
> not been run**, so every number here is 40 cases unless it says otherwise.
> Facts-and-reports-only studies (which need no inference) do cover all 622 and
> say so.

> **What is named here but not shipped.** This document is written against the
> research repo and cites its diagnostics by path. The public release ships the
> report-producing path, the official metric and the fact-level survey; it does
> not ship `code/eval/compare_sources.py`, `scripts/evaluation.sh`,
> `scripts/postprocess_val_now.sh`, `scripts/competition_sim.sh`,
> `code/ground_truth/audit_gt/report_gt_tables.py` or anything under
> `code/studies/`. Those names are provenance for a measurement, not
> instructions — every number below is stated here.

---

# 1. What is applied

## The arm — `aksssr_v7_trained_arm6`, Final **0.4658**

The current best, and what `default.yaml` describes.

| | |
|---|---|
| model | `models/Qwen3.5-9B-AWQ-dental-cbct-sft` (arm 6) — how it was trained is `docs/vision_sft_plan.md`; nothing below depends on it |
| schema | v7.1 |
| postprocess arm | `configs/postprocess/default.yaml` — all eleven source rules on, cross-source gates off |
| scored | validate-40, judge `qwen3-14b-text` via `judge_server.sh`, 2026-08-17 22:52 |
| run dir | `outputs/aksssr_v7_trained_arm6_validate/` |

| | arm 6 | arm 6, 07:12 |
|---|---|---|
| **Final Score** | **0.4658** | 0.4557 |
| Clinical (RadFact logical F1) | 0.5139 | 0.5014 |
| RadFact precision / recall | 0.5306 / 0.4982 | 0.5409 / 0.4674 |
| BLEU-4 (corpus) / METEOR | 0.1626 / 0.3848 | 0.1629 / 0.3830 |
| Captioning (mean of the two) | 0.2737 | 0.2730 |

**Both columns are the same model on byte-identical predictions.** Arm 6 was
inferred once. What moved between 07:12 and 22:52 is postprocess — the condyle,
maxilla-FOV and atrophy commits of that day — worth **+0.0101**, all of it
clinical. Quote 0.4658; `val_arm6/` no longer holds the reports it scored.

## The stages

```
predictions/{case}_pred.json
  │  normalize_pred.normalize_prediction()   repairs schema violations in memory
  ├─ postprocess_pred.py                     classify, group, reconcile across sources
  │    └─ source_rules.apply()               THE ELEVEN RULES, as a post-pass
  └─ summaries/{case}_summary.json
       └─ synthesize_report.py               deterministic templates, never an LLM
            └─ synthesized_reports/{case}.txt
```

**`--facts-dir` is what turns the rules on.** Without it every rule is a no-op,
the summaries are byte-identical to the pre-rules code, and nothing in the
output says so. `infer.py` warns.

## The eleven rules

Each moves one finding to the source that actually knows it, rather than voting
between every read that can assert it. All are switches in
`source_rules.RULES`, all default `True`, each can be turned off on its own from
a config file.

| rule | config key | what is applied | reaches the report |
|---|---|---|---|
| absent teeth | `absent_teeth` | `facts.teeth_absent` ∩ permanent FDIs, **minus the upper arch when `fov.maxilla == "excluded"`** | yes |
| impaction | `impaction` | re-derived per the schema's own `_definitions.impacted`: `eruption_state` from the composite (3D where no crop), `orientation` from the 3D fact. Wisdom slots only | yes |
| endodontic | `endodontic` | `tooth_{fdi}_morphology.with_endo` alone; the panoramic's `root_canal_treatment` is not read | yes, teeth only |
| fillings | `fillings` | `tooth_{fdi}_morphology.with_fillings` alone; the arch survey's aliased `restoration` is dropped | yes, gated on the rule |
| crown | `crown` | `facts.crowns` | **no** — renderer silenced |
| implants | `implants` | positions from `facts.implants`; `with_crown` / `osseointegration_status` carried over from the VLM entry at the same FDI | yes |
| canal-adjacent | `canal_adjacent` | `facts.ian_close_teeth`; neither read consulted | yes |
| canal location | `canal_location` | the composite's per-tooth value where 36–38 / 46–48 agree, else the prior `lingual` | yes |
| alveolar atrophy | `atrophy` | state atrophy where the mask makes the arch fully edentulous; otherwise silent. The model's own `atrophy` is not read | yes |
| fixed bridges | `bridges` | `facts.bridge_arches` (written by `audit_facts.py --derive-bridge-arches`), per arch. **No span is claimed** | yes |
| condyle scope | `condyle_fov` | `fov.condyles == "excluded"` → `not_included`; `fov.maxilla == "excluded"` implies it; else the merged read stands | yes |

**Two more findings ride the same `--facts-dir` flag and are not in `RULES`:**

- **maxilla FOV scope** (`maxilla_fov.*`) — `postprocess_pred.py`'s arch gate
  takes `maxilla_included` from `facts.fov.maxilla` and overwrites the model's
  answer **in both directions**: the arch is emptied when the mask has no
  maxilla and **kept** when it has one, whatever the render looked like.
- **condyle folding** (`CONDYLE_SCOPE_BINARY`) — `fully_/partially_included`
  fold to `included` and the two sides merge to one statement, *any side
  `not_included` wins*, before the rule above applies.

This is why **`NO_FACTS=1` and `no_source_rules.yaml` are different
ablations**: the environment variable drops `--facts-dir` and takes the maxilla
FOV override with it; the config keeps the facts file and turns off only the
eleven rules.

## The gate — what an absent position suppresses

A per-tooth finding asserted at an FDI in `absent(case)` is dropped, whatever
source claimed it. Nothing can be restored, root-filled, carious or
periodontally involved if it is not there.

| | |
|---|---|
| **suppressed** | caries/defect, fillings, crowns, post-and-core, endodontic treatment and filling quality, per-tooth periodontal status, per-tooth bone quality, canal adjacency |
| **exempt** (`GATE_EXEMPT`) | implants, bridge spans and pontics, root remnants, impacted/unerupted teeth, and every arch- or volume-level fact (atrophy, per-arch bone quality, maxilla scope, sinus findings, canal position) |

Each exemption is a real finding *at a position with no tooth* — an implant sits
exactly where a tooth is missing, a pontic spans an absent position by
definition. **The gate runs last**, after every rule that re-sources a per-tooth
finding; see *Which mistakes has this code already made?*

## What reaches the report

`synthesize_report.py` is **deterministic string templates, never an LLM**. A
template cannot invent a *new* clinical error on top of whatever the VQA already
got wrong, and that property is not given up.

**The silencing rule: a finding reaches the report only if it is measurably
better than silence.** Its operational form is a **≥0.35 precision bar** —
every finding in the report text scores at least 0.35 precision, or is an
accuracy measure at 0.46–0.86. Everything below the bar is summary-only.

Silenced, and staying that way: **root remnants, caries, periodontal
resorption, crown, post-and-core, bone quality, intrasinusal teeth, endodontic
filling quality, impaction angle, root fracture.**

Two rules about them, both deliberate:

- **Every silenced fact stays in the summary JSON.** Killing the fact would
  remove the instrument that would later show an improvement.
- **Each silenced renderer keeps its sentence verbatim in its docstring**, so
  re-enabling one is restoring a function body, not reconstructing a template.

Three template decisions are also load-bearing:

| | |
|---|---|
| `"No signs of bone atrophy."` | **removed entirely.** The negative occurs **zero** times across the 40 validate references, in any wording, while 24 of 40 mention atrophy positively. It could only ever have matched by accident |
| `"No definite osteolytic or osteocondensing lesions."` | **once per report, closing the mandible** — not once per arch. 64 copies in 40 reports became 39. It *is* a corpus form (the most frequent sentence in the training reports) but a whole-jaw negation, not an arch-scoped one |
| canal position | **always stated**: `"Right mandibular canal with a regular course, predominantly lingual."` — the corpus form verbatim. *Regular course* is the anomaly judgement and unconditional; *predominantly lingual* is the position and carries the prior |

## The arm is a config file

Every rule above, every cross-source gate and the FOV policy is a key in
`configs/postprocess/<arm>.yaml`. The code does not change between experiments;
the config does. `code/pipeline/postprocess/rules_config.py` is the binding
table that maps each key to the module constant it sets, and the only place that
mapping is written down.

| file | arm |
|---|---|
| `default.yaml` | **the main arm** — arm 6, Final 0.4658 / F1 0.5139 |
| `no_source_rules.yaml` | the eleven rules off, facts file still read |
| `cross_validate.yaml` | the pre-2026-08-07 precision-gate arm |

Every summary records what built it under `postprocess_config` — the config path
and only the keys that differ from `default.yaml`. `outputs/` is gitignored and
job logs rotate, so this is the only durable answer to "which arm produced this
file". `configs/postprocess/README.md` has the how-to; an unknown key is a load
error, not a warning.

## Running it

`official_ranking.py` runs on the login node, not via `sbatch` — `conda` is not
on `PATH` in a non-interactive shell, so the batch path fails its `nltk`
preflight, and compute nodes are not guaranteed outbound HTTPS.

```bash
ssh hpc
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cbct_base
cd ~/project_ToothFairy4

# predictions -> summaries -> reports -> survey. CPU only, seconds per case.
STAGE=post RUN_NAME=aksssr_v7_trained_arm6 scripts/run_infer.sh validate

# ablations
STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \
    scripts/run_infer.sh validate                # rules off, facts kept
NO_FACTS=1 STAGE=post RUN_NAME=<run> scripts/run_infer.sh validate  # rules AND facts off

# free, deterministic, and the right way to choose between arms
NO_RADFACT=1 OUT_DIR=<run_dir>/rank_norad_ \
  scripts/run_eval.sh <run_dir> validate

# the official number: persistent qwen3-14b judge on an A100
sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/judge_server.sh
OUT_DIR=<run_dir>/official_ranking_rules \
  scripts/run_eval.sh <run_dir> validate
scancel -n judge
```

## Verifying a change

There is no test suite. Verification here is three things, in this order:

1. **Rebuild and diff the survey against the previous one.** The post stage
   writes `survey/survey_facts_<stamp>.{txt,json}` without overwriting and diffs
   the previous stamp automatically. `outputs/` is gitignored, so `git diff`
   cannot show you what a change did to the output; the survey diff can. `PRED`
   cannot move — postprocess does not touch the raw reads — so any row whose
   `SUMMARY` column shifted is what the change did.
2. **`NO_RADFACT=1` scoring** — free, deterministic, and the right way to choose
   between arms.
3. **RadFact only for the final number**, and score twice when the expected
   difference is small. See *What should I not trust?*

---

# 2. Why

## 2.1 The current strategy

Two tables carry the whole argument. **Table 1** says which source knows what —
every rule is chosen off it. **Table 2** says what the rules did once applied.

Fact-level numbers come from `structured_findings_evaluation.py` /
`compare_sources.py` against the generated ground truth; report-level numbers
from `official_ranking.py` against the reference reports.

**The `facts` column is not circular, and that is what the rules rest on.**
`dataset/$SPLIT/facts/<case>.json` is a function of the segmentation mask —
`extract_facts.py` over the mask reproduces the upstream file byte for byte, and
no report was ever read to build it. So scoring it against report-derived ground
truth is a fair test, and the competition container can compute it from a CBCT
with no report in the room.

**Why an SFT arm is a fair place to measure this.** Arm 6's *reads* are its own;
a different model moves every VLM column. The `facts` column would not — it is
identical across arms by construction, and that invariance is the check that a
re-measurement is sound: **any future table where `facts` moves has a bug in it,
not a finding.** Every rule rests either on that column or on a comparison made
within one run.

### Two scoring rules the numbers depend on

Not caveats — the second one sets the absolute level of every number on the
absence axis, which is the largest row in the table.

**Recall is scored over what the source was shown.** The sources do not cover
the same positions — the 3D renders answer four wisdom slots, the composites are
expanded only for teeth the segmentation places — so an unrestricted denominator
reports a difference in *coverage* as a difference in *accuracy*. Precision is
never restricted: a claim is right or wrong wherever it was made.

**Absence is charged over all 32 positions**, the same standard the fact-level
survey uses, so the two tables state one number for that axis rather than two.
It is a harsh denominator: no reference report enumerates all 32 positions — the
generated GT answers a mean of **16.9 of 32** per case — so a call at a position
no report mentions is counted wrong rather than unscored. **103 of the mask's
358 absence calls** are of exactly that kind, which is the whole distance
between its 0.77 precision and its 0.55. The penalty scales with how much a
source claims, so it falls hardest on the mask, the one witness that enumerates
every position. `compare_sources.py --gt-answered-only` restricts both sides to the
positions the GT settles and prints the comparison without that penalty (facts
0.77/0.85, SUMMARY 0.83/0.85). **The ordering between sources is identical
either way** — only the absolute level moves — and the restriction was retired
on 2026-08-23 for that reason: one standard both tables share beats a fairer one
that only half the diagnostics use. Never applied to the finding axes in either
mode, where an arch read of `unremarkable` *is* an answer.

### The table

**Table 1 — per-source accuracy, validate-40, arm 6, rules on.** One row per
finding, one column per source: the mask-derived **facts**, the image reads
(**3d**, **panoramic**, **detail**), then `SUMMARY` after reconciliation. Every
cell is a straight read of `compare_sources.py`; nothing is derived by hand.
Cells are precision/recall unless the axis is an enum, where they are accuracy.

| axis | N | facts | 3d | panoramic | detail | SUMMARY | status |
|---|---|---|---|---|---|---|---|
| absent teeth | 232 | **0.55/0.85** ᵃ | 0.54/0.94 | 0.53/0.67 | — | **0.69/0.85** | on |
| impaction | 33 | — | 0.62/0.48 | 0.67/0.06 | **0.79/0.82** ᵇ | **0.78/0.85** | on |
| endodontic | 90 | — | — | 0.52/0.13 | **0.85/0.74** | **0.88/0.64** | on (teeth only) ᶜ |
| post_and_core | 17 | — | — | — | 0.08/0.09 | 0.06/0.12 | **off** |
| crown | 49 | **0.33/0.29** | — | — | 0.32/0.32 | **0.29/0.31** | **off** |
| fillings | 74 | — | — | — | 0.31/0.26 | **0.30/0.24** | on |
| caries | 26 | — | — | 0.00/0.00 | 0.14/0.05 | **0.14/0.04** | **off** |
| root remnants | 13 | — | — | — | 0.05/0.17 | **0.05/0.08** | **off** |
| implants | 32 | **0.68/0.53** | — | 0.44/0.12 | — | **0.68/0.53** | on |
| fixed bridges (cases) | 11 | **1.00/0.55** | — | 1.00/0.09 | — | **1.00/0.55** ᵈ | on |
| canal-adjacent | 26 | 0.69/0.69 | **0.70/0.27** | — | 0.17/0.32 | **0.69/0.69** | on |
| canal position (sides) | 63 | — | 0.84 | — | — | **0.86** ᵉ | on |
| maxilla scope (cases) | 40 | **0.85** | 0.82 | — | — | **0.88** ᶠ | on |
| atrophy (arches) | 80 | — | 0.76 | — | — | **0.78** | on |
| sinus mucosa (sides) | 80 | — | — | — | — | **0.85** ᵍ | on |
| sinus content (sides) | 80 | — | — | — | — | **0.95** ᵍ | on |
| sinus scope (sides) | 80 | — | — | — | — | **0.73** ᵍ | on |
| periodontal resorpt. (arches) | 80 | — | — | 0.90 | — | 0.90 | **off** |
| bone quality (arches) | 80 | — | — | 0.66 | — | 0.62 | **off** |

**status** = does this finding reach the report today. `off` means its renderer
returns empty, so read that row as diagnostic only: improving it changes nothing
until the renderer is turned back on.

**—** means the source is *structurally* silent, not that it claimed nothing:
v7.1's arch vocabulary cannot tell a crown from a large filling (both fold to
`restoration`), a mask knows nothing about root canals, and the composite is
never asked about implants. Each blank is one line in `SPEAKS` with its reason.

ᵃ The raw `teeth_absent` list. With the FOV gate it is **0.69/0.85** — the row
holds the raw number because the column is a source, not a rule. Precision on
this axis is held down by the reference reports' silence, not by the mask; see
*Two scoring rules* above and `--gt-answered-only` for the same row at
0.77/0.85 → 0.83/0.85.
ᵇ **Available, not consumed.** The composite carries no `impaction` fact, but it
carries `eruption_state`, which is half the schema's own definition. The only
derived cell in the table, and a floor.
ᶜ **teeth only** = the report names *which* teeth are treated and nothing about
*how*; `quality_groups` stays in the summary and never reaches the text.
ᵈ The rule is per **arch**, this row is per case.
ᵉ Validate is the friendly split here — see *THE RULE — canal location*.
ᶠ Not in `source_rules.py`; it fires in `postprocess_pred.py`'s arch gate.
ᵍ Read from the sinus crops, a source with no column here.

**Four readings, and they are what every rule below acts on:**

- **The mask wins wherever it can speak at all** — implants 0.68/0.53 against
  the read's 0.44/0.12, bridges 1.00/0.55 against 1.00/0.09, canal-adjacent
  0.69/0.69 against PRED's 0.22/0.46, maxilla scope 0.85 against 0.82. Its crown
  list at 0.33/0.29 is the one exception: level with the composite's 0.32/0.32.
- **The detail crop is the best read** on every axis it and another source both
  answer — endodontic 0.85/0.74 against the arch's 0.52/0.13, impaction
  0.79/0.82 against the 3D fact's 0.62/0.48. The arch survey earns its place
  through coverage, not accuracy.
- **PRED can be beaten by one of its own inputs.** On impaction the union scores
  0.62/0.48 while the composite alone scores 0.79/0.82 — the better source sits
  unused in the prediction file. That is the pattern the per-source split exists
  to expose.
- **Every row below 0.2 precision is `off` and no `on` row sits below 0.29** —
  the silencing rule still holding after a split it was not derived from.

### What the rules changed

**Table 2 — arm 6, same 40 cases, same predictions**, `compare_sources.py`'s
SUMMARY column, one run with `--facts-dir` and one without, so the two columns
differ by the rules and nothing else.

| axis | pre-rules | with the rules |
|---|---|---|
| absent teeth | 0.67/0.75 | **0.69/0.85** |
| impaction | 0.67/0.48 | **0.78/0.85** |
| endodontic | 0.75/0.66 | **0.88/0.64** |
| fillings | 0.21/0.35 | **0.30/0.24** |
| crown | 0.30/0.20 | **0.29/0.31** |
| implants | 0.44/0.12 | **0.68/0.53** |
| canal-adjacent | 0.67/0.46 | **0.69/0.69** |
| canal position | 0.83 | **0.86** |
| atrophy | 0.76 | **0.78** |
| fixed bridges | 1.00/0.09 | **1.00/0.55** |
| maxilla scope | 0.85 | **0.88** |

**Nine of the eleven are a clear gain, and the two that are not are known.**
`fillings` buys 0.09 precision with 0.11 recall — the one rule that removes
correct claims, and it is meant to. `crown` loses 0.01 precision for 0.11 recall
on an axis whose two sources are now level.

**Implants and bridges are where the rules do their real work**, both almost
entirely recall: 0.12 → 0.53 and 0.09 → 0.55. **Absence gains recall, not
precision** (0.75 → 0.85 at a flat 0.67/0.69) — the FOV-gated mask list reaching
positions no read answered.

**In the prose**, the same 40 predictions rendered twice through current code:

| | pre-rules | with the rules |
|---|---|---|
| sentences | 408 | **461** |
| normal / scope statements | 274 (67.2%) | **275 (59.7%)** |
| positive findings | 134 | **186** |

**More text, and a larger share of it says something.** The assert-nothing half
does not move — 274 against 275 — while positive findings go up by **52**.

**At report level, per sub-dataset** (official convention
`0.8 × Clinical + 0.2 × mean(cBLEU-4, METEOR)`):

| dataset | clinical | cBLEU-4 | METEOR | RF-P | RF-R | **Final** |
|---|---|---|---|---|---|---|
| A | 0.4784 | 0.0826 | 0.2554 | 0.5116 | 0.4492 | **0.4169** |
| F | 0.4043 | 0.2137 | 0.4832 | 0.2960 | 0.6376 | **0.3922** |
| P | 0.5659 | 0.1447 | 0.3771 | 0.6168 | 0.5227 | **0.5028** |
| S | 0.4948 | 0.2072 | 0.4234 | 0.6978 | 0.3833 | **0.4588** |
| **ALL** | 0.5139 | 0.1626 | 0.3848 | 0.5306 | 0.4982 | **0.4651** |

(`per_dataset_breakdown.py --corpus-bleu` recomputes cBLEU per group, so its ALL
lands at 0.4651 against the official 0.4658 — the BLEU denominator, not the arm.)

**P is the best prefix by a distance**, 0.5028 against A's 0.4169 — a reversal,
because P was the weakest prefix in every arm before the source rules. The rules
are mask-sourced and P's cases are where the reads were furthest off. **The four
prefixes fail in opposite directions**: S and P are precision-heavy and
recall-poor, F is the mirror at precision **0.32**, for the reason in *What
should I not trust?*

One artefact working against this run: `S0021` produced a report with **zero
parseable phrases**, so `official_ranking.py` forced its precision to 0.0. One
case in 40.

**Verification.** Without `--facts-dir` the summaries are byte-identical to what
the pre-rules code produced, re-checked on 2026-08-17 by rebuilding all 40
against the previous commit. And no per-tooth finding survives on a mask-absent
position across the 40 cases.

## 2.2 The questions

### THE RULE — absent teeth · why the mask, and why gate the upper arch?

**The mask dominates the raw prediction outright** — better on precision *and*
recall at once, so there is no operating point at which the read is preferable:

| source | claims | tp | fp | fn | prec | rec | F1 |
|---|---|---|---|---|---|---|---|
| **facts (mask)** | 255 | 197 | 58 | 35 | 0.77 | **0.85** | **0.81** |
| 3d (wisdom slots only) | 62 | 45 | 17 | 3 | 0.73 | 0.94 | 0.82 |
| panoramic (arch survey) | 207 | 159 | 48 | 57 | 0.77 | 0.74 | 0.75 |
| PRED (all reads) | 220 | 167 | 53 | 65 | 0.76 | 0.72 | 0.74 |

Absence is a segmentation question, and the mask is what every image was
rendered *from* — the reads are guessing at something `facts.json` already
knows. Unioning the panoramic back in buys 2 true positives for 6 false ones;
intersecting costs 40 true positives.

**The gate exists because a segmentation cannot tell *absent* from
*unassessable*.** When the mask carries no tooth at a position, either the tooth
is missing — a finding — or the position was never imaged, which is not. Left
ungated, `teeth_absent` asserts sixteen upper absences on a scan that stops at
the palate.

**What the reports do when the maxilla is out of the volume** (all 1000
references, `code/studies/survey_upper_mentions.py`, no GT involved):

| `fov.maxilla` | cases | any report names an upper tooth | any calls one absent |
|---|---|---|---|
| **`excluded`** | 354 | 53 (**15%**) | 18 (**5%**) |
| `partial` | 213 | 143 (67%) | 55 (26%) |
| no `maxilla` key | 55 | 49 (89%) | 32 (58%) |

A list asserting sixteen upper absences in those cases is not disagreeing with
the radiologist; it is answering a question the radiologist declined to answer.

**At fact level the gate is a wash, and that has to be said plainly** — shipped
F1 0.858 against ungated 0.860 over 622 cases, trading 56 true positives for 66
false ones on report-settled positions. **What the fact-level score cannot see
is why it stays**: of the 2026 upper positions the mask calls absent in excluded
cases, only 122 are ever settled by a report. The other **1904 land on positions
no reference discusses** — fact-level scoring drops them as unanswered, RadFact
charges for every sentence.

*Rejected:* gating `partial` too (recall 0.860 → 0.655 — partial means *some*
upper teeth are outside the volume, and the mask is right about the ones inside);
gating only where the mask carries no upper tooth (0.860, and **101 of 354**
excluded cases carry the whole upper arch, so the gate must key on `fov.maxilla`
rather than the shape of the list); the ≤4-upper-positions stratum (0.863, ahead
by 0.005 on 33 claims — noise, bought with a second condition).

*Gives up:* an edentulous excluded maxilla is unreportable. That is the right
trade — gating case 3, where all sixteen upper positions are absent and the
maxilla **is** imaged, costs F1 0.858 → 0.850 and deletes A018, whose reference
states complete maxillary edentulism outright.

*Scoring hazard, and it is the reason this rule is argued from report text
rather than from the GT:* `parse_reports_to_gt` fills unstated positions from
`presence_enumerated`, so **2072 of the 2491 GT-answered upper positions in
excluded cases — 83% — were never stated by any report.** Scored on
report-settled positions only, the mask's upper-absence precision is 0.46 under
exclusion against 0.82 elsewhere; the raw GT columns say 0.07 against 0.68. The
gap is real either way; its size is mostly fill-in.

### THE RULE — impaction · why re-derive what the model was told to derive?

Scored on the 66 wisdom slots where the GT commits (33 true / 33 false):

| strategy | tp | fp | fn | prec | rec | F1 |
|---|---|---|---|---|---|---|
| **C — composite `eruption_state` + 3D `orientation`** | 25 | 9 | 8 | **0.74** | **0.76** | **0.75** |
| E — the composite's `eruption_state` alone | 20 | 5 | 13 | 0.80 | 0.61 | 0.69 |
| B — the 3D fact's own two fields | 17 | 8 | 16 | 0.68 | 0.52 | 0.59 |
| A — the 3D fact's `impacted` bool — **what shipped** | 14 | 7 | 19 | 0.67 | 0.42 | 0.52 |

**F1 0.52 → 0.75 for +11 true positives against +2 false ones**, and two
separate effects worth keeping apart:

1. **The model does not reliably apply the rule it was given.** B beats A using
   *the same fact's own fields* — re-deriving `impacted` from the
   `eruption_state` and `orientation` the model wrote beats the `impacted` it
   wrote alongside them. Field order in the schema makes the derivation
   *possible*; it does not make it *right*.
2. **The composite is the better read of eruption** — binary accuracy 0.64
   against the 3D fact's 0.43, which is the partition the rule actually uses.

The 3D fact still supplies `orientation` (the composite has no such field) and
`eruption_state` for the **95 of 160** slots with no crop.

*Caveat:* C over E is 5 true positives against 4 false ones. Do not read that
ordering as settled.

### THE RULE — endodontic treatment · why drop the panoramic instead of unioning?

Scored over the same 90 GT positions:

| strategy | claims | tp | fp | prec | rec | F1 |
|---|---|---|---|---|---|---|
| **composite only** | 78 | 60 | 18 | **0.77** | 0.67 | **0.71** |
| composite where cropped, panoramic elsewhere | 78 | 60 | 18 | 0.77 | 0.67 | 0.71 |
| union — **what shipped** | 100 | 62 | 38 | 0.62 | 0.69 | 0.65 |
| both must agree | 24 | 22 | 2 | 0.92 | 0.24 | 0.39 |
| panoramic only | 46 | 24 | 22 | 0.52 | 0.27 | 0.35 |

**The union buys +2 true positives for +20 false ones** — marginal precision
0.09. The panoramic's headline 0.52 looks tolerable only because 22 of its 24
correct claims are ones the composite already made.

**The impaction-style fallback is a no-op here, and that is the interesting
part**: "composite where cropped, panoramic elsewhere" scores *identically* to
composite-only, because an endodontically treated tooth is present, therefore
segmented, therefore cropped. There is no coverage gap for the arch survey to
fill — it only contradicts a better-placed read at positions already answered.

*Gives up:* 2 true positives out of the report text. The only rule besides
fillings that removes correct claims.

### THE RULE — fillings · why the composite alone, and why is the renderer back on?

| | claims | tp | fp | prec | rec |
|---|---|---|---|---|---|
| composite `with_fillings` alone | 38 | 15 | 23 | **0.39** | 0.20 |
| + arch `restoration` folded in — **what shipped** | 161 | 27 | 134 | 0.17 | 0.36 |

**+12 true positives for +111 false ones**, and the fold is unsound by
construction: v7.1's arch vocabulary has no filling value *because* a crown and
a large filling are one bright capped tooth at that resolution, so `restoration`
is precisely the claim that cannot be resolved into a filling —
`ARCH_VALUE_ALIASES` resolves it anyway.

**Merging the composite's own restoration types is no better.** True positives
do not move — 15 in all three of `fillings`, `crown+fillings`,
`crown+fillings+post_and_core`. Of 74 GT fillings the composite calls 0 of them
"crown" and 0 "post_and_core": it either says filling (15) or says nothing (59).
**The dominant error is silence, not mislabelling**, and no relabelling reaches
it.

At 0.39 the composite clears the ≥0.35 bar where the shipped 0.17 does not — the
finding had been silenced as part of a renderer whose other two members are
genuinely below the bar, not on its own evidence. Turning it on meant splitting
`render_restoration_summary`, and the sentence is **gated on the summary
carrying `source_rules`**, because rendering it without the rule puts the 134
false claims back.

*Caveat:* the rule was sized when the composite read 0.39 and it reads **0.31**
on arm 6 — what it buys is about a third smaller than when it was chosen.

### THE RULE — crown · why source it from the mask and still say nothing?

`crown(case) = facts.structured.crowns`, at **0.33/0.29** against the
composite's 0.32/0.21 and 0.28/0.10 for what postprocess produced before.

**At 0.33 it is below the ≥0.35 floor, so the rule improves the summary JSON and
reaches no report text.** That is deliberate: `crown` is an input to the
bridge/implant reasoning, and the summary is the instrument that would later
show an improvement worth turning on.

> **The margin has since closed and the rule is not settled.** On arm 6 the
> composite reads **0.32/0.32** against the mask's unchanged 0.33/0.29 — level
> on precision, ahead on recall. 49 positions cannot separate two sources this
> close, and a larger split has reversed this axis before. `RULES["crown"] =
> False`, or a re-point at the composite, is a live option to be decided on the
> training split.

**Crowns are not one finding**, which is why a single row understates what is
knowable. The 49 GT positions split: **24 endo+crown**, 16 other, 5
implant+crown, 3 in a bridge (1 position counts twice). Endo+crown is half of
all crowns and sits on teeth the pipeline already identifies well — endodontic
is the composite's *best* axis. Note the denominator question: 11 implants carry
`with_crown`, but only 6 also appear in `with_full_crown`, because an implant
position usually has no `tooth_{fdi}_morphology` block at all.

### THE RULE — implants · why the mask's positions?

**0.68/0.53 against 0.00/0.00 for what shipped** — the largest single gap in the
table, and the only `on` row whose source was wrong every time it spoke.

**The VLM does not find implants. It finds their absence.** The quadrant counts
look like a working detector and are a base rate:

| quadrant implant count | right / scored | acc |
|---|---|---|
| overall | 127 / 146 | 0.87 |
| where the GT count is **0** | 125 / 129 | **0.97** |
| where the GT count is **> 0** | 2 / 17 | **0.12** |

**And the wrong positions are not near-misses**: of 18 false positives, 14 sit
more than two positions from the nearest true implant. This is not FDI drift a
neighbour-tolerant match would forgive — the model puts implants in the wrong
region, which is why the old drop-only gate was measured *harmful* (F1 0.474 →
0.372). Dropping removes the 18 wrong positions and cannot supply the 32 true
ones the mask already has.

`facts.implants` is a bare FDI list, so `with_crown` and the osseointegration
clause stay with the VLM and ride along where its `fdi_number` matches.

*Caveat:* 0.68 is flat per-FDI matching. Reports place implants by region as
often as by tooth, and slot semantics would score no lower.

### THE RULE — canal-adjacent teeth · why neither read?

| source | claims | tp | fp | fn | prec | rec |
|---|---|---|---|---|---|---|
| **facts `ian_close_teeth`** | 26 | 18 | 8 | 8 | **0.69** | **0.69** |
| 3D canal trace | 20 | 8 | 12 | 18 | 0.40 | 0.31 |
| composite `tooth_{fdi}_mandible_canal` | 51 | 9 | 42 | 16 | 0.18 | 0.36 |
| both reads unioned | 66 | 14 | 52 | 12 | 0.21 | 0.54 |
| after the current vote — **what shipped** | 24 | 14 | 10 | 12 | 0.58 | 0.54 |

The mask claims 26 and gets 18 right; the two reads together claim 66 and get 14
right. **This is the one axis measured on the same quantity the model is asked
to eyeball** — proximity of a root apex to the canal is a distance between two
segmented structures, so the mask computes what the VLM estimates by eye.

**The composite is the weakest source here and the strongest almost everywhere
else**: 51 claims for 9 hits, the lowest precision of any source on any axis in
this document. Asked tooth by tooth "is this root near the canal", it says yes
far too readily.

Sourcing also subsumes the old drop-gate entirely — a list sourced from
`ian_close_teeth` cannot contain anything `ian_close_teeth` would drop — and
adds the 8 teeth the mask names that both reads missed.

### THE RULE — fixed bridges · why per arch, and why no span?

Precision **1.00 / recall 0.44 per arch** — 7 arches named, 0 wrong — against
0.67/0.18 per *case* for what shipped, whose facts-side input was a bare
`bridge_present` bool for the whole mouth.

**The ceiling is the segmentation, not the derivation.** Five of the eleven
bridge cases carry no label 8 at all, so nothing downstream can recover them.

**Why no span, having tried.** Label 8 marks the **pontic**, not the abutments,
so the component's extent is the middle of the bridge and never its ends.
Closing it by dilation lands within about one position and not reliably on it:
F067's pontics touch 14 and 16 where the report says *"from 1.4 to 1.7"*;
P014's touch 32 and 42 against *"from 3.3 to 4.3"*. The error is ±1 with no
consistent sign. Carrying the VLM's span instead is worse — **0 of 16** exact
against the reference, only **2 of 16** even overlapped. The sentence is
`"Prosthetic bridge exists."` and the arch is never in doubt.

**A rule is a SOURCE, not a gate — this is the axis that proves it.** Bridges
was written as a gate first: dropping the VLM's wrong-arch bridges left **4 of
the 6** mask-confirmed cases reporting no bridge at all. Gating protects
precision by deleting the finding.

*On training:* run over all 582 facts files, the rule scores **0.81/0.79 per
case** against the panoramic's 0.73/0.20. The 1.00 precision validate showed was
7 arches and did not survive; the 4× recall did, and that is the result that
matters.

### THE RULE — alveolar bone atrophy · why ignore the model's own answer?

| strategy | claims | tp | fp | prec | rec | F1 |
|---|---|---|---|---|---|---|
| **mask says fully edentulous → state atrophy** | 5 | 5 | **0** | **1.00** | **0.23** | **0.37** |
| gate the model's `atrophy` on mask edentulism | 2 | 2 | 0 | 1.00 | 0.09 | 0.17 |
| gate it on its own `fully_edentulous` | 3 | 2 | 1 | 0.67 | 0.09 | 0.16 |
| the model's `atrophy`, ungated — **what shipped** | 5 | 2 | 3 | 0.40 | 0.09 | 0.15 |

**Dropping the model's judgement entirely is what makes it work.** Gating its
`atrophy` keeps only 2 of the 5 arches the prior gets right, because it declines
to call atrophy on 3 edentulous jaws that have it.

**Why an edentulous jaw is enough on its own:** the alveolar process exists to
hold teeth and resorbs once they are gone, so "no teeth in this jaw" implies
bone loss as anatomy, not as image reading. 5 fully edentulous arches by the
mask, 5 with atrophy in the reference, **0 false positives**.

*Gives up:* recall is capped at 0.23, because 17 of the 22 atrophic arches are
only **partially** edentulous. That is not a limitation to fix later by tuning —
on partially edentulous arches the model scores **0 of 19**, so the alternative
to silence is noise. Bone height in a span is a genuine imaging judgement, and
this is the one axis where the source that ought to answer it cannot.

`present` is deliberately **not** a rule: it is a precondition ("at least one
stretch of 3+ missing teeth"), never stated in a report, and pure arithmetic
over the absence list — 1.00/0.66 derived against the model's 1.00/0.16. If it
is ever needed it should be computed, never asked.

### THE RULE — canal location · why a lingual prior?

| strategy | accuracy over 63 sides |
|---|---|
| **composite unanimous, else lingual — THE RULE** | **0.87** |
| constant: always lingual | 0.87 |
| composite majority, 3D as fallback | 0.86 |
| the 3D canal fact alone | 0.83 |
| **3D, else lingual — what shipped** | **0.83** |

**This ties the constant on validate and is chosen anyway**, because it *can*
move and a constant cannot: where the composite agrees on buccal, the rule says
buccal. **No strategy recovers a single buccal side** here — 0 of 8.

**The prior is set from training, and validate is the friendlier split:**

| split | sides | lingual | buccal | always-lingual |
|---|---|---|---|---|
| **training** | 794 | 640 | **154** | **0.806** |
| validate | 63 | 55 | 8 | 0.873 |

So the expected accuracy of the fallback is about **0.81**, not 0.87, and the
+0.04 over shipping is provisional. Deriving the prior from validate would be
tuning on the eval split. Buccal is 19% of training sides against 13% here —
both why the margin may shrink and why a rule that *can* answer buccal is worth
keeping over one that cannot.

*Open:* the composite's per-tooth `location` has never been scored against the
794 training sides. If it still recovers no buccal side over 154 of them, the
rule collapses to the constant and `location` should leave the schema.

### THE RULE — maxilla FOV scope · why does the facts file overrule the model?

```
fov.maxilla == "excluded"  -> not_included : null the maxilla facts, drop 11-28
fov.maxilla == "partial"   -> included     : keep the whole arch
no maxilla key in fov      -> included     : neither FOV condition fired
no fov block / no facts    -> the model's answer decides, as before
```

"Is the maxilla in this volume" is a property of the **acquisition**, not a
clinical judgement, and the mask answers it directly. The VLM answers the same
question off one 3D render, and the failure is asymmetric — a render can look
bare when the mask is not.

| source | acc | prec | rec |
|---|---|---|---|
| **facts `fov.maxilla`** | **0.875** | 0.714 | 0.909 |
| pred arm 6 | 0.825 | 0.667 | 0.727 |
| pred arm 5 / AWQ base | 0.850 | 0.667 | 0.909 |

**The gate — which is what actually deletes an arch — goes 0.850 → 0.875 on all
three model arms, each moving exactly one case, every time in the new gate's
favour.** It is a *different* case per arm (A018, S0037, S0017), so this is not
an A018 quirk: every arm invents one `not_included` on a maxilla the mask can
see, and no two arms invent the same one.

The case it was written for: A018's arch was being deleted on the model's word,
taking a four-sentence reference paragraph — complete edentulism, marked
atrophy, normal sinuses — with it. It also recovered the only two correct sinus
reads postprocess was discarding.

> **This is the one rule whose training evidence disagrees.** On training-550 it
> is a **12–12 tie** (facts 0.789, arm-5 predictions 0.789), trading recall
> 0.904 → 0.860 for precision 0.731 → 0.749 — and those predictions are *in
> sample*, the model at its most flattered. The 24 arches it un-deletes there
> emit 160 upper absence claims of which **6** are true; the reference asserts no
> upper absence in 23 of 24. That failure belongs to the absence rule, not this
> one — the old scope gate happened to mute them by deleting the arch for an
> unrelated reason. A lucky mute, not a filter.

*Rejected:* claiming upper absences under `partial` only when the mask holds no
upper tooth (validate 0.69/0.85 → 0.69/0.67, training F1 0.703 → 0.693); sparing
the `excluded` bucket when the mask still holds ≥K upper teeth (training
accuracy falls monotonically from 0.789 to 0.631 at K=1 — the tooth count
carries no separating signal).

### THE RULE — condyle scope · why binary, and why one sentence for both sides?

**Why binary:** the three-way read carries no signal. Its per-class precision
equals the class prior to three decimals — `not_included` 42/70 = 0.600 against
a 0.600 base rate. And `fully_included` is not a real state in this corpus: 3 of
2734 GT sides, never once in validate-40 or training-582, while the model
emitted it 6 times on training and all 6 were wrong.

**Why merged:** 971 of the 978 reference reports that mention the condyles use
no laterality word, only 3 in 1000 assert a different scope per side, and every
case-level consensus GT is symmetric — while arm 6 contradicts itself between
its two renders in 10 of 40 validate cases.

| variant | validate | heldout | training |
|---|---|---|---|
| pre-rule, per side, no merge | 0.575 | 0.568 | 0.716 |
| merged reads alone (no facts) | 0.600 | 0.591 | 0.716 |
| facts alone, silence read as `included` | 0.525 | 0.545 | 0.699 |
| **THIS RULE** — facts where they speak, else merged | **0.600** | **0.636** | **0.720** |
| *constant `not_included`* | *0.600* | *0.636* | *0.722* |

**The mask is not simply "more accurate".** Read as a complete answer it is the
*worst* variant, because it only ever writes `excluded` — its silence is not
evidence of inclusion, and the key is absent in 7 of 40 validate cases. Used
only where it claims something, it is the best source on the axis: on validate
it speaks for 66 of 80 sides and the reads keep the other 14.

*What none of it buys:* every winning variant lands on `not_included` for all 40
validate cases. 0.600 **is** the majority-class score, all 32 `included` sides
are missed, and on training the constant still edges the rule. This buys a
coherent source hierarchy and a bilaterally consistent statement, not signal.

### Why do periodontal resorption and bone quality stay silenced?

They print 0.90 and 0.66 in Table 1 — the two healthiest-looking `off` rows and
the obvious candidates to switch back on. **They are the always-normal constant
wearing an accuracy score.** Scored on the abnormal class, which is the only
class a report sentence would ever be built from:

| axis (80 arch slots) | claims abnormal | right | prec | GT abnormal | found | acc | the constant |
|---|---|---|---|---|---|---|---|
| periodontal resorption | 2 | 0 | **0.00** | 6 | 0 | 0.90 | always-`none` = **0.925** |
| bone quality | 2 | 1 | 0.50 | 27 | 1 | 0.66 | always-`false` = **0.662** |

**Periodontal resorption is the only row in the table below its own trivial
constant** — answering "no resorption" without opening the image would beat it.
Bone quality reaches the constant to two decimals by answering `false` on 78 of
80 arches. For periodontal resorption, **silence is literally more accurate than
the read.**

**Training does not fix it and it is not a class-balance artefact.** Both fields
were supervised in arm 6 (558 slots each) and bone quality arrived close to
balanced. A field that is balanced in training and still answered `false` 78
times in 80 at inference is most likely **not legible on the panoramic at this
resolution**. What arm 6 changed is the operating point: 4 abnormal claims where
the baseline made 11, accuracy rising purely by moving closer to the constant.

Turning either renderer back on needs a source that does not exist yet, not a
threshold change.

### Why are caries and root remnants silenced?

| | GT | prec / rec |
|---|---|---|
| caries, panoramic `defect` | 26 | 0.00/0.00 |
| caries, composite `with_caries` | 26 | 0.14/0.05 |
| root remnants, composite `is_remnant` | 13 | 0.05/0.17 |

**Root remnants are the lowest-precision row in the whole table**, and the
silencing measurement reached it from the other direction: 107 teeth claimed
across 35 of 40 cases where the references mention a remnant in **10**, and **82
of the 107 claims sit in cases whose reports never raise the subject**. It was
the largest sentence class in the report — 107 of 681 sentences — each carrying
a second claim in its resorption clause.

**Caries has no source left that speaks.** With caries gone
`render_morphology_findings` emits nothing at all. Neither axis has a mask
source: a carious lesion is missing mineral rather than a label, and a fragment
below `TOOTH_MIN_MM3` is dropped rather than recorded as a remnant.

*On the remnant recall gap:* it is coverage, not accuracy. The composite is
expanded only for teeth the segmentation places, and a remnant is exactly the
case where it often does not.

### Why are the sinus rows `on` if the model never claims an abnormality?

Because what they emit is the **normal** sentence, so they add no false claims —
unlike the two silenced rows above, which would add sentences.

| axis | answered | claims abnormal | GT+ | acc | the constant |
|---|---|---|---|---|---|
| mucosa (`thickening`) | 40 | **0** | 6 | 0.85 | always-normal **0.842** |
| content (`fluid`/`mixed`) | 40 | **0** | 2 | 0.95 | always-air **0.947** |

Arm 6 claims the abnormal class **zero times** on either axis, so 0.85 and 0.95
are those constants exactly; 6 thickenings and 2 fluid findings are missed in
full. Training the arch calls changed nothing here.

**`sinus scope` is not a scorable axis at all.** Its GT is `partially_included`
on **80 of 80** sides — and that is not an observation, it is `_resolve_scope`'s
fallback. Arm 6 scores 1.00 by settling on the constant the GT happens to emit.
Worth doing (a model that settles on the *wrong* constant emits 38 wrong
sentences) but it demonstrates no ability to read sinus extent. The axis needs a
ground truth before it needs a rule.

**The sinus is not mentioned where `fov.maxilla == "excluded"`** — across the 14
excluded cases the pipeline answers 0 sinus sides, because no maxillary bone
means no segmented sinus, hence no crop. The references state 28 sinus sides in
those same cases, so this is a real recall cost knowingly accepted, on the same
principle as the absence gate.

### Why is atrophy the one arch-level read that is kept?

It is the counter-example to the four rows above, and the clearest thing
training the arch calls bought:

| run | field | claims | right | prec | GT+ | rec | acc | always-negative |
|---|---|---|---|---|---|---|---|---|
| baseline (AWQ) | `present` | 30 | 21 | 0.70 | 32 | 0.66 | 0.75 | 0.600 |
| **arm 6** | `present` | 23 | 19 | **0.83** | 32 | 0.59 | **0.79** | |
| baseline (AWQ) | `atrophy` | 17 | 11 | 0.65 | 20 | 0.55 | 0.77 | 0.725 |
| **arm 6** | `atrophy` | 23 | 13 | 0.57 | 22 | **0.59** | 0.76 | |

Arm 6 beats the always-negative constant on both fields and beats the untrained
baseline's precision on `present` while claiming fewer times. Note this is the
*read* earning its row; the **rule** above still states atrophy only from mask
edentulism, because that is where precision is 1.00.

### Why is a rule a SOURCE and not a gate?

The old design used the facts file **only to drop** claims it contradicted,
justified by a crown list at 0.45 precision / 0.24 recall — trustworthy for
"there is nothing there", untrustworthy for "there is something there".

**Eight of the eleven rules break that asymmetry, because it does not hold on
their axes**: absence 0.55/0.85, implants 0.68/0.53, bridges 1.00/0.55,
canal-adjacency 0.69/0.69, maxilla scope 0.85 accuracy. Where the mask is the
better witness in **both** directions it may *add* what the reads missed, not
only veto what they invented.

The asymmetry break is axis-by-axis and nothing licenses trusting the facts file
in general — `THE RULE — crown` keeps a field silenced precisely because 0.33
does not clear the bar.

**Three of the eleven are not source swaps at all** — canal location, atrophy
and the arch half of fillings replace a read with a constant or a precondition,
because the read carries no signal. That is a different finding from "the mask
is better", and the one most likely to generalise to axes not yet examined.

### Why is the report a template and not an LLM?

A template cannot introduce a *new* clinical error on top of whatever the VQA
already got wrong. It is a project decision, not an implementation detail, and
the LLM writer (`generate_report_from_pred.py`) is kept only for the
baseline/comparison arms. Do not make `synthesize_report.py` call a model.

### Which mistakes has this code already made?

Four, each of which cost a debugging session. Nothing in the code says them.

**A rule that re-sources a field must be able to write it where the old source
was silent.** Guarding on `if key in block` was written three separate times and
was wrong all three: **55 of 80 arch blocks carry no `impacted_teeth` key at
all**, so the guard made the rule able to *remove* an impaction and never to add
one. The same trap cost **11 of 25** mask implants — recall 0.31 against the
0.53 the rule was chosen on.

**The gate must run last.** `_gate_on_absent` sat inside `apply_absent` first —
before the rules that re-source per-tooth findings — so a filling the gate had
just dropped at an absent position was written straight back, and A019's maxilla
reported *"Absence of 14, 15, …"* and *"composite restoration(s) on teeth 15 and
27"* in the same paragraph. **A gate is only a gate if nothing writes after it.**

**The prompt's words are not the code's tokens.** v7.1's arch vocabulary says
`restoration`/`defect`/`unremarkable`; `ARCH_VALUE_ALIASES` folds those to
`filling`/`caries`/`normal` on the way in, so ~20 behaviours keyed on the old
strings never had to change and stored v6.x predictions still parse. Change the
schema's wording freely; **add an alias rather than renaming those tokens.**

**Field ORDER in `object_fields` is load-bearing upstream.** vLLM decodes with
`guided_json`, which follows that order, so a field derived from another
(`impacted` from `eruption_state` + `orientation`) must come after them or the
model answers the conclusion first.

*And two instrument bugs that capped rows at zero regardless of what the
pipeline did:* `survey_facts._hashable` keyed implants on `fdi`/`tooth`/
`position` — none of which is the schema's `fdi_number` — so an implant failed
to match itself whenever `location` was worded differently; and
`survey_findings.composite_coverage()` keyed on a fact v6.4 retired, so it
reported *no* tooth as having been cropped.

### What did the earlier arms score, and why can't I compare them?

| arm | Final | what it was |
|---|---|---|
| **`aksssr_v7_trained_arm6`** | **0.4658** | current best, this document |
| `aksssr_v7_trained_arm6`, 07:12 | 0.4557 | same predictions, postprocess one day older |
| `aksssr_v7_trained` (arm 5) | 0.4115 | the previous checkpoint, source rules on |
| arm 5, pre-rules | 0.3524 | the same arm without `--facts-dir` |
| `aksssr_v6` | 0.3307 | schema v6.9 + validate facts |
| everything before | ≤0.3193 | **scored under `gpt-4o`** |

**The first five are directly comparable** — all validate-40 under the same
local `qwen3-14b-text` judge. **The last row is not comparable to any of them**,
different judge entirely.

**The whole gain is clinical, not fluency**: RadFact F1 0.4470 → 0.5139 from arm
5 to arm 6, while captioning moved 0.2697 → 0.2737, which is noise.

**What the rules are worth at report level on arm 6 has never been measured** —
only at fact level (Table 2). That needs one rules-off postprocess
(`POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml`) and one
`scripts/run_eval.sh` against the same judge, the same evening.

### What should I not trust?

**The RadFact judge is not deterministic.** On 28 byte-identical reports scored
twice, BLEU-4 matched exactly on all 28 while `clinical_score` differed on
**24** — per-case sd 0.0625, max |Δ| 0.2136, at `temperature=0`. That is roughly
**±0.02** around any 40-case clinical score, the same magnitude as most arm
differences. Choose between arms with `NO_RADFACT=1` and the survey; spend
RadFact on the final number, and score twice when the difference is small.

**Every number here is validate-40**, so no rule is confirmed at scale, and two
of them (`crown`, impaction C-vs-E) rest on margins inside the noise of 40
cases.

**The template overshoots on F**, and it is the only prefix that does:

| | A | F | P | S |
|---|---|---|---|---|
| generated sentences per case | 13.4 | 13.0 | 9.1 | 10.6 |
| reference | 13.6 | 7.5 | 8.2 | 9.2 |
| overshoot | 0.99× | **1.73×** | 1.11× | 1.15× |
| RadFact precision | 0.5322 | **0.3182** | 0.6213 | 0.6918 |

Per-prefix length calibration is an F-only fix and needs no new information —
the pipeline already knows the prefix.

**The validate split now costs the score rather than flattering it.** Validate is
balanced 10/10/10/10; training is **69% P**. Weighting arm 6's per-prefix means
by that mix gives **0.4422** against the balanced 0.4099 — a *bonus* of 0.032,
where every earlier arm took a penalty. The premise inverted with the result: P
was the weakest prefix before the source rules and is now the strongest.

**Roughly 160 of the 275 normal sentences would be emitted identically with no
VLM at all** — canal course (80), scope statements (156 total), the osteolytic
negation (39), sinus aeration (20). Whether to cut them is a question about
n-gram and RadFact credit for correct negatives, not about accuracy.

**A mask-only baseline was not far behind** — `outputs/no_VLM`, template reports
from masks and facts with no VLM, scored 0.2761 under `gpt-4o` against the v6.4
arm's 0.2880. Different arm, different judge, different dates, so the arithmetic
does not transfer — but the point survives, and the source rules sharpen it,
since seven of the eleven now take the finding from the mask. **Worth a same-day
paired run under the current judge before claiming the VLM earns its cost.**

### What don't we know yet?

**Item 1: run arm 6 on the training split.** Everything else is downstream of
it. One inference job; the CPU half is seconds. What it would settle, in the
order the margins deserve:

| | validate says | what training would decide |
|---|---|---|
| **crown** | mask 0.33/0.29 vs composite 0.32/0.32 — level | whether `RULES["crown"]` points at the composite or is turned off |
| **fillings** | composite 0.31, down from the 0.39 the rule was sized on | whether the rule still buys what it costs |
| **canal location** | 3D 0.84, constant 0.873, rule 0.86 | whether `location` should leave the schema |
| **impaction C vs E** | 5 true positives against 4 false, on 66 slots | the closest call in the set |
| **the abnormal-class rows** | 80 arch slots each | atrophy, the three sinus rows, periodontal, bone quality |

**Item 2: score a pre-rules arm 6 at report level** — see *What did the earlier
arms score*.

**Item 3: `THE RULE — maxilla FOV scope` is the one rule whose two splits point
in different directions.** Its own section has the numbers.

**Nothing is outstanding in code.** All eleven rules are implemented, the
fillings renderer is split and gated, and the maxilla FOV gate reads the facts.

### Where do the artifacts live?

Arm 6 lives in **two** directories and only one of them is the number. The
predictions are byte-identical in both; what differs is the postprocess that
read them.

| what | path |
|---|---|
| **final summaries, reports and the 0.4658 ranking** | `outputs/aksssr_v7_trained_arm6_validate/` |
| superseded ranking — 0.4557 | `outputs/training_results/vsft_arm6/val_arm6/official_ranking/` |
| validate-40 predictions (identical in both) | `vsft_arm6/val_arm6/predictions/` |
| adapter, checkpoints, `train_meta.json` | `vsft_arm6/weights/` |
| source comparison (Tables 1 and 2) | `aksssr_v7_trained_arm6_validate/survey/source_compare_20260823_150111.{txt,json}`, and the pre-rules column from `outputs/aksssr_v7_trained_arm6_validate_prerules/` (same predictions, `NO_SOURCE_RULES=1`) |
| fact-level survey | `val_arm6/survey/structured_findings_evaluation_20260817_063256.{txt,json}` |
| held-out 22 | `outputs/training_results/ho_awq_arm6/` |
| the merged servable checkpoint | `models/Qwen3.5-9B-AWQ-dental-cbct-sft` (promoted by rename) |

**`val_arm6` is not a name the split can be read off** — it matches neither
`*validate*` nor `*train*`. `scripts/run_infer.sh` takes the split as its
positional argument and `OUT_DIR` for the directory, so state both rather than
relying on the name.

**Not present, and every "training says" gap above points at it:** there is no
`train_arm6/`. `outputs/` is gitignored (commit `3f8d256`), so all of this lives
on the cluster only — this document and the code are the record git carries.

### What changed when?

| date | what |
|---|---|
| **2026-08-15/16** | `compare_sources.py` written; the per-source work begins. `audit_facts.py --derive-bridge-arches` implemented and run |
| **2026-08-16** | all ten source rules implemented in `source_rules.py`; the fillings renderer split out of `render_restoration_summary` and gated on `source_rules` |
| **2026-08-16** | facts provenance re-measured — the pool is `extract_facts.py` over the upstream masks, byte-identical on re-run |
| **2026-08-17 07:12** | official ranking of **arm 6** → **0.4557**, the new best |
| **2026-08-17** | the abnormal-class view added for the six arch/side rows: periodontal and bone quality are at or below their trivial constants, atrophy is the opposite, sinus scope measures nothing |
| **2026-08-17** | **THE RULE — maxilla FOV scope** and **THE RULE — condyle scope** added; the absent-teeth FOV gate re-measured on all 622 cases and kept on report evidence rather than fact-level score |
| **2026-08-17 22:52** | **arm 6 re-scored after the day's postprocess work → 0.4658**, the final number. Same predictions; +0.0101, all clinical |
| **2026-08-19** | `survey_findings.py` deleted; its hand-read tables survive in `code/ground_truth/audit_gt/report_gt_tables.py` |
| **2026-08-21** | **the arm became a config file** — `rules_config.py` + `configs/postprocess/`, replacing edited module constants |
| **2026-08-22** | this document restructured into *What is applied* / *Why*. The prior version — the full per-rule narrative, the 2026-08-07 v6.4 baseline half, and the superseded arms' tables — is in `git log --follow -- docs/postprocess.md` |
