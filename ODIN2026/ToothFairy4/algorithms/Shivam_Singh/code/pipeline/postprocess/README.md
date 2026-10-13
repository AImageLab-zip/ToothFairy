# `code/pipeline/postprocess/` — predictions → summary → report

Everything downstream of the VLM. Per-fact JSON answers in, one free-text
radiology report out.

## The files

|                        |                                                              |
| ---------------------- | ------------------------------------------------------------ |
| `normalize_pred.py`    | repairs schema-violating VLM output against `schema/schema.json` — wrong enum values, scalars where a list belongs, near-miss key names |
| `postprocess_pred.py`  | `{case}_pred.json` → `{case}_summary.json`. Classification, grouping and cross-source reconciliation |
| `source_rules.py`      | the eleven source rules, applied as a post-pass over the finished summary |
| `synthesize_report.py` | `{case}_summary.json` → `{case}.txt`. **Deterministic string templates** |
| `rules_config.py`      | the binding table: which config key sets which module constant. Not logic — plumbing |

**Which rules and gates run is a sixth file, and it is not in this directory.**
It is `configs/postprocess/<arm>.yaml`; `rules_config.py` binds each key to the
module constant it sets. If the postprocess rules change, only the configs
change; the code stays put.

```bash
# check an arm before spending a run on it -- '*' marks every non-default key
python3 code/pipeline/postprocess/postprocess_pred.py --print-config \
    --config configs/postprocess/my_arm.yaml --pred-dir x --out-dir y
```

## Running it

Normally you don't call these directly — `scripts/run_infer.sh` with
`STAGE=post` drives all three in the right order, which is where the mistakes
otherwise happen (surveying summaries that predate the change being measured, or
reading a report rendered from a stale summary).

```bash
# the CPU tuning loop: predictions -> summaries -> reports -> survey.
# No GPU, no model, seconds per case. Runs on the login node, no queue.
STAGE=post RUN_NAME=aksssr_v7_trained_arm6 scripts/run_infer.sh validate

# a different arm
STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \
    scripts/run_infer.sh validate

# by hand, if you must -- PASS THE SAME CONFIG TO BOTH
python3 code/pipeline/postprocess/postprocess_pred.py \
    --config configs/postprocess/default.yaml \
    --pred-dir <run>/predictions --out-dir <run>/summaries \
    --facts-dir dataset/validate/facts --schema schema/schema.json

python3 code/pipeline/postprocess/synthesize_report.py \
    --config configs/postprocess/default.yaml \
    --summary-dir <run>/summaries --out-dir <run>/synthesized_reports
```

`docs/postprocess.md` is the evidence — every number, every measurement, every
rule that was tried and rejected, what reaches the report and why, how to verify
a change, and the traps this code has already fallen into.
