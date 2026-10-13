# `configs/postprocess/` — the arm is a file

Post-processing in this pipeline is a set of **rules**: which source a finding
is taken from, which cross-source disagreements are gated, what happens to an
arch the acquisition cut off, and what reaches the report text. Those rules
change between experiments. The code that applies them does not.

So the rules live here, one YAML file per arm, and
`code/pipeline/postprocess/` stays fixed. Running an experiment means writing a
config, not editing a module.

```bash
# the main arm — what infer.py uses when nothing is named
STAGE=post scripts/run_infer.sh validate

# an ablation
STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \
    scripts/run_infer.sh validate
```

## The files

| file | what it is |
|---|---|
| `default.yaml` | **the main arm** — `aksssr_v7_trained_arm6`, Final 0.4658 / RadFact F1 0.5139 on validate-40. Every other file is a diff against this one. Do not edit it. |
| `no_source_rules.yaml` | the eleven source rules off, facts file still read — isolates what the rules are worth — `docs/postprocess.md` §2.1, Table 2 |
| `cross_validate.yaml` | the pre-2026-08-07 precision-gate arm: a finding two sources disagree about is dropped rather than reported |

## Writing a new arm

Copy `default.yaml`, change what the experiment changes, name the file for the
question it asks, and put the answer in its `notes:` once you have measured it.
Omitted keys keep the default, so an arm file can be three lines — `cross_validate.yaml`
is one key.

```yaml
arm: aksssr_v7_no_atrophy_rule
based_on: configs/postprocess/default.yaml
notes: does the atrophy rule's recall gain survive its precision cost?

source_rules:
  atrophy: false
```

Check it before spending a run on it. `--print-config` resolves the file over
the defaults and marks every key that differs with `*`:

```bash
python3 code/pipeline/postprocess/postprocess_pred.py \
    --config configs/postprocess/my_arm.yaml --print-config \
    --pred-dir x --out-dir y
```

A key that is not a real setting is an **error**, not a warning. `filling:`
where the schema wants `fillings:` fails at load with the offending path, which
is the failure mode the whole binding table exists to prevent — a misspelled
override that runs clean and scores like the default.

## What is settable, and what is not

Everything settable is listed in `BINDINGS` in
[`code/pipeline/postprocess/rules_config.py`](../../code/pipeline/postprocess/rules_config.py),
which is also the single place that says which module global each key writes.
The sections:

| section | what it controls |
|---|---|
| `source_rules` | the eleven rules of `docs/postprocess.md` §1 — which SOURCE each finding comes from |
| `priors` | what is stated when no read settled the question (today: canal location) |
| `cross_source` | the agreement gates — union vs. drop-on-disagreement |
| `gates` | the standalone drop rules |
| `maxilla_fov` | what happens to an arch the acquisition cut off |
| `extraction` | fallbacks a different schema needs (`schema_dedup.json` needs `wisdom_eruption_fallback`) |
| `report` | what reaches the report text |

**Not settable, on purpose:** FDI ranges, the schema's enum vocabulary,
`ARCH_VALUE_ALIASES`, `ARCH_FINDING_PRIORITY`, `GATE_EXEMPT`, and every builder
and renderer function. Those are what the schema and the anatomy say, not what
an experiment chooses. `schema/schema.json` remains the single source of truth
for fact names, enum values and image requirements — a config file never
restates any of it.

## Three things worth knowing

**The default is not a special case.** `default.yaml` restates the constants the
modules carry, so a run with `--config default.yaml` and a run with no config
produce byte-identical summaries. `--print-config` verifies that and exits
non-zero if the two have drifted apart, so the equivalence is checked rather
than asserted.

**A typed flag beats the file.** `--cross-validate`, `--demote-uncertain` and
the rest still work and still override a config, because a flag on the command
line is an explicit instruction. Every job script and doc example in the repo
therefore keeps its meaning.

**Pass the same config to both stages.** `postprocess_pred.py` builds the
summaries and `synthesize_report.py` renders them; they are one arm. `infer.py`
passes it to both for you — only hand-run invocations can get this wrong.

## Provenance

Every summary carries what built it, so a stored run names its own arm even
though `outputs/` is gitignored and job logs rotate:

```json
"postprocess_config": {
  "config": "configs/postprocess/no_source_rules.yaml",
  "differs_from_default": { "source_rules.atrophy": false, "…": "…" },
  "arm": "aksssr_v7_trained_arm6_no_source_rules"
}
```

Only the keys that differ from `default.yaml` are stored — a run of the main arm
stamps an empty dict rather than thirty restated defaults into all 40 files.

## Where the evidence is

`docs/postprocess.md`. Every key's comment in `default.yaml` names the section
that measured it. The numbers are stated there and nowhere else, so they have
one home and cannot drift.
