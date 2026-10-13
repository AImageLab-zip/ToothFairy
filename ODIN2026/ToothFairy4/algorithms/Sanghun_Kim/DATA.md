# What the pipeline expects on disk

Nothing in this section is distributed with the code. The ToothFairy4 volumes and reports
belong to the challenge organisers, and the training targets fall under the same terms. This
page describes the layout and the file formats instead, so that the code can be read against a
concrete shape.

```
data/
  <pid>/cbct/volume.nii.gz            one CBCT volume per patient
  splits/train.txt, val.txt           one patient id per line
experiments/
  feature_cache/region_features_sc/<pid>.npz          built by toothfairy.cli.cache
  feature_cache/tf2_tooth_density_sc/<pid>.npz        built by toothfairy.cli.cache
  extraction/per_structure_per_report_split/*.json    structure-wise target sentences
  GT/categorical_GT/per_report/*.json                 per-tooth categorical claims
```

The two cache directories are built by `toothfairy.cli.cache`. The two target directories are the
annotations the training reads; they are not distributed, and the formats below are the whole
contract between them and the code.

## Splits

`python -m toothfairy.data.make_splits` writes `data/splits/{train.txt, val.txt,
split_manifest.json}` from `data/splits/split_config.yaml`. The submitted model used a
patient-level split of 560 training and 62 validation patients at seed 42. The split is a
file, not a call to `random.split()`, so an evaluation can be repeated exactly.

## Feature caches

One `.npz` per patient, written by a single pass of each frozen segmentation network. The
input normalisation is the network's own CT normalisation with the upper hard clip replaced
by a soft clip at s = 1000, which is why `toothfairy.cli.cache` fixes that value rather than
exposing it: a cache built with different normalisation no longer matches the checkpoints.

## Structure-wise targets (`train.anat_dir`)

One JSON file per *report*, not per patient — a patient read twice has two files, and
one of them is drawn per epoch. Each file carries `report_id` (the patient
id) and the sentences of that report, already routed to the slot they describe:

```json
{
  "report_id": "A005",
  "fdi:36": ["36 restored with post and core", "periapical radiolucency at 36"],
  "fdi:47": ["47 missing"],
  "maxilla": ["maxillary sinuses clear"],
  "mandible": [],
  "nerve_right": ["right mandibular canal regular in course"],
  "nerve_left": [],
  "cls_upper": ["complete maxillary dentition"],
  "cls_lower": []
}
```

The keys are the 38 anatomical tokens: `fdi:<FDI>` for the 32 permanent teeth, `maxilla`,
`mandible`, `nerve_right`, `nerve_left`, and the two arch summaries `cls_upper` / `cls_lower`.
A key that is absent or empty means that slot has nothing to say, which is a target in its own
right: the slot is still fed and is trained to emit an immediate
end-of-sequence token. `src/toothfairy/data/dataset.py` is the canonical reader, and a file that
carries none of these keys is rejected rather than guessed at — a target that silently lands on
the wrong tooth would not show up in the loss curve.

## Categorical claims (`train.dense_dir`)

One JSON file per report as well, with the **same filename** as its counterpart above. Only
stage 1 reads it: the findings head is trained on these labels, and from stage 2 on there is
no findings head at all.

```json
{"teeth": {"11": {"state": "normal", "endo": "none", "restoration": "none", ...},
           "12": {...}, "...": {}, "48": {...}}}
```

It has to be **dense**: all 32 teeth of `FDI_ALL` and every axis present, with the negative
default written out rather than omitted, because a missing entry cannot be told apart from an
unrecorded one. `build_axis_labels` in `src/toothfairy/schema/claims.py` raises on a gap. The axes
and their categories are defined in that same file, which is the canonical source for the
label space — 10 per-tooth axes plus the region axes.

## Reproducing against your own annotations

Running the training end to end needs the two target directories in the formats above.
Everything the code does with them is specified by `src/toothfairy/data/dataset.py` (which slots
the sentences reach) and `src/toothfairy/schema/claims.py` (the label space), so an equivalent set
of annotations drives the same pipeline.
