# Bite2Text PTv3 staged fine-tuning

## Final v9 recipe (current)

Use the [complete reproduction README](../../reproducibility/README.md),
especially B3–B5 and B7–B8. Final v9 uses **12 heads**, a shared all-200
Bits2Bites initialization, five-fold development training (10 frozen + 60 joint
epochs), and a fixed all-867 model (10 frozen + 47 joint epochs, no validation,
last checkpoint). It constrains report retrieval; it is not the seven-head
template-only system described below. The bootstrap installs `bite2text_dataset.py`
and all necessary upstream training patches. Paths default to `.vendor/Bits2Bites`
and can be overridden by `BITE2TEXT_BITS2BITES_ROOT`.

The section below records the **earlier seven-head baseline only**. Its epoch-49
result, unused photograph socket and sealed development split must not be
reported as the final multimodal v9 configuration.

## Historical seven-head baseline

This directory adapts the all-200 Bits2Bites PT-v3m1 mesh encoder to seven
auditable Bite2Text report facts. The report text is rendered deterministically
from the seven predictions, without free-form language-model decoding.
Incorrect predicted labels can still produce incorrect findings.

Training is intentionally staged:

1. freeze the pretrained PTv3 encoder and fit new report heads;
2. load the best frozen-stage checkpoint, unfreeze the encoder, and fine-tune
   it with a 10x lower backbone learning rate.

`prepare_ptv3_dataset.py` samples each normalized jaw uniformly over STL
triangle area and preserves the paired-jaw frame. `bite2text_dataset.py` is
installed into the Bits2Bites/Pointcept repository by the workflow script.

## Fixed checkpoint and held-out result

The final model is the stage-2 epoch-49 checkpoint selected only by validation
macro-F1. It was evaluated once on the sealed 100-case test split and obtained
macro-F1 `0.4968` (average accuracy `0.6271`). Do not use the held-out test set
to choose another epoch or inference setting.

## Official inference entrypoint

`inference.py` implements the Grand Challenge contract:

- official inputs: `/input/3d-lower-teeth-scan.obj` and
  `/input/3d-upper-teeth-scan.obj`; the OBJ meshes are parsed and exported as
  genuine temporary STL files before IOS-Normalizer runs;
- legacy `/input/files/ios-lower/*.stl` and
  `/input/files/ios-upper/*.stl` layouts remain supported for local testing
  (the photograph socket is accepted but unused);
- preprocessing: paired IOS-Normalizer with preserved occlusion, conservative
  paired post-correction, and 32,768 triangle-area samples per jaw;
- model: the fixed seven-head PTv3 checkpoint and the validation-time voxel
  transform (`NormalizeCoord`, 0.01 `GridSample`);
- output: `/output/diagnostic-imaging-report.json`, containing exactly
  `{"report": "<deterministic English report>"}`.

For a server-side run, use the Pointcept Python environment and override the
container paths:

```bash
export CUDA_VISIBLE_DEVICES=1
export BITE2TEXT_INPUT_PATH=/path/to/official/case/input
export BITE2TEXT_OUTPUT_PATH=/path/to/output
export BITE2TEXT_CONFIG=/path/to/config.py
export BITE2TEXT_CHECKPOINT=/path/to/model_best.pth
export BITE2TEXT_HEAD_VOCABS=/path/to/head_vocabs.json
export BITE2TEXT_NORMALIZER_ROOT=/path/to/IOS-Normalizer
export BITE2TEXT_NORMALIZER_PYTHON=/path/to/IOS-Normalizer/.venv/bin/python
export BITE2TEXT_NORMALIZER_CHECKPOINT=/path/to/IOS-Normalizer/runs/rotation/best.pt
/path/to/Bits2Bites/.venv/bin/python inference.py
```

The diagnostic labels and confidences are printed to stdout only and are never
added to the challenge output JSON.

Single-case inference defaults to FP32 because the current server's `spconv`
build has no FP16 inference algorithm for some sparse single-case shapes. Set
`BITE2TEXT_AMP=1` only after validating the packaged `spconv` build.
