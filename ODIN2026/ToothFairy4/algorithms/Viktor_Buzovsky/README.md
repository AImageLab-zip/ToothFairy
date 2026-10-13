# Segmentation-grounded report generation for CBCT (ToothFairy4)

Code for our entry to **ToothFairy4** at [ODIN 2026](https://odin2026.grand-challenge.org/)
(MICCAI)


```
CBCT volume ──► reorient + resample to 0.3 mm ──► 3D U-Net (77 classes)
            ──► findings record (geometry and intensity rules)
            ──► sentence templates ──► report
```

## Setup

**Environment.** Python 3.13. Install PyTorch for your CUDA version, then the
remaining dependencies:

```bash
pip install -r requirements.txt
```

**Data.** Download from the organisers and place (or symlink) in the repository
root:

| directory | contents | needed for |
|---|---|---|
| `toothfairy4_v03/` | [ToothFairy4](https://ditto.ing.unimore.it/toothfairy4/): `<case>/cbct/volume.nii.gz` and `<case>/reports_en/*.txt` | everything |
| `Dataset123_ToothFairy3/` | [ToothFairy3](https://ditto.ing.unimore.it/toothfairy3) in nnU-Net layout (`imagesTr`, `labelsTr`, `imagesTe`, `labelsTe`) | threshold sweeps only |

`calibrate.py` and `dump_features.py` also accept the `TF4_ROOT` and `TF3_ROOT`
environment variables. The other scripts expect the directories in the repository
root.

**Weights.** Place the 3D Unet weights in `algorithm/model/`.
(for original model contant the authors, yet any model trained on TF3 should do the job given decent performance).
## Reproducing

### Reports for every case

```bash
python tf4/build_findings.py segment    # GPU, caches masks to work/pred_masks/
python tf4/build_findings.py derive     # CPU, writes work/pred_findings.json
python scripts/render_reports.py        # writes work/reports/<case>.txt
```

### Submission container

```bash
cd algorithm
./do_build.sh           # regenerates vendor/ from tf4/ and third_party/, then builds
./do_test_run.sh F001   # expects test/input/samples/F001/images/cbct/F001.mha (USE_GPU=0 for CPU)
./do_save.sh            # exports the image and model.tar.gz for upload
```

`do_build.sh --local` builds a CUDA 12.8 image for local testing on Blackwell
GPUs. The default CUDA 12.6 image is the one that runs on Grand Challenge.

### Local RadFact evaluation

The evaluation container has no network access, so RadFact was measured locally
with the official prompts and `qwen3.6-35b-a3b` served by LM Studio:

```bash
lms load qwen3.6-35b-a3b -c 32768 --parallel 6

python tf4/radfact_fast.py --mode pipeline --n 150 --tag calibrated   # system, 150 cases
python tf4/radfact_fast.py --mode human --n 30 --tag human            # second expert vs first
python tf4/radfact_fast.py --mode pipeline --n 150 --sets F --tag setF
python tf4/radfact_fast.py --compare work/rf_a.json work/rf_b.json    # paired bootstrap
```

Each run writes every per-phrase decision to `work/rf_<tag>.json` and prints the
per-claim attribution table. `--model` and `--base-url` (or `RADFACT_MODEL`,
`RADFACT_BASE_URL`) point it at any OpenAI-compatible endpoint.

### Analyses behind the design

```bash
python tf4/slot_calibration.py                    # entailment rate q per slot (needs work/pred_findings.json)
python tf4/dump_features.py && python tf4/sweep_thresholds.py   # threshold sweeps (ToothFairy3 labels)
python scripts/fit_wording.py                     # greedy wording fit (run from the repo root)
```

The break-even rate is `q* = P / (1 + P²·n / (R²·m))`, where `P` and `R` are
mean per-case precision and recall, and `n` and `m` the mean phrase counts of
the system and the reference. `radfact_fast.py` prints all four for every run
(the `n=` in its summary line is the number of cases).

## Notes on reproducibility

* **Absolute RadFact values depend on the judge.** Local values are not
  comparable with the official score, which uses the organisers' judge. Only
  differences between runs scored with the same judge are meaningful.
* **This is the submitted system.** Several slots were added after the
  150-case measurements in the technical report, so a rerun measures the
  submitted system rather than the calibrated row there. The uncalibrated
  baseline predates this code and is not part of the repository.
* **LLM judging need not be bit-exact.** Rerunning the judge can change
  individual entailment decisions, and with them the scores.
* **Segmentation matches the original run.** Re-segmenting case F001 with this
  repository reproduced the original mask voxel for voxel, and the derived
  findings exactly.
