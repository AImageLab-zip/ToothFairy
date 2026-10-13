# Bite2Text · ODIN 2026

**Orthodontic report generation from intraoral scans and photographs.**
Our submission to the [Bite2Text track](https://ditto.ing.unimore.it/bite2text/)
of the [ODIN 2026 challenge](https://odin2026.grand-challenge.org/) (MICCAI 2026).

Give it an upper and a lower intraoral scan (and, optionally, the five
intraoral photographs) and it writes a clinical report:

> The patient presents correct transverse relationships. From a vertical
> standpoint, the vertical relationship is within normal limits. Sagittally,
> there is a bilateral Class I molar and canine relationship with overjet within
> normal limits. …

The pipeline measures the occlusion geometrically, predicts a fixed set of
clinical findings, and renders them with sentences mined from real clinician
reports. **No language model runs at inference**, so it cannot state a finding
it did not predict.

---

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Sample case](#sample-case)
- [Python API](#python-api)
- [Grand Challenge container](#grand-challenge-container)
- [Repository layout](#repository-layout)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [License and citation](#license-and-citation)

---

## How it works

```
ios_upper.stl ─┐
ios_lower.stl ─┤
               ├─► tooth segmentation ─────────► CLIK crown landmarks ─► 25 occlusal measurements
               │     (per face, 16k-face mesh)     (≈9 per tooth)          (pose-free frame from the landmarks)
               │                                                                 │
               │                                        15 occlusal findings ◄───┘
               │                                   (gradient boosting / trees / majority)
               │                                                 │
intraoral_1..5 ┴─► multi-view ResNet-18 ─► 10 photo findings     │
                    (+ cohort prior for untrusted heads)  │      │
                                                          ▼      ▼
                                              sentence bank mined from training reports
                                                          │
                                                          ▼
                                                     report text
```

| stage | module | what it does |
| --- | --- | --- |
| scan loading | `bite2text/scan_io.py` | detects the mesh format from the file bytes (Grand Challenge names STL data `.obj`) |
| segmentation | `bite2text/segmentation.py` | decimates to 16 000 faces, runs the **tooth segmentation network** (17 classes: gingiva + 16 teeth), absorbs islands < 50 mm², lifts labels back to full resolution, splits one mesh per tooth (FDI numbering) |
| landmarks | `bite2text/landmarks.py` | canonicalises the jaw, samples 2048 points per tooth, runs the **CLIK PointMLP** detector for the tooth type (incisor / cuspid / premolar / molar), maps landmarks back to scan millimetres |
| measurements | `bite2text/features.py` | overjet, overbite, molar and canine antero-posterior relation per side, transverse offsets, midline deviation, arch widths, Little's irregularity index, arch-length discrepancy, curves of Spee and Wilson, tooth counts |
| occlusal findings | `bite2text/slots.py` | transverse constriction, crossbite, molar and canine class (both sides), overjet, overbite, curves, midlines, crowding |
| photo findings | `bite2text/photo_findings.py` | shared-weight ResNet-18 over the five views; gingivitis, restorations and recession come from the network, the other seven findings from a cohort prior (those heads did not beat their majority baseline) |
| rendering | `bite2text/render.py` | exact-match lookup in a sentence bank mined from training reports, with programmatic synthesis as fallback |

Everything that uses report text — label extraction, sentence mining, classifier
fitting, photo-CNN training — used the **791-subject training split only**.

---

## Quick start

Tested with Python 3.11, PyTorch 2.9–2.11, CUDA 12 on an NVIDIA T4.

```bash
git clone https://github.com/Ali5hadman/Bite2Text-odin2026.git
cd Bite2Text-odin2026

# 1. PyTorch for your platform (https://pytorch.org/get-started/locally/), then:
pip install torch torchvision
pip install -r requirements.txt

# 2. CLIK landmark weights (4 × 618 MB) from the CLIK-Diffusion authors' Google Drive
python scripts/download_weights.py

# 3. Run on the bundled held-out case
python predict.py \
    --upper  sample_data/F5535/ios_upper.stl \
    --lower  sample_data/F5535/ios_lower.stl \
    --photos sample_data/F5535 \
    --output outputs/F5535.json
```

| option | meaning |
| --- | --- |
| `--upper`, `--lower` | jaw scans: `.stl`, `.obj`, `.ply` (millimetres) |
| `--photos` | image files and/or folders. `intraoral_1` … `intraoral_5` map to front / right / left / lower / upper views; other names are read in name order. Omit to use the cohort prior for the photo section |
| `--output` | JSON with the report, the predicted findings, and which source the photo section came from |
| `--landmarks` | JSON with every tooth's landmarks, `{fdi: {name: [x, y, z]}}`, in scan coordinates |
| `--device` | `cuda` or `cpu` (default: CUDA if available) |
| `--seed` | seed for landmark point sampling (default `0`; results are deterministic for a given seed) |

Weights are read from `weights/`; set `BITE2TEXT_WEIGHTS` to use another folder.
See [`weights/README.md`](weights/README.md).

**Speed** on the sample case (both jaws + five photos, NVIDIA T4): **33 s** of
inference, ~55 s wall-clock including Python start-up and loading ~2.6 GB of
checkpoints. Per stage: upper scan 18 s, lower scan 7 s, occlusal findings 6 s
(classifiers fitted once per process), photo findings 1 s. CPU-only works but
is several times slower.

---

## Sample case

> **Not yet included.** The sample files are being added once redistribution
> is confirmed. Until then, run on any Bites2Text case, e.g.
> `--upper <case>/ios/ios_upper.stl --lower <case>/ios/ios_lower.stl --photos <case>/intraoral-photo`.

[`sample_data/F5535`](sample_data/) will hold one complete patient from the
Bites2Text dataset: both scans, the five photographs, both clinician reports,
and our expected output. **It is from the frozen held-out test split** — it was
never seen in training, template mining, or any tuning decision.

---

## Python API

```python
from pathlib import Path
from bite2text import ReportGenerator

generator = ReportGenerator(device="cuda", seed=0)     # models load on first use, then stay cached

report = generator(
    upper="sample_data/F5535/ios_upper.stl",
    lower="sample_data/F5535/ios_lower.stl",
    photos=sorted(Path("sample_data/F5535").glob("intraoral_*.jpg")),
)

report.text           # the full report
report.occlusal_text  # scan-derived paragraph
report.photo_text     # photo-derived sentences
report.photo_source   # "model" (patient's photos were read) or "prior"
report.slots          # {"mr_class": "I", "overjet": "normal", ...}
report.photo_slots    # {"gingivitis": False, "restorations": None, ...}
report.landmarks      # {"16": {"Mesiobuccal cusp": [x, y, z], ...}, ...}
report.timings        # seconds per stage
```

Individual stages are usable on their own:

```python
from bite2text.scan_io import load_scan
from bite2text.segmentation import segment_faces, split_teeth
from bite2text.landmarks import get_landmarks

mesh = load_scan("ios_upper.stl")
labels = segment_faces(mesh, "upper")          # (F,) 0 = gingiva, 1..16 = teeth
teeth = split_teeth(mesh, labels, "upper")     # {11: Trimesh, 12: Trimesh, ...}
landmarks = get_landmarks(teeth)               # {11: {"Incisal center": (x, y, z), ...}, ...}
```

---

## Grand Challenge container

`inference.py` is the container entrypoint. It follows the Bite2Text I/O contract:

```
/input/inputs.json                           socket declarations
/input/files/ios-lower/*                     3d-lower-teeth-scan
/input/files/ios-upper/*                     3d-upper-teeth-scan
/input/images/intraoral-photo/*              2d-intraoral-photographs (separate files or one stacked TIFF/.mha)
/opt/ml/model/                               contents of weights/
/output/diagnostic-imaging-report.json       {"report": "..."}
```

Both scans are required; if either is missing the job fails with a non-zero exit
and writes no output. Photographs are optional.

```bash
./scripts/docker_test.sh                 # build + run on sample_data/F5535 with --network none, read-only /input
GPUS=all ./scripts/docker_test.sh        # same, on GPU (needs nvidia-container-toolkit)
./scripts/save_submission.sh             # image tarball + model.tar.gz for upload
```

Without Docker, the entrypoint also runs locally:

```bash
BITE2TEXT_INPUT=/path/to/case BITE2TEXT_OUTPUT=outputs/gc BITE2TEXT_WEIGHTS=weights python inference.py
```

---

## Repository layout

```
Bite2Text-odin2026/
├── predict.py                 command-line inference
├── inference.py               Grand Challenge entrypoint
├── bite2text/                 the pipeline
│   ├── pipeline.py            ReportGenerator: scans + photos -> Report
│   ├── scan_io.py             mesh loading with format sniffing
│   ├── segmentation.py        tooth segmentation + per-tooth meshes
│   ├── landmarks.py           CLIK crown landmarks
│   ├── features.py            occlusal measurements from landmarks
│   ├── slots.py               occlusal finding classifiers
│   ├── photo_io.py            photo decoding (JPEG/PNG/TIFF/BigTIFF/.mha) + view assignment
│   ├── photo_findings.py      photo-finding predictor
│   ├── render.py              findings -> sentences
│   ├── config.py              paths, label conventions, defaults
│   ├── nets/                  segmentation network, PointMLP, MultiViewPhotoNet
│   └── data/                  sentence banks, cohort prior, classifier training table
├── weights/                   tooth segmentation + photo CNN (bundled), CLIK (downloaded)
├── sample_data/F5535/         held-out sample case
├── scripts/
│   ├── download_weights.py    fetch + verify weights
│   ├── score.py               BLEU-4 / METEOR-lite, as computed on the platform
│   ├── docker_test.sh         container smoke test
│   └── save_submission.sh     upload artefacts
├── tests/test_pipeline.py
├── Dockerfile, requirements.txt, requirements-dev.txt
└── LICENSE, NOTICE.md
```

---

## Evaluation

The final challenge score is `0.8 × RadFact-F1 + 0.2 × mean(BLEU-4, METEOR)`,
computed against the photo-derived reference report. Only BLEU-4 and METEOR
are computed on the platform; RadFact needs an LLM judge and is evaluated by
the organisers. `scripts/score.py` reproduces the platform's text metrics:

```bash
pip install -r requirements-dev.txt
python scripts/score.py --prediction outputs/F5535.json \
                        --reference  sample_data/F5535/reference_report_photo_en.txt
```

Tests:

```bash
pytest -q -m "not slow"      # seconds, no weights needed
pytest -q                    # + end-to-end on the sample case, checks the expected output
```

---

## Limitations

- **Research code, not a medical device.** Reports must be reviewed by a clinician.
- The scans must be single jaws in millimetres, with upper and lower in a shared occlusion frame (as in Bites2Text).
- Several findings sit at the inter-rater noise floor — curve of Spee, crowding, midline deviation — and are emitted as the training-set majority rather than predicted.
- Photo findings are presence-level only and never localised to specific teeth; seven of the ten come from the cohort prior, which asserts common findings without seeing the patient.
- Third molars are segmented but get no landmarks; mixed dentitions can be mis-numbered.
- **GPU and CPU runs are not bit-identical.** Floating-point differences move landmarks slightly, which can flip a finding that sits on a class boundary. On the sample case the CPU run reports a left canine Class II instead of Class I. `expected_output.json` and the end-to-end test assume a CUDA GPU, as on Grand Challenge. Results are deterministic for a given device, seed and the pinned package versions.

---

## License and citation

Code: **GNU AGPL-3.0** (see [`LICENSE`](LICENSE)). Third-party code and weights
keep their own terms, listed in [`NOTICE.md`](NOTICE.md). The CLIK checkpoints
are downloaded from their authors and not redistributed.

Data: the sample case, sentence banks, cohort prior and classifier training
table are derived from the Bites2Text dataset and are licensed
**CC BY-NC-SA 4.0** (non-commercial). If you use them, cite:

```
L. Lumetti, F. Rizzo, F. Cremonini, E. Candeloro, L. Lombardo, C. Grana, F. Bolelli.
Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines.
European Conference on Computer Vision (ECCV), 2026.
```

This work builds on:

- **ortho-landmark-dgcnn** — the landmarking and occlusal-measurement extraction build heavily on
  [github.com/Ali5hadman/ortho-landmark-dgcnn](https://github.com/Ali5hadman/ortho-landmark-dgcnn):
  A. Shadman Yazdi, B. Baldini, G. Tartaglia, G. Baselli, *Automated Orthodontic Diagnosis via Geometric Deep Learning: A Clinically Interpretable Framework*, 2026.

```bibtex
@article{shadmanyazdi2026orthodontic,
  title   = {Automated Orthodontic Diagnosis via Geometric Deep Learning: A Clinically Interpretable Framework},
  author  = {Shadman Yazdi, Ali and Baldini, Benedetta and Tartaglia, Gianluca and Baselli, Giuseppe},
  year    = {2026}
}
```

- **TSGCNet** — L. Zhang et al., *TSGCNet: Discriminative Geometric Feature Learning with Two-Stream Graph Convolutional Network for 3D Dental Model Segmentation*, CVPR 2021.
- **CLIK-Diffusion** — Y. Dou et al., *CLIK-Diffusion: Clinical Knowledge-informed Diffusion Model for Tooth Alignment*, Medical Image Analysis 106, 2025.
- **Multi-head heatmap landmark detection** — M. Lazarevski, A. Shadman Yazdi, G. Baselli, *A Multi-Head Heatmap Architecture for Full-Mouth 3D Dental Landmark Detection*, Proc. 13th Int. Conf. on Electrical, Electronic and Computing Engineering (icETRAN), 2026.

```bibtex
@inproceedings{lazarevski2026multihead,
  author    = {Marko Lazarevski and Ali Shadman Yazdi and Giuseppe Baselli},
  title     = {A Multi-Head Heatmap Architecture for Full-Mouth 3D Dental Landmark Detection},
  booktitle = {Proc. 13th Int. Conf. on Electrical, Electronic and Computing Engineering (icETRAN)},
  year      = {2026},
}
```
