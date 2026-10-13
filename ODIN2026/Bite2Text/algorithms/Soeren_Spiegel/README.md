# RGBite — ODIN 2026 training code

Training code for our ODIN 2026 submissions:

- **Task 1 (ToothFairy4)**: CBCT volume → diagnostic report
- **Task 2 (Bite2Text)**: intraoral scans + photos → diagnostic report

Both tasks fine-tune `Qwen/Qwen2.5-VL-7B-Instruct` with LoRA (rank 16, α 32, dropout 0.05) on
the language-model projections and the vision tower. The model emits a structured dental chart,
and the report is rendered from that chart. For deployment the adapter is merged and the model is
quantised to int4 (NF4).

## Layout

| Path | Contents |
|---|---|
| `toothfairy/` | Task 1 preprocessing: volume IO + LAS reorientation, HU calibration, arch fit, panoramic / occlusal / MPR renders, cross-section video, chart format + parser |
| `experiments/finetune_qwen_v2/` | Task 1 training (`train.py`), dataset, chart loss, run scripts for the submitted models |
| `experiments/finetune_qwen_v1/` | Dataset module and helpers shared with the Task 1 inference container |
| `data_canonical/` | Task 1 target construction: record validation, per-tooth state, left/right mirroring, chart and prose rendering |
| `bite2text_pipeline/` | Task 2 preprocessing (mesh views, photos), dataset, training, run script |
| `data_bite2text/` | Task 2 record schema (`SCHEMA.md`), chart/prose rendering, target builder |
| `experiments/baseline_hu_v1/03_lora_train/` | Adapter merge + int4 quantisation |
| `splits/`, `data_bite2text/splits/` | Frozen case-level splits (seed 2026) |
| `eval/explore/tf_zip.py` | Streaming reader for the ToothFairy4 zip, which is never extracted |

## Setup

```bash
pip install torch            # the build matching your CUDA
pip install -r requirements.txt
```

Download the challenge data and point the code at it:

```bash
export TF_ZIP_PATH=/path/to/toothfairy4_v03.zip     # Task 1
export TF_CACHE_DIR=/path/to/cache                  # optional local volume cache
export B2T_ZIP=/path/to/Bite2Text.zip               # Task 2
```

## Training targets

The models are trained on structured per-report records (FOV statuses, per-tooth status,
restorations, endodontics, lesions, and the non-tooth findings) that are rendered into the chart
and prose targets. **These records are derived from the challenge reports and are not included in
this repository.** Place them as follows, then build the target files:

```bash
# Task 1: per-case merged records  -> data_canonical/raw_batches/*.json   ({case_id: record})
#         per-report records       -> data_canonical/raw_reports/<case>__<report_id>.json
python data_canonical/validate.py --emit            # -> splits/canonical/{train,val,test}.jsonl
cat splits/canonical/{train,val,test}.jsonl > splits/canonical/all622.jsonl

# Task 2: per-case records (schema: data_bite2text/SCHEMA.md) -> data_bite2text/raw_cases/<case>.json
python data_bite2text/build_canonical.py           # -> data_bite2text/splits/canonical/*.jsonl
```

`validate.py` defines the record schema and controlled vocabularies for Task 1. Run
`python data_canonical/chart_v2.py --test` and `python data_canonical/chart_render.py --test` to
check the round-trip and mirror invariants on your records.

## Training the submitted models

Each run trains on one NVIDIA RTX PRO 6000 (96 GB) with batch 1 × gradient accumulation 8.

| Submission | Command | Epochs (submitted) | Wall-clock |
|---|---|---|---|
| Task 1, two panoramics | `bash experiments/finetune_qwen_v2/run_chartv2_pano_v1.sh` | 8 (ep 7) | ~2.4 h |
| Task 1, five stills + video | `bash experiments/finetune_qwen_v2/run_chartv2_fiveviews_nt_all_v1.sh` | 8 (ep 4) | ~6.3 h |
| Task 2 | `bash bite2text_pipeline/run_b2t_chartqa_v1.sh` | 10 (ep 10) | ~8.4 h |

Adapters are saved per epoch under `runs/<run>/adapter_ep<N>`. Pass `--wandb` (and optionally
`--wandb-project` / `--wandb-entity`) to log to Weights & Biases.

## Export

For deployment each adapter is merged into the base model
(`experiments/baseline_hu_v1/03_lora_train/merge_export.py`) and quantised to int4 NF4
(`quantize_int4.py` in the same directory). For Task 2 one script runs merge, quantisation,
tarball and a one-case verification:

```bash
WORK_DIR=/path/to/work OUT_DIR=/path/to/tarballs bite2text_pipeline/export_model_tarball.sh 10
```

The Grand Challenge inference containers are not part of this repository.
