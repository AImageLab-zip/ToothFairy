# Reproduce the final Bite2Text v9 system

This is the executable recipe for **Task 2 only**, not the earlier seven-head
baseline. Run commands from the repository root in **Bash on Linux/amd64**.
Read [validation](validation/README.md) for what has actually been retested.
The historical submitted source is `0d0d83e1a962f361bc3b70e4d36ae129d338708b`;
the reproduction tooling recovers missing training/build dependencies.

There are two different reproduction targets:

| Route | Required assets | What it reproduces |
|---|---|---|
| A: published submission | Full submission bundle + an authorized input case | Exact submitted image and checkpoint files; offline inference |
| B: retrain | Authorized Bits2Bites and Bite2Text training data, upstream pretrained weights, CUDA development environment | Data preparation, CV, fixed full-data training, retrieval-bank reconstruction and new runtime assets |

Retraining is **not guaranteed byte-identical** across CUDA, GPU, library or
filesystem versions. Exact historical checkpoints are available through Route A.
Hidden-test references, official Arena judgments and organizer RadFact results
are not recoverable by training on the public data. No paid LLM API is needed
for the final algorithm. Optional GLM-based RadFact-Lite is a proxy evaluation,
not a component of inference or the official challenge evaluator.

## A. Run the released submission

### A1. Download, verify, unpack

Download into `artifacts/` using the browser (large Drive downloads may show a
confirmation page; do not save that HTML as a ZIP):

- **Full submission bundle:** [ODIN2026_Bite2Text_v9_Final_Submission.zip](https://drive.google.com/file/d/1K4eBy1yikNn5vnwmpTz215F-FK8_KDiU/view), 4,479,620,106 bytes.
- **Trained model weights:** [Bite2Text_v9_public_weights.zip](https://drive.google.com/file/d/1gjF4qhSN8jg0XQ_ysvf9toVUjn4hxH4a/view), 125,784,118 bytes.

The full submission bundle contains the Docker export **and a separate model archive**;
both are required. The weights-only archive is insufficient: it omits the pretrained IOS-Normalizer
and the report/label/index retrieval bank. Neither contains the all-200
initialization checkpoint, which Route B trains and exports explicitly.
The [machine-readable manifest](v9_assets.json) records all hashes and the
expected Docker image ID. Public access is not a replacement for the applicable
dataset/model licenses. The author's original code and trained-model contributions
are MIT-licensed; [MODEL_LICENSE](../MODEL_LICENSE) identifies the covered
checkpoints and provides the notice. Third-party preprocessing weights, data and
retrieval banks retain their applicable terms. This declaration applies to the
existing checkpoint bytes, so the published archives and hashes are unchanged.
When redistributing the covered checkpoints, include that notice alongside them.

```bash
python3 scripts/v9_assets.py verify-archive submission artifacts/ODIN2026_Bite2Text_v9_Final_Submission.zip
python3 scripts/v9_assets.py verify-archive weights artifacts/Bite2Text_v9_public_weights.zip
python3 scripts/v9_assets.py unpack-submission artifacts/ODIN2026_Bite2Text_v9_Final_Submission.zip artifacts/v9-release
python3 scripts/v9_assets.py verify-model artifacts/v9-release/model
docker load --input artifacts/v9-release/odin2026-bite2text-hybrid-photo-test-v9_2026-08-18T11-22-00.039286625+08-00.tar.gz
docker image inspect --format '{{.Id}}' odin2026-bite2text-hybrid-photo-test-v9:latest
```

Expected image ID:
`sha256:1422835134dd43439969a40c3f88dc05bce3963fcb9b58a2d77d4ecc80e8a920`.
If this tag already points at your own image, record/tag that image separately
before loading. The unpacker refuses existing destinations and verifies hashes
**before** any checkpoint deserialization. Only load trusted PyTorch checkpoints.
Allow space for the ZIP, extracted image archive and expanded Docker layers.
The manifest retains the published archive's historical metadata filenames for
compatibility; the archive bytes, download links and checksums are unchanged.

The model mount contains exactly these nine required assets:

```text
config.py                 head_vocabs.json          model_final.pth
photo_model_final.pt       photo_view_classifier.pt ios_normalizer_best.pt
retrieval_index.npz        retrieval_reports.json   retrieval_labels.json
```

### A2. Run one case with network access disabled

Runtime: Linux/amd64, Docker, NVIDIA Container Toolkit, one CUDA-compatible GPU,
16 GB container RAM. Historical development used RTX 4090 D (24 GB).
The exported runtime contains PyTorch 2.9.1/CUDA 12.6; Route B uses a separate
PyTorch 2.5.0/CUDA 12.4 training environment. Driver compatibility must cover
the runtime you actually use. Do not run this on a GPU occupied by another job.

For an already exported Grand Challenge input, use its parent as the input root.
Alternatively create a fixture from **an authorized training case**, replacing
`CASE_ID` with a real directory name:

```bash
python3 -m venv .venv-fixture
.venv-fixture/bin/python -m pip install Pillow==12.3.0
.venv-fixture/bin/python task2_bite2text/hybrid_submission_v9_final/make_official_photo_fixture.py \
  --case-id CASE_ID --raw-root /absolute/path/Bite2Text_raw \
  --output-root "$PWD/outputs/smoke/input"
export BITE2TEXT_TEST_INPUT_ROOT="$PWD/outputs/smoke/input"
export BITE2TEXT_TEST_CASE=CASE_ID
export BITE2TEXT_TEST_MODEL_ROOT="$PWD/artifacts/v9-release/model"
export BITE2TEXT_TEST_OUTPUT_ROOT="$PWD/outputs/smoke/output"
export BITE2TEXT_TEST_GPU=0
bash task2_bite2text/hybrid_submission_v9_final/do_test_run.sh
```

Accepted layouts: official `3d-lower-teeth-scan.obj` and
`3d-upper-teeth-scan.obj`, or legacy `files/ios-lower/*.stl` and
`files/ios-upper/*.stl`; photos under `images/intraoral-photo/`, including
multi-page TIFF, with `inputs.json` socket metadata. The fixture includes
photos. Missing photos exercise a fallback, not the complete multimodal path.
The container runs as a non-root user: grant read/traverse permission on a
dedicated input/model copy if necessary. The runner never changes their permissions.

Success: process exit 0 and a nonempty `diagnostic-imaging-report.json` containing
exactly `{"report": "..."}`. Output goes to a fresh case/timestamp directory;
inspect logs to confirm the photo branch, normalization and model loading were
actually used. A training-case smoke run is **not a held-out score** and can
retrieve its own training report.

### A3. Rebuild the source image (optional)

```bash
bash scripts/build_v9_image.sh
```

This builds PTv3-v3 → hybrid-v5 → final-v9. It fetches pinned upstreams, applies
three Bits2Bites patches and one IOS-Normalizer patch, and installs the local
Bite2Text dataset adapter. These patches include frozen-backbone support,
empty-label loss handling, macro-F1 checkpoint selection, deterministic scan
sampling and the `seed` argument used by inference. A rebuilt image need not
have the historical image digest (base tags/OS packages can change). Use Route
A1 when exact submitted bytes are required. The build creates/replaces the
three local image tags; use a dedicated Docker environment for source rebuilds.

## B. Reproduce training and development experiments

Use a **new clone and new output/vendor directories**, not the original server
checkout. Some historical training scripts reuse a completed checkpoint by
epoch count, and are not general resume managers. Never mix old experiments
with a new dataset under the same run name. On failure, preserve logs and use
a fresh run directory after diagnosing the error. Do not rerun into partial data.

### B1. Software and paths

Install Git, rsync, Bash, util-linux (`flock`), a C++ toolchain, CUDA 12.4 toolkit
(`nvcc`) and `uv`. Native Pointcept extensions require a development toolkit,
not only the driver. Fetch the pinned upstreams:

```bash
export ROOT="$PWD"
export BITE2TEXT_PROJECT_ROOT="$ROOT"
export BITE2TEXT_VENDOR_ROOT="$ROOT/.vendor"
export BITE2TEXT_BITS2BITES_ROOT="$BITE2TEXT_VENDOR_ROOT/Bits2Bites"
export BITS="$BITE2TEXT_BITS2BITES_ROOT"
export IOS="$BITE2TEXT_VENDOR_ROOT/IOS-Normalizer"
export WORK="$ROOT/outputs/reproduce-v9"
export RAW_BITS=/absolute/path/Bits2Bites_raw
export RAW_BITE=/absolute/path/Bite2Text_raw
export NORMALIZER="$ROOT/artifacts/v9-release/model/ios_normalizer_best.pt"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
mkdir -p "$WORK"
bash scripts/bootstrap_upstreams.sh
(
  cd "$BITS"
  uv sync --frozen --extra gpu
  uv pip install --python .venv/bin/python ./libs/pointops --no-build-isolation
)
uv venv --python 3.10 "$IOS/.venv"
uv pip install --python "$IOS/.venv/bin/python" -r reproducibility/requirements-normalizer.txt
uv venv --python 3.10 .venv-photo
uv pip install --python .venv-photo/bin/python -r task2_bite2text/photo_pipeline/requirements-training.txt
export BP="$BITS/.venv/bin/python"
export NP="$IOS/.venv/bin/python"
export PP="$ROOT/.venv-photo/bin/python"
export PT="$ROOT/task2_bite2text/ptv3_finetune"
export PH="$ROOT/task2_bite2text/photo_pipeline"
export PYTHONPATH="$BITS:${PYTHONPATH:-}"
"$BP" -c 'import torch, spconv.pytorch, pointops; print(torch.__version__, torch.cuda.is_available())'
```

Bits2Bites uses its pinned `uv.lock` (Python 3.10, torch 2.5.0, CUDA 12.4).
The normalizer dependency snapshot was checked on the development server on
2026-09-26; it is not proof of byte-identical historical package resolution.
The photo requirements are pinned separately. ImageNet ResNet-18 weights
(`ResNet18_Weights.IMAGENET1K_V1`, torchvision's `DEFAULT`) download on first
training use; allow network access then or populate `TORCH_HOME` beforehand.
Do not replace this backbone with Qwen: the algorithm URL is historical, not
the model architecture. IOS-Normalizer weights can also be obtained from its
pinned upstream README; verify the checksum against `v9_assets.json`.

### B2. Authorized data layout and cohort policy

Obtain Bits2Bites v01 through the [upstream project](https://github.com/AImageLab-zip/Bits2Bites)
and the official Bite2Text training release through the
[challenge](https://odin2026.grand-challenge.org/). Accept their terms and unpack
outside Git; set the paths above to the directories **inside** any ZIP wrapper:

```text
Bits2Bites_raw/Annotations.csv
Bits2Bites_raw/<numeric-patient>/lower.stl, upper.stl
Bite2Text_raw/Fxxxx/ios/ios_lower.stl, ios_upper.stl
Bite2Text_raw/Fxxxx/intraoral-photo/<photographs>
Bite2Text_raw/Fxxxx/reports_intraoral-photo_en/*.txt
Bite2Text_raw/Fxxxx/reports_ios_en/*.txt
```

The historical raw Bite2Text release has 997 patient directories. The label
parser uses seed 20260807 and **no external incomplete-case exclusion list**;
missing scans/references are filtered downstream. There are 994 raw scan pairs;
the historical normalization input excludes `F5500`, leaving 993 normalized
pairs. There are 993 manifest records and 991 point records after excluding
`F4885` (the frozen preprocessing exclusion). Restrict final supervised training and
the retrieval bank to the **867** cases with at least one supported label.
Do not include test-phase hidden inputs or obtain their reports.

These counts are guards for this release, not universal assumptions about
future dataset versions. Stop and inspect audits if they differ. `train/val/test`
in the parser output are development partitions of the authorized training
release, **not** the challenge hidden test set. The official-photo-first target
uses the first lexicographically ordered English photograph report. Labels are
automatically derived weak supervision, not new expert annotations.

### B3. Bits2Bites five folds and all-200 encoder

```bash
(
  cd "$BITS"
  "$BP" pointcept/datasets/preprocessing/dental/prepare_bits2bites.py \
    --dataset-root "$RAW_BITS" --output-dir data/dental_landmarks_mesh --workers 8
)
BITE2TEXT_BITS_FOLDS="1 2 3 4 5" bash run_bits2bites_folds2_to5_all200.sh
```

This prepares each fold explicitly, trains 200 epochs per fold (seed 2026,
batch 8), tests best/last checkpoints, aggregates metrics, then trains all 200
without a validation set and exports only the encoder. The architecture and
optimizer come from pinned `configs/dental/cls-ptv3-base.py` and its base config.
To reproduce **only the fixed final model**, skip repeating Bits2Bites CV:

```bash
BITE2TEXT_BITS_SKIP_CV=1 bash run_bits2bites_folds2_to5_all200.sh
export ENCODER="$BITS/exp/dental/ptv3_mesh_mtl_all200_seed2026/model/ptv3_encoder_all200_seed2026.pth"
test -s "$ENCODER"
```

Run either the full-CV command or the skip-CV command as appropriate. Both end
at the same all-200 initialization artifact; do not substitute a random
upstream task-head checkpoint or the final Bite2Text checkpoint for the encoder.

### B4. Normalize scans, parse reports, construct points and folds

```bash
python3 scripts/prepare_reproduction.py stage-ios --raw-root "$RAW_BITE" --output-root "$WORK/ios-input" --exclude-patient F5500
"$NP" "$IOS/scripts/batch_orient_scans.py" \
  --input-dir "$WORK/ios-input" --output-dir "$WORK/ios-oriented" \
  --checkpoint "$NORMALIZER" --device cuda --preserve-occlusion \
  --center-and-orient --save-matrix --workers 4 --seed 20260809
"$NP" scripts/postcorrect_ios_normalizer.py \
  --input-dir "$WORK/ios-oriented" --output-dir "$WORK/ios-final" --log-interval 50
mkdir "$WORK/metadata-empty"
"$PP" scripts/audit_bite2text.py --dataset-root "$RAW_BITE" --output-dir "$WORK/raw-audit"
python3 task2_bite2text/labeling/parse_bite2text_reports.py \
  --data-root "$RAW_BITE" --metadata-dir "$WORK/metadata-empty" --output-dir "$WORK/labels"
export HEADS=right_molar_relation,right_canine_relation,left_molar_relation,left_canine_relation,overjet,vertical_relation,midline_relation,crossbite,upper_crowding,lower_crowding,curve_spee,curve_wilson
python3 task2_bite2text/mesh_baseline/prepare_manifest.py \
  --labels-csv "$WORK/labels/report_labels.csv" --data-root "$RAW_BITE" \
  --output-dir "$WORK/manifest" --target-policy official_photo_first --heads "$HEADS"
export DATA="$BITS/data/bite2text_ptv3_surface32k_v3_official_12head"
export CV="${DATA}_cv5"
export FULL="${DATA}_full867"
"$BP" "$PT/prepare_ptv3_dataset.py" --manifest "$WORK/manifest/manifest.jsonl" \
  --head-vocabs "$WORK/manifest/head_vocabs.json" --normalized-root "$WORK/ios-final" \
  --output-root "$DATA" --points-per-jaw 32768 --workers 4 --seed 2026 --exclude-patient F4885
"$BP" "$PT/make_cv_folds.py" --data-root "$DATA" --output-root "$CV" --folds 5 --seed 20260810
"$BP" "$PT/make_full_dataset.py" --data-root "$DATA" --output-root "$FULL"
"$BP" scripts/check_reproduction_data.py --data "$DATA" --cv "$CV" --full "$FULL"
```

Keep upper/lower jaws in their common occlusal frame. The lower-jaw transform
is applied to both, followed by the same paired post-correction. Point sampling
is area-weighted, 32,768 points per jaw, upper first, and patient-hashed seed 2026.
Do not independently center jaws. `metadata-empty` is intentional and should
contain no `incomplete_cases.csv`, matching the original parser audit.

### B5. PTv3 CV and fixed full-data training

```bash
bash task2_bite2text/ptv3_finetune/run_ptv3_v3_cv5.sh
bash task2_bite2text/ptv3_finetune/run_ptv3_v3_full867.sh
export FINAL_PT="$BITS/exp/dental/bite2text_ptv3_v3_official_12head_full867_stage2_joint_seed20260815"
test -s "$FINAL_PT/model/model_last.pth"
```

CV: five folds, seed 20260810 + fold − 1; frozen stage 10 epochs, joint stage
60 epochs, batch 8; select by mean head macro-F1. Historical best joint epochs:
17, 59, 47, 38, 48. Full model: seed 20260815, frozen 10 + joint **47** fixed
epochs, all 867, **no validation**, last checkpoint. AdamW, head LR 1e-4,
joint backbone LR 1e-5, weight decay .01, cosine schedule, weighted masked CE.
Frozen stage leaves normalization buffers adapting while backbone parameters
are gradient-free. Transform details are in `make_ptv3_configs.py`: joint
NormalizeCoord, scale/shift/z-rotation/dropout, voxel .01, shuffle for training.
The final checkpoint's `best_metric_value=-inf` is expected with evaluation off.

For fixed final training only, CV training is optional, but **fold assignment
generation remains required** for the photo cohort. Full-data training does not
reselect 47 using any hidden-test result. No original encoder weights are shipped
in the weights-only archive, so B3 is mandatory unless you already have that exact initialization.

### B6. Photos: view selection, CV, final model

```bash
"$PP" "$PH/audit_intraoral_photos.py" --raw-root "$RAW_BITE" --output-dir "$WORK/photo-audit" --skip-contact-sheets
"$PP" "$PH/train_view_classifier.py" --manifest "$WORK/photo-audit/photo_manifest.csv" \
  --output-dir "$WORK/view" --label-scheme structural3 --epochs 6 --batch-size 64 \
  --workers 8 --learning-rate 1e-4 --weight-decay 1e-4 --image-size 224 --val-fraction .2 --seed 2026
export VIEW="$WORK/view/view_classifier_best.pt"
"$PP" "$PH/select_five_views.py" --manifest "$WORK/photo-audit/photo_manifest.csv" \
  --checkpoint "$VIEW" --output-dir "$WORK/selection" --batch-size 128 --workers 8 \
  --diversity-weight 2 --incomplete-fill-threshold .20 --skip-montages
"$PP" "$PH/prepare_photo_cache.py" --selection "$WORK/selection/five_view_selection.csv" \
  --output-dir "$WORK/photo-cache" --max-side 640 --quality 92 --workers 8
for fold in 1 2 3 4 5; do
  "$PP" "$PH/train_multiview_12head.py" \
    --cache-manifest "$WORK/photo-cache/cached_selection.csv" --labels "$FULL/labels.csv" \
    --fold-assignments "$CV/fold_assignments.csv" --head-vocabs "$FULL/head_vocabs.json" \
    --view-checkpoint "$VIEW" --fold "$fold" --output-dir "$WORK/photo-cv/fold$fold" \
    --epochs 15 --batch-size 16 --workers 8 --image-size 224 --backbone-lr 2e-5 \
    --head-lr 2e-4 --weight-decay 1e-4 --patience 4 --seed 2026
done
"$PP" "$PH/train_multiview_full.py" \
  --cache-manifest "$WORK/photo-cache/cached_selection.csv" --labels "$FULL/labels.csv" \
  --fold-assignments "$CV/fold_assignments.csv" --head-vocabs "$FULL/head_vocabs.json" \
  --view-checkpoint "$VIEW" --output-dir "$WORK/photo-full" --epochs 10 --batch-size 16 \
  --workers 8 --image-size 224 --backbone-lr 2e-5 --head-lr 2e-4 --weight-decay 1e-4 --seed 2026
```

The view classifier really ran **6**, not its CLI default of 8 epochs; the
historical best was epoch 2. It uses canonical five-file cases for structural
weak labels (frontal/lateral/occlusal), 865 patients split 692/173. Unique
Hungarian assignment selects 1 frontal + 2 lateral + 2 occlusal slots; absent
views are masked, not fabricated. The final photo model learns all 12 heads
but inference uses only eight supported heads. Full photo training runs 10
epochs on 867 cases; CV seeds are 2026 + fold. To reproduce only the fixed
final model, omit the five-iteration CV loop, not view training/cache generation.

### B7. Rebuild the 867-report retrieval bank and assemble runtime assets

```bash
"$BP" "$PT/evaluate_geometry_retrieval.py" --data-root "$FULL" \
  --manifest "$WORK/manifest/manifest.jsonl" --raw-data-root "$RAW_BITE" \
  --output-dir "$WORK/retrieval-index" --query-split all --database-split all --leave-one-out
"$BP" "$PT/package_retrieval_assets.py" --index "$WORK/retrieval-index/retrieval_index.npz" \
  --manifest "$WORK/manifest/manifest.jsonl" --raw-data-root "$RAW_BITE" --output-dir "$WORK/retrieval-assets"
python3 "$PT/package_hybrid_labels.py" --manifest "$WORK/manifest/manifest.jsonl" \
  --retrieval-reports "$WORK/retrieval-assets/retrieval_reports.json" --output "$WORK/retrieval-assets/retrieval_labels.json"
"$BP" scripts/prepare_reproduction.py assemble-model --output "$WORK/model" \
  --ptv3 "$FINAL_PT" --data "$FULL" --photo "$WORK/photo-full/model_final.pt" \
  --view "$VIEW" --normalizer "$NORMALIZER" --retrieval "$WORK/retrieval-assets"
```

Use **FULL**, not the unfiltered 991-point root. Expected index shape: (867,
3720), identical patient ordering in all three bank files. The final PTv3
config and last checkpoint are copied, not recalibrated. New training hashes
are recorded in `retrained_asset_hashes.json`; they need not equal the published
hashes. Run A2 with `BITE2TEXT_TEST_MODEL_ROOT="$WORK/model"` to smoke test the
retrained system using either the released image or A3's source-built image.

### B8. Recreate development OOF predictions and frozen v9 evaluation

This requires both PTv3 and photo CV checkpoints; do not use the full-867
checkpoints to claim OOF performance.

```bash
"$BP" "$PT/predict_cv_oof.py" --bits2bites-root "$BITS" --data-root "$CV" \
  --manifest "$WORK/manifest/manifest.jsonl" --raw-data-root "$RAW_BITE" \
  --output "$WORK/ptv3-oof.jsonl" --folds 1,2,3,4,5 --batch-size 8 --num-workers 4 \
  --device cuda --seed 20260816 --experiment-prefix bite2text_ptv3_v3_official_12head --base-seed 20260810
"$PP" "$PH/evaluate_photo_ptv3_fusion.py" --photo-cv-root "$WORK/photo-cv" \
  --ptv3-oof "$WORK/ptv3-oof.jsonl" --head-vocabs "$FULL/head_vocabs.json" --output-dir "$WORK/photo-oof"
"$PP" "$PH/evaluate_risk_aware_reranking.py" --ptv3-oof "$WORK/ptv3-oof.jsonl" \
  --photo-oof "$WORK/photo-oof/oof_probabilities.npz" --head-vocabs "$FULL/head_vocabs.json" \
  --retrieval-index "$WORK/retrieval-assets/retrieval_index.npz" \
  --retrieval-reports "$WORK/retrieval-assets/retrieval_reports.json" \
  --retrieval-labels "$WORK/retrieval-assets/retrieval_labels.json" --output-dir "$WORK/v9-oof" \
  --top-k 50 --geometry-lambda .5 --photo-lambda .2 --midline-threshold .45 --unsupported-gate 5 \
  --margins .02 --unsupported-penalties .005 --contradiction-thresholds .65 \
  --contradiction-gates .01 --contradiction-penalties .5 \
  --min-contradiction-improvement .015 --no-new-unsupported
```

The fusion utility also computes experimental cross-fit fusion; v9 reranking
uses the saved **photo probabilities**, not those fused probabilities. The
final controls are fixed above, not a fresh grid search. Inspect `summary.json`
and the emitted JSONL predictions. The script's `bleu_4_lite`/`meteor_lite` are
approximate offline metrics, **not** the organizer's BLEU/METEOR. To score with
an independently obtained official evaluator, first use
`prepare_official_text_eval.py --predictions-jsonl <selected JSONL> --output-root <new directory>`
to construct its input/ground_truth/output directories, then follow that
evaluator's version-specific instructions. The organizer evaluator is not
redistributed here. Historical official-evaluator development metrics were
BLEU-4 .2684 / METEOR .4700; compare metric definitions before comparing numbers.

## Evaluation limitations and honest claims

The five-fold split separates patients for **Bite2Text task-specific head/joint
training**, and the reranker excludes same-fold report candidates. It is not
fully nested, end-to-end independent validation: all folds share the all-200
Bits2Bites initialization, the view classifier is not refitted per outer fold,
descriptor scaling is fit on the available retrieval pool, and final scoring
reliability weights/settings were selected during development. No subject-level
non-overlap claim between Bits2Bites and Bite2Text is made here. Recreating this
historical protocol is distinct from designing a new strictly nested study.

Source tests and a container smoke run establish executability, not numerical
equality, scientific independence, clinical validity or complete retraining.
Never publish patient-level generated outputs just because source tests pass.
The validation record deliberately distinguishes tested steps from long GPU
training not repeated for this documentation change.
