# Training and data provenance

This directory documents the training behind the nine checkpoints used by
`bite2text_submission_bitemark_final2.ipynb`, and provides the recovered Bitemark
training scripts. It supplements the unchanged [final2 inference release](../README.md).
The audit used the final2 weight manifest, checkpoint metadata, original training
scripts, split summaries and run results, retrieved on 2026-10-13.

## 📋 Checkpoint-to-training map

| Released checkpoint | Model / initialization | Training data | Training entrypoint |
|---|---|---|---|
| `w/view.pt` | EfficientNet-B3 / ImageNet | AI HUB oral images, five views | [train_view.py](iop/train_view.py) |
| `w/crowding-upper.pt` | EfficientNetV2-S / ImageNet | AI HUB upper occlusal images and complexity labels | [train_crowding.py](iop/train_crowding.py) |
| `w/crowding-lower.pt` | EfficientNetV2-S / ImageNet | AI HUB lower occlusal images and complexity labels | [train_crowding.py](iop/train_crowding.py) |
| `w/yolo11m.pt` | YOLO11m-seg / upstream pretrained checkpoint | AI HUB tooth polygons, 24 FDI classes | [prepare](iop/prepare_yolo_fdi24.py), [train](iop/train_yolo_fdi24.py) |
| `w/instseg.ckpt` | Released 3DTeethLand instance model | Upstream 3DTeethLand training, not Bitemark training | [Upstream training](#3dteethland-instance-and-landmark-models) |
| `w/landmarks.ckpt` | Released 3DTeethLand landmark model | Upstream 3DTeethLand training, not Bitemark training | [Upstream training](#3dteethland-instance-and-landmark-models) |
| `w/spee.pt` | PT-v3m1 Official Base / scratch | Bite2Text scans, report-derived curve labels | [train_curves.py](ios/train_curves.py), Spee profile |
| `w/wilson.pt` | PT-v3m1 Official Base / scratch | Bite2Text scans, report-derived curve labels, earlier cohort | [train_curves.py](ios/train_curves.py), reconstructed Wilson profile |
| `w/overjet.pt` | PT-v3m1 Official Base / scratch | 3D dental model dataset with pre/post-orthodontic treatment | [prepare](ios/prepare_overjet.py), [train](ios/train_overjet.py) |

**The final PTv3 checkpoints were trained from scratch, not fine-tuned from a
generic pretrained PTv3 checkpoint.** Overjet training used the orthodontic
STL dataset, not Bits2Bites. Bits2Bites' role in rule calibration / earlier
experiments should not be conflated with this checkpoint's training data.

## 🗂️ Data access and preparation

Obtain data under the respective providers' access conditions. No patient images,
meshes, reports, row-level split manifests or curated patient labels are included.

- **AI HUB:** [구강 이미지 합성데이터, dataset 71688](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71688).
  The local training assets were extracted images, `metadata_final_v20_1.json`
  (including `image_filepath` and `complexity`), and tooth-polygon annotation ZIPs.
  The English label “oral image synthesis dataset” refers to this resource, not
  an additional Bitemark-generated synthetic dataset.
- **Bite2Text:** obtain authorized scans and reports from the
  [challenge provider](https://odin2026.grand-challenge.org/). Curve labels came
  from the local report-derived, reviewed calibration table
  `curve_rulebase_threshold_calibration_20260715.csv`. The review helper below
  is not a substitute for that reviewed table.
- **Overjet:** Wang et al., [*A 3D dental model dataset with pre/post-orthodontic
  treatment for automatic tooth alignment*](https://www.nature.com/articles/s41597-024-04138-7),
  Scientific Data 11, 1277 (2024). Request the data through
  [Zenodo](https://zenodo.org/records/11392406) and follow the provider's DUA process.
  Use the pre-treatment `ori/U_Ori.stl`, `ori/L_Ori.stl` and corresponding JSON
  tooth labels, plus the provider's label files and official train/test lists.
- **3DTeethLand:** upstream data and checkpoint sources are linked from the
  [authors' repository](https://github.com/nnistelrooij/3dteethland).

Expected local layout (paths are examples, not files supplied here):

```text
/datasets/aihub/
  views/{right,front,left,upper,lower}/*.png
  crowding/{upper,lower}/...images...
  metadata.json
  mask_annotations/*.zip
/datasets/bite2text/
  <case>/ios/ios_upper.stl
  <case>/ios/ios_lower.stl
  curve_labels.csv
  segmentations/<case>_upper.json
  segmentations/<case>_lower.json
  ras_pointsets/<case>.json
/datasets/orthodontic/
  scans/<case>/ori/{U_Ori,L_Ori}.{stl,json}
  labels/{Dentition,Crowding,Malocclusion,Overbite,Overjet}.txt
  labels/train-test-split.txt
```

The view pool and crowding pool are **different subsets**. Crowding used the
locally combined source training/validation pool. Keep all original filenames:
the IOP trainers join labels by image basename. These recovered splits are
image-stratified, not verified patient-group-disjoint splits. Do not describe
them as patient-disjoint without an additional grouping audit.

## ⚙️ Training environment

Use the final2 CUDA environment and bundled Pointcept implementation. The
training extension below adds iterative stratification and the logging import
required by Pointcept; it does not upload training runs to W&B.

```bash
# Use current main for training sources (they are newer than v1.0.0-final2).
git clone https://github.com/yoonhaeo/bitemark.git
cd bitemark
docker build -t bitemark:final2 submission
docker build -t bitemark:training -f training/Dockerfile .
mkdir -p outputs
sudo chown 1000:1000 outputs

# Shell helper used in commands below. Replace /absolute/path/to/datasets.
train() {
  docker run --rm --gpus all --shm-size=16g \
    -v /absolute/path/to/datasets:/datasets:ro \
    -v "$PWD/outputs:/outputs" bitemark:training "$@"
}
```

Initial torchvision / YOLO pretrained downloads require network access. Supply
the official YOLO checkpoint under `/datasets/pretrained/yolo11m-seg.pt`.
The training Dockerfile is a portable environment recipe, not an independently
verified recreation of every original training environment. A full CUDA retrain
has not been executed as part of this publication.

## 🖼️ Photograph models

### View classifier

EfficientNet-B3 uses torchvision ImageNet initialization, 320-pixel input,
batch size 32 and dropout 0.45. The five labels are `right`, `front`, `left`,
`upper`, `lower`. Seed **20260812** is stored in the final checkpoint.
The 80/10/10 split is stratified by view × complexity (11,611 / 1,451 / 1,452).
The second split uses seed + 1.

The recovered recipe uses inverse-frequency replacement sampling, AdamW
(head LR 4e-4, backbone LR 1.2e-4, weight decay 5e-4), one head-only epoch,
label smoothing 0.06 and cosine decay to 1e-6. Augmentation includes resized
crops, color jitter, blur, rotation and erasing. Horizontal flips exchange left
and right labels. Validation macro-F1, then accuracy, selects `best.pt`.
The released B3 checkpoint records **epoch 2**.

```bash
train training/iop/train_view.py \
  --data-root /datasets/aihub/views --metadata /datasets/aihub/metadata.json \
  --output-dir /outputs/view --models efficientnet_b3 \
  --epochs 35 --patience 8 --freeze-epochs 1
```

### Upper and lower crowding

Two independent EfficientNetV2-S models use torchvision ImageNet initialization,
384-pixel input, batch size 32, dropout 0.25 and LR 2e-4. Raw complexity labels
1/2/3 map to checkpoint classes `none` / `mild` / `moderate`. Final2 report phrases
(`None to mild`, `Mild to moderate`, `Moderate to severe`) are downstream vocabulary
mapping, not the raw training label names.

The source recipe uses label-stratified 80/10/10 splits, seed 2026,
inverse-frequency replacement sampling, AdamW weight decay 6e-4, label smoothing
0.08 and per-step cosine decay. Augmentations include crops, color jitter, blur,
rotation and random erasing. Validation macro-F1 selects `best.pt`.

| View | Train | Validation | Test | Released best epoch |
|---|---:|---:|---:|---:|
| Upper | 24,682 | 3,085 | 3,086 | 13 |
| Lower | 33,464 | 4,183 | 4,183 | 33 |

```bash
train training/iop/train_crowding.py \
  --data-root /datasets/aihub/crowding --metadata /datasets/aihub/metadata.json \
  --output-dir /outputs/crowding --models efficientnet_v2_s --views upper lower \
  --epochs 40 --patience 8 --seed 2026
```

The original view/crowding launch commands were not recovered. Maximum epoch
budgets, patience and other CLI defaults above are the **recovered source recipe**,
not evidence that every original CLI flag was unchanged. Checkpoint fields and
archived split counts are recorded separately from these defaults.

### YOLO11m-seg: 24 FDI classes

The preparation script matches AI HUB images to polygon-annotation ZIPs,
normalizes polygon coordinates, simplifies contours to at most 256 points and
rejects degenerate polygons. Classes are 11–16, 21–26, 31–36 and 41–46.
It uses **85/15 train/validation per view**, seed 42 with view-specific seed
offsets. This recipe has no separate held-out test split.

Ultralytics 8.4.120 trains from `yolo11m-seg.pt`: 160 epochs, patience 30,
image size 768, batch 32, AdamW LR 0.001, final LR fraction 0.01, weight decay
0.0005, three warmup epochs, cosine scheduling and AMP. It uses mild affine /
perspective augmentation, mosaic 0.5 (closed for the final 15 epochs), and no
horizontal/vertical flips, mixup or copy-paste. The final file is the run's
validation-selected `weights/best.pt`, not an untrained YOLO checkpoint.

```bash
train training/iop/prepare_yolo_fdi24.py \
  --images /datasets/aihub/views --labels /datasets/aihub/mask_annotations \
  --project-root /outputs/yolo --seed 42 --val-fraction 0.15
train training/iop/train_yolo_fdi24.py \
  --project-root /outputs/yolo --pretrained /datasets/pretrained/yolo11m-seg.pt
```

Preparation requires matching images/annotations and refuses to delete an
existing dataset directory. Use a fresh output root for a new preparation run.

## 🦷 Scan models

### 3DTeethLand instance and landmark models

Bitemark uses the upstream `instseg_full.ckpt` and `landmarks_full.ckpt` without
Bite2Text fine-tuning. They become `instseg.ckpt` / `landmarks.ckpt` in final2.
These are **not newly trained Bitemark components**. The released checkpoint
metadata records epoch 499 (zero-based), with 112,000 and 59,000 global steps,
respectively, using PyTorch Lightning 2.3.3. It does not contain enough information
to establish the exact upstream training cohort/split or complete launch config.

For upstream training, use the authors' [training entrypoint](https://github.com/nnistelrooij/3dteethland/blob/424252e3d94a1565c8c2090eb5bb456b76386b93/train.py)
and [configuration](https://github.com/nnistelrooij/3dteethland/blob/424252e3d94a1565c8c2090eb5bb456b76386b93/teethland/config/config.yaml).
This is a pinned documentation reference, **not a claim that this was the commit
used to train the released weights**. In the upstream environment, after configuring
authorized data, annotation paths and stage settings:

```bash
python train.py instseg --devices 1
python train.py landmarks --devices 1
```

The referenced `instseg` branch uses `DentalNet.load_from_checkpoint`; configure
its initialization checkpoint first. These commands expose upstream training,
but are not a from-scratch reconstruction of the authors' released checkpoints.
Use the released weight hashes for exact Bitemark inference reproduction.

### Spee and Wilson: shared tooth-mask representation

The trainer selects tooth vertices using 3DTeethLand FDI/instance masks, excludes
non-tooth surfaces and samples **12,288 points approximately equally across
detected teeth**. Both arches share a RAS transform. Features have 20 dimensions:
coordinates 3, normals 3, arch one-hot 2, tooth-position one-hot 8, quadrant one-hot 4.
Metric scale is preserved; grid size is 0.4 mm.

Both final heads are binary `Normal` / `Increased`, trained independently from
scratch. The recipe uses AdamW (LR 1e-4, weight decay 0.01), weighted cross-entropy,
label smoothing 0.03, cosine decay, batch 8, gradient clipping 1.0, maximum 400
epochs and patience 75. Online augmentation uses scale 0.98–1.02, Z rotation
±0.08 rad, translation ±0.5 mm and optional 10% point dropout.

| Final model | Cohort and joint label stratification | Train / val / test | Selected checkpoint |
|---|---|---|---|
| Spee | 956 cases, excludes Spee `Reversed`; 41 exclusions | 764 / 96 / 96 | `best_macro_f1`, epoch 52, run 369367253595 |
| Wilson | 976 cases, includes Spee `Reversed` in split strata; 21 exclusions | 780 / 98 / 98 | `best_accuracy`, epoch 74, run 369367253578 |

Both use seed 42 and 80/10/10 joint Spee × Wilson stratification, but **not the same
case cohort**. The earlier Wilson script snapshot was not recovered. The public
`wilson-final2` switch reconstructs the documented class-filter difference using
the archived result and obsolete-Spee marker; it is not a byte-identical original
Wilson training script. Separate split/output directories prevent mixing profiles.

Required curve CSV columns are `Patient`, `Curve of Spee`, `Curve of Wilson Degree`.
Masks contain per-vertex `labels` and `instances`, in the compacted mesh vertex
order. Each RAS JSON contains `transform.homogeneous_original_to_ras` (4×4).
Do not independently align the upper and lower arches.

The [report review helper](labels/build_curve_review.py) extracts candidate labels
and conflict/evidence fields from authorized reports:

```bash
train training/labels/build_curve_review.py \
  --dataset-root /datasets/bite2text --output /outputs/curve_review.csv
```

Review conflicts and map `case_id`, `curve_of_spee`, `curve_of_wilson_degree` to
the required trainer headers (inspect the helper's output header first).
This procedure creates a new review table; it does **not** recover the original
curated labels automatically. Exact historical cohort reproduction requires
the original reviewed table and the same segmentation/RAS outputs.

Generate segmentations/landmarks with upstream inference or the supplied final2
runtime. The [RAS builder](../final2/model/q/build_ras_landmark_pointsets.py)
provides `--data-root`, `--prediction-dir` and `--output-root` for this intermediate
representation. Inspect its expected prediction filenames before staging data.

```bash
train training/ios/train_curves.py --task spee --split-profile spee-final2 \
  --dataset-root /datasets/bite2text --labels-csv /datasets/bite2text/curve_labels.csv \
  --segmentation-root /datasets/bite2text/segmentations \
  --pointset-root /datasets/bite2text/ras_pointsets \
  --cache-dir /outputs/spee/cache --split-dir /outputs/spee/splits \
  --output-root /outputs/spee/models --seed 42
train training/ios/train_curves.py --task wilson --split-profile wilson-final2 \
  --dataset-root /datasets/bite2text --labels-csv /datasets/bite2text/curve_labels.csv \
  --segmentation-root /datasets/bite2text/segmentations \
  --pointset-root /datasets/bite2text/ras_pointsets \
  --cache-dir /outputs/wilson/cache --split-dir /outputs/wilson/splits \
  --output-root /outputs/wilson/models --seed 42
```

### Overjet: raw dual-arch PTv3

Pre-treatment STL triangles are sampled proportional to area, **4,096 points per
arch**. FDI crown centroids define a common RAS frame, but FDI labels are not model
features. Coordinates, normals and arch one-hot yield 8 features per point.
The combined cloud is centered and normalized by maximum radius.

Preparation retains the dataset's official 250-case test split and divides the
810 official-training cases into 648 training / 162 validation cases with
multi-label stratification (seed 42, 80/20 within official training). The script
records reconciliation of source split IDs against the available STL cases.
This is **not an overall 80/10/10 split**. All five source label files are required
for the joint stratification, even when training only overjet.

The final reduced-label task excludes edge-to-edge labels, leaving `Normal`,
`Deep`, `Negative` classes. There are 621 training and 155 validation cases after
filtering. Balanced replacement augmentation uses 296 examples per class per
epoch (888 total), with uniform CE weights. PT-v3m1 Official Base is initialized
from scratch, using depths (2,2,2,6,2), channels (32,64,128,256,512), heads
(2,4,8,16,32), and patch size 48 at each stage.

The run uses AdamW LR 1e-4, weight decay 0.01, cosine scheduling, batch 8,
gradient clipping 1.0, maximum 500 epochs and patience 100. Augmentation includes
axis-wise scale 0.95–1.05, translation ±0.02, Z rotation ±18°, clipped coordinate
jitter and optional point dropout. Grid size is 0.01 in normalized coordinates.
Final2 uses **best validation accuracy**, epoch 122, run 369367248013.

```bash
train training/ios/prepare_overjet.py \
  --dataset-root /datasets/orthodontic/scans --label-root /datasets/orthodontic/labels \
  --output-root /outputs/overjet/prepared --points-per-arch 4096 --seed 42
train training/ios/train_overjet.py \
  --prepared-root /outputs/overjet/prepared --output-root /outputs/overjet/models \
  --input-mode raw_stl --tasks overjet --balanced-augmentation \
  --epochs 500 --patience 100 --batch-size 8 --eval-batch-size 8 --seed 42
```

## 🔎 Evidence and reproduction limits

- [Source provenance](provenance/sources.json) records the retrieved source hashes
  and original relative training paths. Public adaptations replace private defaults,
  add YOLO path arguments, refuse destructive overwrite, and add the documented
  Wilson cohort switch. Numerical training functions otherwise follow recovered code.
- [Aggregate IOP split counts](provenance/split_counts.json) were computed from
  archived split manifests; patient-level manifests are not redistributed.
- [Checkpoint facts](provenance/checkpoints.json) distinguish recovered Bitemark
  training, reconstructed Wilson settings and upstream checkpoint reuse.
- [Released weight hashes](../release/weights.json) identify the actual inference
  checkpoints. Training a new model does not overwrite those artifacts automatically.
- Seeds reproduce splitting only with the same ordered inputs and labels. The
  original curve trainer uses an unseeded `np.random.default_rng()` for online
  augmentation, and CUDA kernels can be nondeterministic. Bitwise-identical
  retraining is not promised.
- The checked-in training scripts pass syntax / structural tests. A complete
  GPU retraining and comparison against released checkpoints has not been run.
- Rule-based molar/canine/anterior/transverse decisions are not additional neural
  checkpoints. Their inference-consumed calibrated parameters are in the existing
  final2 sources; this directory does not claim to recreate all historical rule
  calibration searches.

Bitemark-owned training sources are provided under the repository's GNU AGPL-3.0
terms. Upstream code, checkpoints and datasets retain their respective terms and
attribution; see [NOTICE.md](../NOTICE.md).
