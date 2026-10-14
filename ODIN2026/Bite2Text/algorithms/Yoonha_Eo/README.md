# Bitemark — Bite2Text Challenge( final2 )Training and Inference

Bitemark generates structured orthodontic reports from intraoral photographs (IOPs)
and paired intraoral scans (IOSs). This release contains the **actual final2
test-phase inference code, calibrated rules, and all nine checkpoint files**.


**[Download final2 models](https://github.com/yoonhaeo/bitemark/releases/tag/v1.0.0-final2)**
 · [License and third-party notices](NOTICE.md)
 · [Checkpoint manifest](release/weights.json)
 · [Release provenance](release/provenance.json)
 · [Training code, datasets and recipes](training/README.md)

## Included components

| Component | Final2 implementation | Checkpoint in model bundle |
|---|---|---|
| Photograph view assignment | EfficientNet-B3, five views | `w/view.pt` |
| Upper / lower crowding | Two EfficientNetV2-S classifiers | `w/crowding-upper.pt`, `w/crowding-lower.pt` |
| 2D FDI tooth segmentation | YOLO11m-seg, used for frontal midline geometry | `w/yolo11m.pt` |
| 3D tooth instances / FDI labels | Released 3DTeethLand model | `w/instseg.ckpt` |
| 3D landmarks | Released 3DTeethLand model | `w/landmarks.ckpt` |
| Curve of Spee / Wilson | Separate PTv3 classifiers, shared FDI cloud | `w/spee.pt`, `w/wilson.pt` |
| Overjet | PTv3 classifier, raw dual-arch cloud | `w/overjet.pt` |
| Molar, canine, anterior and transverse relationships | Segmented geometry, calibrated rules and canine gate | `final2/pipeline.py`, `final2/model/u/` |
| Missing teeth / report assembly | FDI presence and deterministic sentence generation | `final2/pipeline.py` |

The nine checkpoints are byte-for-byte identical to those in the final2 model
archive. SHA-256 hashes are in `release/weights.json`. No random replacement
weights or placeholder thresholds are used by the final2 entrypoint.

The shared FDI cloud has 12,288 points with 20-dimensional features for Spee/Wilson.
Overjet instead uses 4,096 area-sampled points per arch (8,192 total) with
8-dimensional features. It is not a geometric overjet threshold classifier.

## Reproduce inference

Use Linux/amd64 (or WSL2 with GPU-enabled Docker), an NVIDIA CUDA-compatible GPU,
Docker and NVIDIA Container Toolkit. The submitted build targets compute
capabilities 7.5 and 8.6. Other GPU architectures need an appropriate extension
build configuration and have not been verified by this release.

Allow several GB for the extracted model bundle and substantially more for the
CUDA build images. The final2 container uses PyTorch 2.7.0 / CUDA 12.6, and builds
both Pointcept and 3DTeethLand CUDA extensions from the included source.
It refuses to skip trained components or silently use a CPU fallback.

```bash
git clone https://github.com/yoonhaeo/bitemark.git
cd bitemark
git checkout v1.0.0-final2

# Python 3.10+; downloader uses only the standard library.
python tools/download_models.py
python tools/download_models.py --verify-only

# Use this build context, not the legacy demonstration Dockerfile.
docker build -t bitemark:final2 submission
```

Stage one case using paired upper/lower meshes in their acquired bite relationship
and its photographs. Do not align or scale the two arches independently.
Supply the full available photograph set, ideally the five standard views.

```bash
python tools/prepare_input.py \
  --upper /path/to/ios_upper.stl \
  --lower /path/to/ios_lower.stl \
  --photos /path/to/photos/*.jpg \
  --output ./input

mkdir -p output
# The container runs as UID 1000; make output writable for that UID.
sudo chown 1000:1000 output
docker run --rm --gpus all --network none --shm-size=8g \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  -v "$PWD/models/final2:/opt/ml/model:ro" \
  bitemark:final2
cat output/diagnostic-imaging-report.json
```

The output socket contains `{"report": "..."}`. Existing Grand Challenge inputs
with `inputs.json` can be mounted directly instead of using `prepare_input.py`.
The adapter supports STL/OBJ scans and MHA, TIFF, JPEG, PNG or WebP photographs.
Run a separate container invocation for each case.

To install an archive downloaded in a browser:

```bash
python tools/download_models.py --archive /path/to/bitemark-final2-model.tar.gz
```

## Training

The [training guide](training/README.md) maps every final2 checkpoint to its data
source, initialization, split, training recipe and checkpoint selection. It includes
the recovered view, crowding, YOLO segmentation and PTv3 training scripts, data
preparation tools, and upstream 3DTeethLand training references.

The final PTv3 models were trained from scratch. The earlier Wilson cohort profile
is explicitly marked as reconstructed. Original reviewed labels and authorized
datasets are required for historical training reproduction and are not redistributed.
The published final2 weights and inference sources remain unchanged.

## Source layout and reproducibility boundaries

- `submission/`: original final2 Grand Challenge adapter, Dockerfile, pinned
  dependencies and CUDA extension sources.
- `final2/pipeline.py`: executable inference cell extracted from the final2
  notebook. The release builder wraps this source in an output-free notebook
  because the submitted adapter loads a notebook cell.
- `final2/model/`: bundled Pointcept / 3DTeethLand runtime, configuration,
  tooth-ROI helpers and actual canine calibration assets.
- `tools/`: checksum-verified model download, bundle reconstruction and input staging.
- `release/`: original checkpoint hashes, public bundle integrity and provenance.
- `training/`: recovered trainers, data preparation, per-weight provenance and
  instructions, with documented reconstruction limits.
- `src/bitemark`, `configs/public_placeholder.yaml`, `Dockerfile`: **legacy synthetic
  demonstration only**, not final2 inference. See `docs/LEGACY.md`.

The original adapter and trained checkpoint bytes are preserved. The public
notebook excludes outputs and example patient identifiers. Unused calibration
evaluation records and case lists are omitted, while every inference-consumed
value is retained. The adapter already substitutes a temporary case identifier
and runtime paths. No external prediction cache, patient report or ground-truth
file is needed at inference time.

The tagged `v1.0.0-final2` asset is the **inference reproduction release**;
current `main` additionally contains the training sources above.
Obtain evaluation inputs through their
authorized providers. Without the organizers' hidden inputs and evaluator, this
repository cannot independently reproduce or verify a leaderboard score.

Release checks cover source parity, checkpoint hashes, bundle integrity, input
staging and CPU unit tests. **A fresh end-to-end CUDA execution of this public
repackaging has not been verified.** The Dockerfile and pinned requirements are
from the submitted package, but upstream package availability, base-image updates,
GPU kernels and hardware can affect rebuilding and numerical reproducibility.


## License

Bitemark's combined final2 inference release and Bitemark-owned checkpoints are
provided under **GNU AGPL-3.0**. Preserve the license and corresponding source
when distributing covered works, and satisfy the AGPL source-offer requirement
when applicable to modified network-served versions. 

Third-party components retain their own notices and applicable licenses.
3DTeethLand checkpoints are credited to their upstream authors, not claimed as
Bitemark-trained or relicensed independently. See [NOTICE.md](NOTICE.md) and
[LICENSES/](LICENSES/). Dataset access and usage terms remain separate.

Research use only. This software is not a validated clinical diagnostic device.

## Development checks

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
python tools/check_public_release.py
```

To reconstruct the archive from public sources and the nine exact checkpoints:

```bash
python tools/build_final2_bundle.py --weights /path/to/w --output /path/to/new-model.tar.gz
```

Archive hashes can differ due to gzip metadata. Individual model and weight
hashes remain verifiable.
