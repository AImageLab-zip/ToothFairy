# Third-party code and models

This repository contains original work plus files derived from the projects
below. Where a file is derived, its header says so.

## U-Mamba2 — CC BY-NC 4.0

<https://github.com/zhiqin1998/U-Mamba2>

The segmentation stage is U-Mamba2, the ToothFairy3 winner, trained within
nnU-Netv2. **The framework itself is not vendored here.** It is licensed
Attribution-NonCommercial 4.0 International, and redistributing it inside this
repository would mix licences without benefit; clone it at build time instead:

```bash
git clone https://github.com/zhiqin1998/U-Mamba2 umamba2
```

One file in this repository is derived from it:

| file | origin | modification |
|---|---|---|
| `code/pipeline/segmentation/task1_inference.py` | `documentation/competitions/Toothfairy3/task1_inference.py` | sliding-window logits accumulator and per-class resample/argmax moved to CPU, so inference memory is bounded by patch size rather than volume size |

## nnU-Net — CC BY-NC 4.0

<https://github.com/MIC-DKFZ/nnUNet>

U-Mamba2 is built on nnU-Netv2 and inherits its licence. Reached only as a
dependency of the above; no nnU-Net source is redistributed here.

## Mamba / causal-conv1d

<https://github.com/state-spaces/mamba> and
<https://github.com/Dao-AILab/causal-conv1d>

CUDA kernels required by the U-Mamba2 architecture. Installed from the
projects' own prebuilt wheels at image-build time; not redistributed here.

## Qwen3.5-9B

The report stage serves a fine-tuned, AWQ-quantised Qwen3.5-9B. The base
model's licence governs the derived weights, which are distributed separately
from this repository.

## NonCommercial notice

Because the segmentation stage depends on CC BY-NC 4.0 code, the pipeline **as
a whole** is usable for research and other non-commercial purposes only. Note
that CC BY-NC has no share-alike term: the original code in this repository is
not obliged to adopt that licence, and is covered by the repository's own
LICENSE.
