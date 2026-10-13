# Third-party components

This repository is released under the GNU AGPL-3.0 (see `LICENSE`). It includes
or depends on the following third-party work, each under its own terms.

| component | where | source | license |
| --- | --- | --- | --- |
| landmarking and occlusal-measurement extraction | `bite2text/landmarks.py`, `bite2text/features.py` | Shadman Yazdi et al., *Automated Orthodontic Diagnosis via Geometric Deep Learning: A Clinically Interpretable Framework*, 2026 — [github.com/Ali5hadman/ortho-landmark-dgcnn](https://github.com/Ali5hadman/ortho-landmark-dgcnn) | MIT |
| TSGCNet network definition | `bite2text/nets/tsgcnet.py` | Zhang et al., *TSGCNet: Discriminative Geometric Feature Learning with Two-Stream Graph Convolutional Network for 3D Dental Model Segmentation*, CVPR 2021 — [github.com/ZhangLingMing1/TSGCNet](https://github.com/ZhangLingMing1/TSGCNet) | no license file upstream |
| PointMLP landmark network | `bite2text/nets/point_mlp.py` | Dou et al., *CLIK-Diffusion: Clinical Knowledge-informed Diffusion Model for Tooth Alignment*, MedIA 2025 — [github.com/ShanghaiTech-IMPACT/CLIK-Diffusion](https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion) | Apache-2.0 |
| CLIK crown-landmark checkpoints | `weights/clik/` (downloaded, not redistributed) | CLIK-Diffusion authors' Google Drive, linked from their README | see upstream |
| BLEU-4 / METEOR-lite metric code | `scripts/score.py` | ODIN 2026 Bite2Text evaluation — [github.com/AImageLab-zip/ToothFairy](https://github.com/AImageLab-zip/ToothFairy) | MIT |
| ResNet-18 architecture | `bite2text/nets/photo_cnn.py` (via torchvision) | He et al., CVPR 2016; torchvision | BSD-3-Clause |

## Data derived from Bites2Text

The following files are derived from the Bites2Text dataset (Lumetti et al.,
ECCV 2026), which is licensed **CC BY-NC-SA 4.0**. They are distributed under
the same license, for non-commercial use, with attribution:

| file | derived from |
| --- | --- |
| `bite2text/data/templates_occlusal.json`, `bite2text/data/templates_photo.json` | sentences mined from training-split reports |
| `bite2text/data/photo_prior.json` | findings selected against training-split reports |
| `bite2text/data/occlusal_train_fit.csv`, `bite2text/data/occlusal_train_dev.csv` | landmark measurements and report-extracted labels, training split only |
| `weights/tsgcnet/`, `weights/photo_cnn/` | model weights (the photo CNN was trained on training-split photographs) |
| `sample_data/` | one held-out case — see `sample_data/README.md` |

> L. Lumetti, F. Rizzo, F. Cremonini, E. Candeloro, L. Lombardo, C. Grana, F. Bolelli.
> *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines.*
> European Conference on Computer Vision (ECCV), 2026.
