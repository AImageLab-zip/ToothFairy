# EviDent-CBCT

**Evidence-Bottlenecked Report Generation from Dental CBCT under Non-Exhaustive Report Supervision**

Ruiyang Hao, Zhi Qin Tan, Yulan He, Owen Addison, and Yunpeng Li

[![arXiv](https://img.shields.io/badge/arXiv-2610.02375-b31b1b.svg)](https://arxiv.org/abs/2610.02375)

[Paper](https://arxiv.org/abs/2610.02375) · [Challenge results](#challenge-results) · [Getting started](#installation) · [Roadmap](#roadmap) · [Citation](#citation)

## Overview

EviDent-CBCT is a framework for generating dental reports from three-dimensional
cone-beam computed tomography (CBCT). It addresses report generation under
non-exhaustive report supervision, where a reference report may not describe
every finding present in a scan.

The framework uses an explicit evidence bottleneck between image analysis and
report generation. Anatomy segmentation guides three evidence-prediction
branches, and their structured predictions are filtered and checked for dental
consistency before being rendered into a report. An optional image-blind
language-model rewrite refines the report wording.

This repository accompanies the [paper](https://arxiv.org/abs/2610.02375) and
currently provides the model architectures and end-to-end inference pipeline.
Training code will be released in a future update.

## Challenge results

EviDent-CBCT achieved the following results in the **ODIN 2026 ToothFairy4
Challenge**:

| Evaluation                | Rank          |
| ------------------------- | ------------- |
| Automated evaluation      | **2nd place** |
| Clinical arena evaluation | **3rd place** |

See the [official challenge winners](https://odin2026.grand-challenge.org/challenge-winners/) for the results.

## Method

The framework separates anatomical analysis, evidence prediction, and report
generation into three stages:

1. **Anatomy segmentation.** A U-Mamba2 backend segments the input CBCT volume
   and supports the construction of anatomy-aware inputs for evidence prediction.
2. **Structured evidence prediction.** Three branches predict global,
   tooth-level, and relation-level evidence. Fixed inference filters and dental
   consistency rules process the predictions before report generation.
3. **Evidence-to-report generation.** A deterministic renderer converts the
   processed evidence into a report. An optional Qwen LoRA rewriter operates
   without access to the image to refine the text.

```text
CBCT -> anatomy segmentation -> three evidence-prediction branches
     -> fixed inference filtering/consistency rules -> deterministic report
     -> optional image-blind LoRA rewrite
```

Inference outputs include the report, structured evidence before and after
post-processing, and filtering and consistency-projection audit files, allowing
users to inspect the intermediate evidence used to construct the report.

## Release status

The current release includes model definitions, single-case and directory
inference, the challenge adapter, frozen inference configuration, deterministic
report rendering, and optional language-model rewriting.

Training code, reference-evidence construction, and evaluation and ablation
tools are not included in this release. Datasets, clinical reports, and model
weights are not bundled with the repository. Running inference requires the
model artifacts described in [Model artifacts](#model-artifacts).

## Repository layout

```text
evigre/
  networks/      released model architectures
  dataloading/   inference tensor loaders
  inference/     end-to-end inference and challenge adapter
  stage3/        deterministic renderer and optional LoRA rewrite
  configs/       frozen challenge inference configuration
docker/          weight-free runtime image and interface checks
docs/            inference, model-artifact, dependency, and Docker notes
scripts/         external dependency helper and release check
```

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are required. Stage 1 uses
a pinned U-Mamba2 dependency:

```bash
git clone <repository-url> EviDent-CBCT
cd EviDent-CBCT
bash scripts/fetch_umamba2.sh
python -m pip install -e .
```

See [Dependencies](docs/DEPENDENCIES.md) for dependency setup details.

## Model artifacts

Model artifacts are not stored in this source repository. Place the separately
provided artifacts according to [Model artifacts](docs/MODELS.md), or set `EVIDENT_MODELS` to a
compatible artifact root.

## Inference

Create a local configuration from the shipped example and run:

```bash
cp evigre/configs/inference/evident_cbct.example.json local_inference.json
python -m evigre.inference INPUT_NIFTI_OR_DIRECTORY OUTPUT_DIRECTORY \
  --config local_inference.json --clean-work
```

Single-case output contains the final report plus structured evidence and an
audit trail of inference-only filtering/projection steps. Directory mode writes
one result directory per volume and a `reports.csv` summary.

To check CUDA, configuration, and model paths before inference, add
`--preflight-only`. To use the deterministic report without the optional
language-model rewrite, add `--skip-rewrite`.

See [Inference](docs/INFERENCE.md) for details.

## Docker

The Docker source contains code and dependencies only. Model artifacts are
mounted read-only at runtime. See [Docker](docs/DOCKER.md).

## Scope

This is research software for challenge reproducibility. Generated reports
require clinical review and must not be used as autonomous diagnostic advice.

## Roadmap

- [x] Release model architectures and the end-to-end inference pipeline.
- [x] Provide inference configuration and Docker deployment instructions.
- [ ] Release training code in a future update.

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
You may share and adapt the material for non-commercial purposes, provided you
give appropriate credit, include a link to the license, and indicate any changes.
See [LICENSE](LICENSE) for the full terms. Third-party components remain subject
to their respective licenses.

## Citation

If you find our method, code, or results useful for your research, please cite our paper:

```bibtex
@misc{hao2026evidentcbct,
  title         = {EviDent-CBCT: Evidence-Bottlenecked Report Generation from Dental CBCT under Non-Exhaustive Report Supervision},
  author        = {Ruiyang Hao and Zhi Qin Tan and Yulan He and Owen Addison and Yunpeng Li},
  year          = {2026},
  eprint        = {2610.02375},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  note          = {Preprint},
  url           = {https://arxiv.org/abs/2610.02375}
}
```
