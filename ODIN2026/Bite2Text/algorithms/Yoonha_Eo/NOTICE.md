# Licensing and provenance notices

The final2 release combines Bitemark inference logic, trained checkpoints and
third-party components. Bitemark-owned final2 code, calibrated parameters and
checkpoints are released under GNU AGPL-3.0 (see `LICENSE`). This is not a grant
of rights to third-party datasets, trademarks or patient records.

| Included component | Location | Upstream and license |
|---|---|---|
| Pointcept runtime and pointops | `final2/model/s/a/pointcept`, `submission/pointops` | [Pointcept](https://github.com/Pointcept/Pointcept), MIT, Copyright (c) 2023 Pointcept |
| 3DTeethLand / ToothInstanceNet runtime and pointops | `final2/model/s/b`, `submission/teethland_pointops` | [3DTeethLand](https://github.com/nnistelrooij/3dteethland), MIT, Copyright (c) 2026 Niels van Nistelrooij |
| PointGroup-derived CUDA operators | `submission/pointgroup_ops` | [PointGroup](https://github.com/dvlab-research/PointGroup), Apache-2.0; distributed through Pointcept |
| YOLO implementation, installed by Docker | `ultralytics==8.4.120` | [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 |
| Sonata code bundled within Pointcept (not the selected final2 checkpoint) | `final2/model/s/a/pointcept/models/sonata` | [Sonata](https://github.com/facebookresearch/sonata), Apache-2.0; no Sonata pretrained weights are distributed |
| Earlier Bitemark synthetic skeleton | `src/bitemark`, original demo tests/configs | Previously released under Apache-2.0; original grant retained in `LICENSES/Apache-2.0-legacy.txt` |

Copies of upstream notices are supplied in `LICENSES/` and the model bundle.
Existing source headers are retained. The vendored runtime reflects the final
submission snapshot, including its local adaptations; it is not claimed to be
identical to current upstream HEAD. Upstream commit IDs were not recorded in
the submission archive. Per-file hashes in the public model manifest identify
the distributed runtime exactly.

3DTeethLand's instance and landmark checkpoints were taken from the released
upstream models without Bite2Text fine-tuning. The upstream project publishes
checkpoint links in its [inference instructions](https://github.com/nnistelrooij/3dteethland#inference).
These are third-party artifacts, not Bitemark-owned training results. The upstream
MIT notice is retained. No exclusive ownership or independent relicensing
of those artifacts is asserted here.

The other seven checkpoints are the submission's view classifier, upper/lower
crowding classifiers, YOLO11m segmenter, and Spee/Wilson/overjet PTv3 classifiers.
Checkpoint bytes are unmodified. The YOLO checkpoint identifies AGPL-3.0 in its
metadata. Dependencies downloaded during Docker builds (including PyTorch, CUDA,
Open3D and spconv) retain their respective licenses. Their licenses are not
replaced by this repository's license. No prebuilt CUDA container is redistributed
by this source/model release.

Public-export changes: notebook outputs and example patient-ID comments removed;
default case changed to a synthetic identifier already overridden by the adapter;
unused case-level calibration records removed; model manifest regenerated.
Numerical inference code, required calibration values and trained weight bytes
are retained. Packaging and reproduction helpers are new AGPL-covered Bitemark code.

The `training/` Bitemark sources are also covered by GNU AGPL-3.0. Their recovered
source hashes and adaptations are documented in `training/provenance/sources.json`
and `training/README.md`. The earlier Wilson cohort profile is reconstructed from
archived results, not represented as an original archived trainer. Upstream
3DTeethLand training is linked separately with its original attribution and terms.
No dataset redistribution rights or patient-level training records are included.
