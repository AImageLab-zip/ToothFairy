# ODIN 2026 Bite2Text geometry-retrieval debugging v4

This is an independent debugging submission built on top of the validated v3
input/normalization image. It replaces the seven-head PTv3 output template with
a nearest-neighbour report from a 3,720-dimensional paired-jaw geometry index.

The model resource contains:

- `ios_normalizer_best.pt`
- `retrieval_index.npz`
- `retrieval_reports.json`

Before an online debugging submission, the image must pass all three known
debug-case container runs and the output reports must be checked with the
official Bite2Text evaluator layout.

