# ODIN 2026 Bite2Text PTv3 debugging submission

This directory packages the fixed stage-2 epoch-49 seven-head PTv3 model and
the paired IOS-Normalizer into one Grand Challenge algorithm image.

The model resource is mounted at `/opt/ml/model` and contains `config.py`,
`model_best.pth`, `head_vocabs.json`, and `ios_normalizer_best.pt`. The image
reads the official `/input/3d-lower-teeth-scan.obj` and
`/input/3d-upper-teeth-scan.obj` sockets. Grand Challenge can mount an uploaded
binary STL at these configured `.obj` paths, so the loader validates both OBJ
and STL parsers against the file contents instead of trusting the suffix. It
then stages genuine STL meshes for IOS-Normalizer and writes exactly
`/output/diagnostic-imaging-report.json`. Legacy STL layouts remain supported
for local compatibility.

`torch.segment_reduce` replaces the equivalent `torch-scatter.segment_csr`
operations because torch-scatter has no PyTorch 2.9 binary. Equality of the
four reductions used by the model was checked against the training runtime.

Build and test:

```bash
./do_build.sh
./do_test_run.sh
./do_save.sh
```
