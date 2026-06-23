# ToothFairy4 Example Algorithm

This is a minimal Grand Challenge algorithm template.

Input socket:

```text
cbct-image
```

Expected local input path inside the container:

```text
/input/images/cbct/*.mha
```

Output socket:

```text
diagnostic-imaging-report
```

Expected output path:

```text
/output/diagnostic-imaging-report.json
```

Output JSON:

```json
{"report": "Generated report text."}
```

Local test:

```bash
./do_test_run.sh
```

Full pipeline test data:

```text
test/input/samples/A003/images/cbct/A003.mha
test/input/samples/F001/images/cbct/F001.mha
test/input/samples/P001/images/cbct/P001.mha
```

These `.mha` files are generated from `/home/llumetti/Downloads/toothfairy4-cbct---raw/<case>/cbct/volume.nii.gz` by the root `run_full_pipeline.sh` script.

Save for upload:

```bash
./do_save.sh
```

The placeholder model is implemented in `run_model()` in `inference.py`. Replace that function with the real model logic.
