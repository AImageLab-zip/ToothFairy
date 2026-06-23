# Bite2Text Example Algorithm

This is a minimal Grand Challenge algorithm template.

Input sockets:

```text
3d-lower-teeth-scan
3d-upper-teeth-scan
2d-intraoral-photographs
```

Expected local input paths inside the container:

```text
/input/files/ios-lower/*.stl
/input/files/ios-upper/*.stl
/input/images/intraoral-photo/*.tiff
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
test/input/samples/F5535/files/ios-lower/ios_lower.stl
test/input/samples/F5535/files/ios-upper/ios_upper.stl
test/input/samples/F5535/images/intraoral-photo/intraoral-photo.tiff
test/input/samples/F5520/files/ios-lower/ios_lower.stl
test/input/samples/F5520/files/ios-upper/ios_upper.stl
test/input/samples/F5520/images/intraoral-photo/intraoral-photo.tiff
test/input/samples/F5405/files/ios-lower/ios_lower.stl
test/input/samples/F5405/files/ios-upper/ios_upper.stl
test/input/samples/F5405/images/intraoral-photo/intraoral-photo.tiff
```

These sample files are copied from `/mnt/c/Users/Kevin/Downloads/Bite2Text/<case>/ios/` for the STL scans and from `/mnt/c/Users/Kevin/Downloads/Bite2Text/<case>/` for `intraoral-photo.tiff` by the root `run_full_pipeline.sh` script.

The root pipeline writes its runtime data to `pipeline_work/` in the repository root. That directory is generated locally, safe to delete, and already ignored by Git.

The separate `files/ios-lower`, `files/ios-upper`, and `images/intraoral-photo` folders are intentional. They keep the three modalities aligned with the three Grand Challenge sockets and make the input manifest explicit for each case.

Save for upload:

```bash
./do_save.sh
```

The placeholder model is implemented in `run_model()` in `inference.py`. Replace that function with the real model logic.
