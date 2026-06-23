# Bite2Text Algorithm And Evaluation Pipeline

Run the complete local pipeline:

```bash
./run_full_pipeline.sh
```

The script does the following:

1. Copies three case samples, each with three modalities (lower IOS STL, upper IOS STL, and intraoral TIFF):

```text
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5535/ios/ios_lower.stl -> algorithm/test/input/samples/F5535/files/ios-lower/ios_lower.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5535/ios/ios_upper.stl -> algorithm/test/input/samples/F5535/files/ios-upper/ios_upper.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5535/intraoral-photo.tiff -> algorithm/test/input/samples/F5535/images/intraoral-photo/intraoral-photo.tiff
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5520/ios/ios_lower.stl -> algorithm/test/input/samples/F5520/files/ios-lower/ios_lower.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5520/ios/ios_upper.stl -> algorithm/test/input/samples/F5520/files/ios-upper/ios_upper.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5520/intraoral-photo.tiff -> algorithm/test/input/samples/F5520/images/intraoral-photo/intraoral-photo.tiff
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5405/ios/ios_lower.stl -> algorithm/test/input/samples/F5405/files/ios-lower/ios_lower.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5405/ios/ios_upper.stl -> algorithm/test/input/samples/F5405/files/ios-upper/ios_upper.stl
/mnt/c/Users/Kevin/Downloads/Bite2Text/F5405/intraoral-photo.tiff -> algorithm/test/input/samples/F5405/images/intraoral-photo/intraoral-photo.tiff
```

2. Builds the example algorithm Docker image.

3. Builds the evaluation Docker image.

4. Runs the algorithm once per case with input sockets `3d-lower-teeth-scan`, `3d-upper-teeth-scan`, and `2d-intraoral-photographs`.

5. Writes a Grand Challenge-style `predictions.json` using output socket `diagnostic-imaging-report`.

6. Extracts `evaluation/evaluationgroundtruth.tar.gz` to simulate `/opt/ml/input/data/ground_truth/`.

7. Runs evaluation on the algorithm outputs.

The split between `files/ios-lower`, `files/ios-upper`, and `images/intraoral-photo` is intentional. It mirrors the three Grand Challenge input sockets, keeps each modality unambiguous, and matches the container's declared input schema.

The final metrics are written to:

```text
pipeline_work/evaluation_output/metrics.json
```

`pipeline_work/` is created by the local pipeline run, is safe to delete, and is already ignored by Git.

To use a different raw data location:

```bash
RAW_DATA_DIR=/path/to/bite2text---raw ./run_full_pipeline.sh
```

Current defaults in `run_full_pipeline.sh`:

```text
RAW_DATA_DIR=/mnt/c/Users/Kevin/Downloads/Bite2Text
CASES=(F5535 F5520 F5405)
```
