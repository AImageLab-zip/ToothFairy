# ToothFairy4 Algorithm And Evaluation Pipeline

Run the complete local pipeline:

```bash
./run_full_pipeline.sh
```

The script does the following:

1. Converts the three raw CBCT samples from NIfTI to MHA using SimpleITK:

```text
/home/llumetti/Downloads/toothfairy4-cbct---raw/A003/cbct/volume.nii.gz -> algorithm/test/input/samples/A003/images/cbct/A003.mha
/home/llumetti/Downloads/toothfairy4-cbct---raw/F001/cbct/volume.nii.gz -> algorithm/test/input/samples/F001/images/cbct/F001.mha
/home/llumetti/Downloads/toothfairy4-cbct---raw/P001/cbct/volume.nii.gz -> algorithm/test/input/samples/P001/images/cbct/P001.mha
```

2. Builds the example algorithm Docker image.

3. Builds the evaluation Docker image.

4. Runs the algorithm once per case with input socket `cbct-image`.

5. Writes a Grand Challenge-style `predictions.json` using output socket `diagnostic-imaging-report`.

6. Extracts `evaluation/evaluationgroundtruth.tar.gz` to simulate `/opt/ml/input/data/ground_truth/`.

7. Runs evaluation on the algorithm outputs.

The final metrics are written to:

```text
pipeline_work/evaluation_output/metrics.json
```

To use a different raw data location:

```bash
RAW_DATA_DIR=/path/to/toothfairy4-cbct---raw ./run_full_pipeline.sh
```
