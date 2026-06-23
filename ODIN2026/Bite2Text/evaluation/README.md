# Bite2Text Evaluation

Build:

```bash
docker build --platform=linux/amd64 -t bite2text-eval-test .
```

Expected prediction output:

```text
/input/<job-pk>/output/diagnostic-imaging-report.json
```

```json
{"report": "Generated English report text."}
```

Expected ground truth:

```text
/opt/ml/input/data/ground_truth/<patient-id>.txt
```

Create the Grand Challenge ground-truth tarball:

```bash
./create_ground_truth_tarball.sh
```

By default this reads `/mnt/c/Users/Kevin/Downloads/Bite2Text/<patient-id>/reports_intraoral-photo_en/*.txt`, keeps only the first report per patient ordered alphabetically, and packages a flat `evaluationgroundtruth.tar.gz` layout:

```text
./F5535.txt
./F5520.txt
./F5405.txt
```

These files are extracted directly under `/opt/ml/input/data/ground_truth/` at runtime.

Custom source/output:

```bash
./create_ground_truth_tarball.sh /path/to/evaluationgroundtruth /path/to/evaluationgroundtruth.tar.gz
```

Local test:

```bash
./do_test_run.sh
```

Local test with RadFact:

```bash
ENABLE_ONLINE_METRICS=1 OPENAI_API_KEY=... ./do_test_run.sh
```

Optional RadFact settings: `RADFACT_MODEL`, `RADFACT_TIMEOUT`, `RADFACT_MAX_RETRIES`.

Metrics are written to `test/output/metrics.json`.

When you run the full repository pipeline, the evaluator still writes its final output to `pipeline_work/evaluation_output/metrics.json` under the repository root. That folder is generated locally and should not be uploaded.
