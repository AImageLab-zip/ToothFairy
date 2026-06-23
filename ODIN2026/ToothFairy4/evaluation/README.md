# ToothFairy4 Evaluation

Build:

```bash
docker build --platform=linux/amd64 -t toothfairy4-eval .
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

By default this reads `/home/llumetti/Downloads/toothfairy4-cbct---raw/<patient-id>/reports_en/*.txt`, keeps only the first report per patient ordered alphabetically, and packages a flat `evaluationgroundtruth.tar.gz` layout:

```text
./A001.txt
./A002.txt
./F001.txt
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
