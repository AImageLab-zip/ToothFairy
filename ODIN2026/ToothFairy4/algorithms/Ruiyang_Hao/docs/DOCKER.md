# Docker

The Docker source contains no model weights. Prepare the external dependency
and compatible Mamba wheels, then build:

```bash
bash scripts/fetch_umamba2.sh
mkdir -p docker/wheels
# Place compatible causal-conv1d and mamba-ssm wheels in docker/wheels.
bash docker/build_image.sh evident-cbct:latest
```

Local inference mounts models read-only:

```bash
EVIDENT_MODELS=/path/to/models bash docker/run_container.sh INPUT OUTPUT
```

The challenge adapter reads one volume from `/input` and writes
`/output/diagnostic-imaging-report.json`. `docker/test_submission.sh`
enforces a 900-second timeout, 32 GiB DRAM, eight CPUs, no network, and validates
the output JSON. Hardware-specific runtime must still be checked on the target
platform.

