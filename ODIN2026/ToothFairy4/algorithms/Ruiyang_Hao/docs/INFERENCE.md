# Inference

The public entry point accepts one `.nii`/`.nii.gz` CBCT volume or a directory
searched recursively for a single volume per case:

```bash
python -m evigre.inference INPUT OUTPUT --config local_inference.json
```

For a single case, `OUTPUT` contains the final report, the deterministic template,
structured evidence before/after the fixed inference filters, filter/projection
audits, and a pipeline manifest/log.

The runtime sequence is:

1. Stage-1 anatomy segmentation using the pinned U-Mamba2 backend.
2. Inference-only construction of global, tooth, and relation inputs.
3. Three released Stage-2 branch checkpoints.
4. Frozen challenge post-processing and dental consistency projection.
5. Deterministic report rendering.
6. Optional image-blind Qwen LoRA rewrite.

Use `--preflight-only` to validate CUDA, paths, input geometry, thresholds, and
rewriter artifacts without running model inference. Use `--skip-rewrite` to emit
the deterministic template as the final report.
