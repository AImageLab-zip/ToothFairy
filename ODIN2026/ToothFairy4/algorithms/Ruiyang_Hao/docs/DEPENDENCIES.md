# Dependencies

The repository owns the challenge inference orchestration, Stage-2 evidence
models, inference post-processing, and report realization code.

Stage 1 uses U-Mamba2 at commit
`2046d29785087b656ca69fa02dd40e43e69cfb42`. It is fetched into
`third_party/U-Mamba2` by `scripts/fetch_umamba2.sh` and is not vendored here.
U-Mamba2 remains subject to its own license.

Container runtime versions are pinned in `docker/requirements-runtime.txt`.
The Mamba CUDA extensions are platform-specific and are supplied as compatible
wheels when building the container.
