# Legacy synthetic demonstration

The original `src/bitemark` package, root `Dockerfile`, `configs/public_placeholder.yaml`
and their demonstration tests are retained for compatibility. They were published
under Apache-2.0 and use synthetic or placeholder logic. They do not reproduce the
final submission. The `bitemark` console command still refers to that demo.

Use `submission/Dockerfile`, `submission/inference.py`, `final2/` and the public
final2 model release for actual inference. The prior private-request-only weight
policy has been superseded. Real calibrated values are in `final2/pipeline.py`
and `final2/model/u/`; never substitute the legacy YAML for them.
