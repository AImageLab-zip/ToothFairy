FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /opt/baseline
COPY prepare_manifest.py train.py evaluate.py evaluate_ensemble.py probe_geometry.py inference.py export_inference_checkpoints.py ./

ENTRYPOINT ["python"]
