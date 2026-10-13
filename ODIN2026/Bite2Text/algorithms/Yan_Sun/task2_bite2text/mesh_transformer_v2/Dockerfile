FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /opt/mesh_transformer_v2
COPY common.py model.py train.py evaluate.py ./

ENTRYPOINT ["python"]
