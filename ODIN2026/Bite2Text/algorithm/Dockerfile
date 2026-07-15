FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS bite2text_algorithm_amd64

ENV PYTHONUNBUFFERED=1
LABEL gc_upload_version="2026-07-14-b2t-correct-task"

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
