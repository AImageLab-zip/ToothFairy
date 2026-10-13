FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    BITE2TEXT_WEIGHTS=/opt/ml/model

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# torch and torchvision come with the base image
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user bite2text /opt/app/bite2text
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
