FROM --platform=linux/amd64 docker.io/library/python:3.11-slim AS toothfairy4-evaluation

ENV PYTHONUNBUFFERED=1
ENV ENABLE_ONLINE_METRICS=0
ENV RUNNING_ON_GRAND_CHALLENGE=1
ENV PATH=/home/user/.local/bin:${PATH}

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

RUN python - <<'PY'
import evaluate

evaluate.load("bleu").compute(
    predictions=["a b c d"],
    references=[["a b c d"]],
    max_order=4,
)
evaluate.load("meteor").compute(
    predictions=["a b c d"],
    references=["a b c d"],
)
PY

ENV HF_HUB_OFFLINE=1

COPY --chown=user:user evaluate.py /opt/app/evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
