FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output
# RUN mkdir -p /output/images/inferior-alveolar-canal

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
# RUN python -m piptools sync requirements.txt
RUN python -m pip install --user -r requirements.txt


COPY --chown=user:user checkpoints /opt/app/checkpoints
COPY --chown=user:user src /opt/app/src
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
