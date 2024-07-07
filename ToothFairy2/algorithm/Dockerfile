FROM pytorch/pytorch

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/
# RUN python -m piptools sync requirements.txt
RUN python -m pip install --user -r requirements.txt


COPY --chown=user:user process.py /opt/app/
# COPY --chown=user:user PosPadUNet3D.py /opt/app/
# COPY --chown=user:user checkpoints.pth /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
