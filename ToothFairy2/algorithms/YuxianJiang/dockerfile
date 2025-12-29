# FROM pytorch/pytorch
# FROM --platform=linux/amd64 pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /workspace /input /output /workspace/networks /workspace/weight\
    && chown user:user /workspace /input /output /workspace/networks /workspace/weight
# RUN mkdir -p /output/images/oral-pharyngeal-segmentation

USER user
WORKDIR /workspace

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

#COPY --chown=user:user ./   /workspace

# RUN pip install pip -U
RUN pip install pip -U
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ --user

COPY --chown=user:user requirements.txt /workspace
RUN python -m pip install --user -r requirements.txt


COPY --chown=user:user test_3D.py /workspace
#COPY --chown=user:user predict.sh /workspace
#COPY --chown=user:user name_config.py /workspace
COPY --chown=user:user test_3D_util_mirror.py /workspace
#COPY --chown=user:user spacing_config.py /workspace
COPY --chown=user:user /weight /workspace/weight
COPY --chown=user:user /networks /workspace/networks

#CMD ["/workspace/predict.sh"]
#ENTRYPOINT ["sh"]

ENTRYPOINT [ "python", "test_3D.py" ]
# ENTRYPOINT [ "python", "-m", "process" ]
