FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/evident/U-Mamba2:/opt/evident/EviDent-CBCT \
    EVIDENT_INFERENCE_CONFIG=/opt/evident/EviDent-CBCT/evigre/configs/inference/evident_cbct.container.json \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN test "${TARGETARCH:-amd64}" = "amd64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 libgomp1 python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY container/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        -r /tmp/requirements-runtime.txt

COPY wheels/ /tmp/wheels/
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        /tmp/wheels/causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl \
        /tmp/wheels/mamba_ssm-2.2.5-cp312-cp312-linux_x86_64.whl \
    && rm -rf /tmp/wheels /tmp/requirements-runtime.txt

COPY U-Mamba2/ /opt/evident/U-Mamba2/
COPY EviDent-CBCT/ /opt/evident/EviDent-CBCT/
COPY container/entrypoint.sh /usr/local/bin/evident-cbct-cli

RUN chmod +x /usr/local/bin/evident-cbct-cli \
    && python3 -c "import importlib.metadata as m; import cc3d, nibabel, nnunetv2, peft, torch, transformers; import evigre; assert m.version('mamba-ssm') == '2.2.5'" \
    && python3 -m evigre.inference.toothfairy4_submission --help >/dev/null

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL org.opencontainers.image.title="EviDent-CBCT" \
      org.opencontainers.image.version="public-v1"

WORKDIR /opt/evident/EviDent-CBCT
ENTRYPOINT ["python3", "-m", "evigre.inference.toothfairy4_submission"]

