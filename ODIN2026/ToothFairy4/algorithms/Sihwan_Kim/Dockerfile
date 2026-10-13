FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir packaging ninja \
    && python -m pip install --no-cache-dir --no-build-isolation \
       causal-conv1d==1.5.2 mamba-ssm==2.2.5
RUN git init /opt/backbone \
    && git -C /opt/backbone remote add origin https://github.com/zhiqin1998/U-Mamba2.git \
    && git -C /opt/backbone fetch --depth 1 origin 2046d29785087b656ca69fa02dd40e43e69cfb42 \
    && git -C /opt/backbone checkout FETCH_HEAD
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps -e /opt/backbone
COPY src/repgen/compat/checkpoint_loader.py /opt/backbone/nnunetv2/training/nnUNetTrainer/variants/competitions/repgen_inference.py
RUN rm -rf /opt/backbone/.git

FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/backbone /opt/backbone
RUN groupadd --system user && useradd --system --gid user --create-home user

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/home XDG_CACHE_HOME=/tmp/cache TORCH_HOME=/tmp/torch \
    TRITON_CACHE_DIR=/tmp/triton NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nnUNet_compile=false \
    nnUNet_raw=/tmp/nnunet_raw nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/opt/ml/model/nnUNet_results PYTHONPATH=/opt/app/src:/opt/backbone

WORKDIR /opt/app
COPY --chown=user:user src /opt/app/src
COPY --chown=user:user configs /opt/app/configs
COPY --chown=user:user inference.py /opt/app/inference.py
USER user
ENTRYPOINT ["python", "/opt/app/inference.py"]
