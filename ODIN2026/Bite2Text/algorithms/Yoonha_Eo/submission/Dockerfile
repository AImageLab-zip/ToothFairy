FROM --platform=linux/amd64 pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel AS extension_builder

ENV TORCH_CUDA_ARCH_LIST="7.5;8.6" MAX_JOBS=4
WORKDIR /build
RUN apt-get update && apt-get install --yes --no-install-recommends libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir ninja wheel
COPY pointops /build/pointops
COPY pointgroup_ops /build/pointgroup_ops
COPY teethland_pointops /build/teethland_pointops
RUN python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheels /build/pointops \
    && python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheels /build/pointgroup_ops \
    && cd /build/teethland_pointops \
    && TORCH_CUDA_ARCH_LIST="7.5;8.6" MAX_JOBS=4 python setup.py build_ext --build-lib /teethland-pointops


FROM --platform=linux/amd64 pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    CUDA_MODULE_LOADING=LAZY \
    TOKENIZERS_PARALLELISM=false
LABEL gc_upload_version="2026-08-18-bite2text-bitemark-final2"

USER root
RUN apt-get update && apt-get install --yes --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY requirements.txt /opt/app/
COPY --from=extension_builder /wheels /opt/wheels
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt \
    && python -m pip install --no-cache-dir --no-color --no-index \
        torch-scatter==2.1.2+pt27cu126 \
        torch-cluster==1.6.3+pt27cu126 \
        torch-sparse==0.6.18+pt27cu126 \
        --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html \
    && python -m pip install --no-cache-dir --no-color torch-geometric==2.8.0.post1 \
    && python -m pip install --no-cache-dir --no-color /opt/wheels/*.whl \
    && python -m pip check \
    && rm -rf /opt/wheels /root/.cache /opt/conda/pkgs/*

COPY --from=extension_builder /teethland-pointops /opt/teethland-pointops
COPY teethland-python /opt/teethland-python
RUN chmod 0755 /opt/teethland-python \
    && python -c "import pointgroup_ops, pointops, spconv.pytorch, torch, torch_cluster, torch_geometric, torch_scatter, torch_sparse, ultralytics; assert torch.__version__.startswith('2.7.0')" \
    && PYTHONPATH=/opt/teethland-pointops python -c "import torch, open3d, pointops, pymeshlab, pytorch_lightning; assert hasattr(pointops, 'ballQuery')"

RUN groupadd --gid 1000 algorithm && useradd --uid 1000 --gid algorithm --create-home algorithm \
    && mkdir -p /input /output /opt/ml/model \
    && chown -R algorithm:algorithm /input /output /opt/app
COPY --chown=algorithm:algorithm inference.py /opt/app/
USER algorithm
ENTRYPOINT ["python", "inference.py"]
