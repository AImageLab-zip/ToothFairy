FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS bite2text_ptv3_amd64

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    BITE2TEXT_POINTCEPT_ROOT=/opt/app \
    BITE2TEXT_CONFIG=/opt/ml/model/config.py \
    BITE2TEXT_CHECKPOINT=/opt/ml/model/model_best.pth \
    BITE2TEXT_HEAD_VOCABS=/opt/ml/model/head_vocabs.json \
    BITE2TEXT_NORMALIZER_ROOT=/opt/ml/ios-normalizer \
    BITE2TEXT_NORMALIZER_CHECKPOINT=/opt/ml/model/ios_normalizer_best.pt \
    BITE2TEXT_AMP=0

LABEL gc_upload_version="2026-08-10-bite2text-ptv3-debug-v3-content-aware-mesh"

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && python -m pip install --no-cache-dir --disable-pip-version-check \
        addict==2.4.0 \
        fpsample==1.0.2 \
        numpy==1.26.4 \
        scipy==1.15.3 \
        spconv-cu126==2.3.8 \
        timm==1.0.27 \
        trimesh==4.12.2 \
        yapf==0.43.0 \
    && apt-get purge -y --auto-remove g++ \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r user \
    && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY pointcept /opt/app/pointcept
COPY scannormalizer /opt/ml/ios-normalizer/src/scannormalizer
COPY overrides/models_init.py /opt/app/pointcept/models/__init__.py
COPY overrides/datasets_init.py /opt/app/pointcept/datasets/__init__.py
COPY overrides/point_transformer_v3_init.py /opt/app/pointcept/models/point_transformer_v3/__init__.py
COPY overrides/point_prompt_training_init.py /opt/app/pointcept/models/point_prompt_training/__init__.py
COPY overrides/segment.py /opt/app/pointcept/models/utils/segment.py

# torch-scatter does not publish a torch-2.9 wheel. These exact segment CSR
# reductions are native in modern PyTorch and avoid compiling CUDA extensions.
RUN sed -i \
        's/^import torch_scatter$/from pointcept.models.utils.segment import segment_csr/' \
        /opt/app/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py \
    && sed -i \
        's/torch_scatter\.segment_csr/segment_csr/g' \
        /opt/app/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py \
    && sed -i \
        's/^from torch_scatter import segment_csr$/from pointcept.models.utils.segment import segment_csr/' \
        /opt/app/pointcept/models/multi_task_classifier/multi_task_classifier_v1m1_base.py \
    && sed -i '/from pointcept\.engines\.hooks import HookBase/d' /opt/app/pointcept/models/modules.py \
    && sed -i 's/class PointModel(PointModule, HookBase):/class PointModel(PointModule):/' /opt/app/pointcept/models/modules.py

COPY inference.py normalize_pair.py prepare_ptv3_dataset.py report_renderer.py /opt/app/

RUN chown -R user:user /opt/app /opt/ml/ios-normalizer

USER user

ENTRYPOINT ["python", "inference.py"]
