# cu126 is what SUBMITS: Grand Challenge runs T4 (sm_75) and A10G (sm_86), both
# supported. Do not change the default.
#
# For LOCAL GPU testing on Blackwell (RTX 5090, sm_120) cu126 has no compatible
# kernels, so build with the cu128 base instead -- same torch version, so the
# only difference is the CUDA runtime:
#   ./do_build.sh --local
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
FROM --platform=linux/amd64 ${BASE_IMAGE} AS toothfairy4_algorithm_amd64

ENV PYTHONUNBUFFERED=1
# The caching allocator fragments badly on a 14.7 GB T4: cuDNN asks for a ~7.7 GB
# conv workspace, the allocator caches that block, and the next request cannot
# find a contiguous one -- "7.54 GiB is reserved by PyTorch but unallocated".
# Expandable segments let the allocator grow a segment instead of needing one
# contiguous block, which is exactly what PyTorch's own OOM message recommends.
# PyTorch 2.9 renamed this to PYTORCH_ALLOC_CONF and warns on the old name, but
# older versions only read the old one. Both are set so the setting survives
# either way -- it is load-bearing, not cosmetic.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user vendor /opt/app/vendor
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
