# Pinned to YOUR training stack: torch 2.5.1 + cu124, cp311.
# -devel (not -runtime) because it ships nvcc, needed to compile mamba-ssm.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS toothfairy4_algorithm_amd64

ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
# Her VLM must not phone home at inference (network is off at run time).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# No network at run time: these two stop vLLM from blocking on
# telemetry POSTs that never fail fast.
ENV VLLM_NO_USAGE_STATS=1
ENV DO_NOT_TRACK=1
# Skips vLLM's cudagraph memory estimation, which otherwise triggers a full
# profiling/warmup pass. Measured on an A100: profiling 114.5s -> 0.58s and
# engine init 177s -> 15s, with the KV pool unchanged (475,728 tokens).
ENV VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# nnU-Net prints warnings if these are unset; point them at throwaway paths.
ENV nnUNet_raw=/tmp/nnraw
ENV nnUNet_preprocessed=/tmp/nnprep
ENV nnUNet_results=/tmp/nnres
# Cap parallel compile jobs so the mamba build doesn't OOM the builder.
ENV MAX_JOBS=4
# Fragmentation guard for the segmentation stage.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app
ENV PATH=/home/user/.local/bin:${PATH}

# -- Mamba CUDA kernels: compiled against the image's torch 2.5.1 ----------
# Build time only (network on now). --no-build-isolation so they see the
# already-installed torch. causal-conv1d first (mamba uses it during build).
# Exact versions match your umamba2 env.
# Prebuilt mamba wheels are vendored: GitHub release downloads are blocked
# from inside the build, and compiling from source needs nvcc plus ~20 minutes.
# Fetch them once on the host (see wheels/README) and install offline here.
COPY --chown=user:user wheels/ /opt/app/wheels/
# --no-deps keeps pip from resolving torch (already in the base image, and a
# different build would break the compiled kernels), so the few pure-Python
# deps the wheels need are installed explicitly.
RUN python -m pip install --user --no-cache-dir \
        einops ninja huggingface_hub packaging \
 && python -m pip install --user --no-cache-dir --no-deps /opt/app/wheels/*.whl

# -- Rest of the Python deps -----------------------------------------------
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --requirement /opt/app/requirements.txt \
 && python -c "import causal_conv1d, mamba_ssm, transformers; \
    print('mamba kernels OK |', transformers.__version__)"

# -- Your U-Mamba2 / nnU-Netv2 fork ---------------------------------------
COPY --chown=user:user umamba2/ /opt/app/umamba2/
RUN python -m pip install --user --no-cache-dir /opt/app/umamba2

# -- VLM report arm: SEPARATE venv (vllm 0.19.0 -> torch 2.10.0+cu128) -----
# ABI-incompatible with the torch 2.5.1+cu124 the mamba kernels were compiled
# against, so it gets its own interpreter. report_side.py launches every VLM
# script (and the vLLM server) with /opt/vlmenv/bin/python; the two never share
# a process.
#
# vllm 0.19.0 specifically: it ships cu128 wheels (torch 2.10.0+cu128) and its
# compiled extension links against CUDA 12.x. vllm 0.22.0 is a CUDA 13 build
# (`ImportError: libcudart.so.13`) and cannot run on Grand Challenge, which uses
# NVIDIA driver 570 / CUDA 12.8.
USER root
RUN python -m venv /opt/vlmenv \
 && /opt/vlmenv/bin/pip install --no-cache-dir --upgrade pip \
 && chown -R user:user /opt/vlmenv
USER user

COPY --chown=user:user requirements-vlm.txt /opt/app/
RUN /opt/vlmenv/bin/pip install --no-cache-dir -r /opt/app/requirements-vlm.txt \
 && /opt/vlmenv/bin/python -c "import torch, vllm; \
    assert torch.version.cuda.startswith('12'), \
    f'need a CUDA 12.x torch, got {torch.__version__} ({torch.version.cuda}) \
- GC driver 570 cannot run cu13x'; \
    print('VLM env OK:', torch.__version__, torch.version.cuda, vllm.__version__)"

# -- VLM report arm: her CBCT_VQA code + schema ----------------------------
# The scripts cross-import (create_tooth_detail imports from create_panoramic,
# run_vqa_inference imports postprocess_pred), so copy the whole tree.
COPY --chown=user:user vqa/code/   /opt/app/vqa/code/
COPY --chown=user:user vqa/schema/ /opt/app/vqa/schema/

# ---- VLM runtime settings: A10G (24 GiB) + fp16 weights ------------------
# Placed LOW in the file on purpose: these change often, and every layer below
# an edited line is rebuilt. Down here, tweaking a value costs seconds instead
# of recompiling the mamba kernels.
#
# fp16 Qwen3.5-9B loads ~17.7 GiB. On a 24 GiB A10G that leaves very little for
# the KV cache, hence util 0.92 and a short context (her prompts are one image
# plus a short question, nowhere near 8k tokens).
# AWQ weights (11.2 GiB) rather than fp16 (17.66 GiB): on a 24 GiB card that
# is the difference between ~10 GiB of KV cache and ~4 GiB, and the KV pool is
# what sizes inference concurrency. Marlin kernels need compute capability
# 8.0+, so this is A10G-only -- a T4 (7.5) cannot run it.
ENV VLM_ENABLED=1
# Her measured config does NOT use --enforce-eager: the compiled kernels are
# what Step 3 bakes into the image, and eager mode would skip the compilation
# there is a cache for. Cold that costs ~190s; with the baked cache, ~28s.
ENV VLM_ENFORCE_EAGER=0
ENV VLM_MAX_MODEL_LEN=32768
ENV VLM_GPU_UTIL=0.90
ENV VLM_DTYPE=float16
ENV VLM_CPU_OFFLOAD_GB=0
ENV VLM_PREFIX_CACHING=1
ENV VLM_STARTUP_TIMEOUT_S=600
ENV REPORT_FALLBACK_ON_ERROR=1

# -- vLLM torch.compile cache ---------------------------------------------
# Generated once on an sm80 card with the same model/flags. A miss is harmless:
# vLLM just recompiles and writes fresh artefacts, which is why the directory
# must be writable by the runtime user rather than root-owned.
# All three caches, at the exact paths they were generated under: the vLLM
# graph cache alone is not enough, because Inductor and Triton write their
# compiled kernels to /tmp/torchinductor_* and ~/.triton by default. Grand
# Challenge empties /tmp and gives a fresh HOME, so those were always cold and
# startup stayed ~250s. With all three: init engine 21s.
COPY --chown=user:user vllm_cache/     /opt/vllm_cache/
COPY --chown=user:user inductor_cache/ /opt/inductor_cache/
COPY --chown=user:user triton_cache/   /opt/triton_cache/
ENV VLLM_CACHE_ROOT=/opt/vllm_cache
ENV TORCHINDUCTOR_CACHE_DIR=/opt/inductor_cache
ENV TRITON_CACHE_DIR=/opt/triton_cache

# -- Pipeline code (your half + her half) ----------------------------------
COPY --chown=user:user config.py orientation.py segmentation.py facts.py \
     report_side.py inference.py /opt/app/
COPY --chown=user:user extract_facts.py /opt/app/
# The repo's challenge predictor is an in-repo FILE, not part of the installed
# package: copy it from ~/U-Mamba2/documentation/competitions/Toothfairy3/
COPY --chown=user:user task1_inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]