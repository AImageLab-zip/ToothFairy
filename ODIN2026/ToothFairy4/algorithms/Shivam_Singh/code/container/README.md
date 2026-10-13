# Container — the Grand Challenge submission path

This directory is the thing that joins the two halves. Stage 1 (segmentation,
`../pipeline/segmentation/`) turns a CBCT volume into an FDI-labelled mask and
a facts file; stage 2 (`../pipeline/`, the VQA pipeline) turns those into
radiology prose. Grand Challenge runs one case per container, so both stages
have to live in one image and finish inside a per-case time limit.

```
/input/images/cbct/<case>.mha
        |
        |  inference.py
        v
  segmentation.py  -> mask.nii.gz          (main env,  torch 2.5.1+cu124)
  facts.py         -> facts.json
        |
        |  report_side.py
        v
  competition_runner.py                    (/opt/vlmenv, torch 2.10.0+cu128)
        |
        v
/output/diagnostic-imaging-report.json
```

## Build

```bash
git clone https://github.com/zhiqin1998/U-Mamba2 umamba2   # CC BY-NC 4.0, not vendored
./do_build.sh
./do_save.sh            # image tarball + model.tar.gz
```

`model.tar.gz` must contain two directories at its root:

```
nnunet_seg/   dataset.json, plans.json, fold_all/checkpoint_final.pth
vlm/          the AWQ-quantised VLM
```

---

## Four things that are not obvious

Each of these cost a submission to find. They are recorded here so they do not
have to be found again.

### 1. Orientation: the volumes are L-first, the model is not

All 622 ToothFairy4 v2 volumes are stored **L-first** (LAS 527, LPS 95). The
model was trained on R-first array content, so the left/right array axis is
inverted relative to the training convention. `segmentation.py` flips array
axis 2 before inference and flips the mask back afterwards.

Verified against single-implant ground truth in the reference reports:

| case | stored | report says | flip axis 2 | no flip |
|---|---|---|---|---|
| F030 | LAS | implant 25 | **25** | 15 |
| P061 | LAS | implant 36 | **36** | 46 |
| A046 | LPS | implant 35 | **35** | 45 |

Two traps here. Header-based reorientation does **not** fix it — `DICOMOrient`
and identity-matrix checks do not touch the left/right axis, so a pipeline that
"handles orientation" can still be mirrored throughout. And the failure is
silent: a mirrored jaw is still an anatomically plausible jaw, so segmentation
metrics look fine while every tooth number is wrong.

### 2. Two Python environments, because the stacks cannot share a process

| env | torch | used by |
|---|---|---|
| main (`python`) | 2.5.1+cu124 | segmentation; the mamba CUDA kernels are compiled against this ABI |
| `/opt/vlmenv` | 2.10.0+cu128 | vLLM 0.19.0 and the whole VQA half |

`report_side.py` launches every VQA process with `/opt/vlmenv/bin/python`. The
two never share an interpreter.

**vLLM 0.19.0 specifically.** Its wheels are built against CUDA 12.x. vLLM
0.22.0 is a CUDA 13 build and fails at import with `libcudart.so.13: cannot
open shared object file` on Grand Challenge, which runs NVIDIA driver 570 /
CUDA 12.8. Pinning torch to a cu128 build does not help: vLLM's own compiled
extension is what needs the CUDA 13 runtime.

### 3. Three compile caches, not one

vLLM cold start is ~250 s, paid **per case** because every case is a fresh
container. Shipping a warm cache brings it to ~25 s — but only if all three
caches are shipped:

| cache | default location | why it must move |
|---|---|---|
| vLLM graph cache | `VLLM_CACHE_ROOT` | — |
| Inductor kernels | `/tmp/torchinductor_*` | Grand Challenge empties `/tmp` at container start |
| Triton kernels | `~/.triton` | fresh `$HOME` every run |

All three are redirected to `/opt/*_cache` and baked into the image. Shipping
only the vLLM cache buys nothing: it loads, then Inductor and Triton recompile
from scratch and startup stays ~250 s.

The cache is also keyed on the **model path** and the **Python version**, not
only the GPU architecture. Generate it at `/opt/ml/model/vlm` under Python
3.11, on a compute-capability 8.6 card (A10/A10G), or it will miss. It is
**not** keyed on the weights: the same cache serves any fine-tune of the same
architecture at the same serving flags.

A miss is harmless — vLLM recompiles — so a stale cache costs time, not
correctness.

### 4. Inference memory is bounded by patch size, not volume size

The sliding-window logits accumulator and the final resample both run on the
CPU (see the header of `task1_inference.py`). A 47-class full-resolution
accumulator is ~24 GiB and will not fit a 24 GiB card alongside the model. With
the accumulator in system RAM and the resample done per class channel with a
running argmax, peak GPU memory is one patch and peak RAM is ~2 GiB, whatever
the field of view.

---

## Runtime settings

Grand Challenge passes no environment variables, so everything is baked into
the Dockerfile. The ones that are not free to change:

| variable | value | why |
|---|---|---|
| `VLM_MAX_MODEL_LEN` | 32768 | whole-jaw calls request 8192 output tokens on a ~12.5k-character prompt. At 8192 they fail with a 400 while per-tooth calls still succeed — the run exits 0 and writes a report with the whole-jaw findings **silently missing** |
| `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK` | 1 | with no network a telemetry POST does not fail fast, it blocks until timeout |
| `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `/opt/*_cache` | see above |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | 1 | no network at inference |

## Measured per case (A10G, 24 GiB)

| stage | small case | largest case |
|---|---|---|
| segmentation | 80 s | 128 s |
| fact extraction | 12 s | 52 s |
| image generation | 47 s | 128 s |
| vLLM startup (cached, overlapped with image generation) | ~25 s | ~25 s |
| VQA inference | ~135 s | ~310 s |

The four image generators run concurrently and overlap vLLM startup, so the
stage costs roughly the slowest generator rather than their sum.

## Packaging

Save the **tag**, not the repository name — `docker save <repo>` exports every
tag under it and Grand Challenge rejects a tarball containing more than one
image:

```bash
docker save toothfairy4-example-algorithm:latest | gzip -c > submission.tar.gz
gunzip -c submission.tar.gz | tar -xO index.json | python3 -m json.tool | grep -c '"digest"'   # want 1
```
