# `code/pipeline/` — the inference path

One CBCT volume in, one radiology report out.

`infer.py` is the entry point and `scripts/run_infer.sh` submits it; the four
subdirectories below are the stages it drives, in order:

| | |
|---|---|
| `segmentation/` | the mask + facts handover — segmentation, fact extraction, and the mask-audit corrections all live here now (see [The facts audit](#the-facts-audit)) |
| `preprocess/` | volume + mask → rendered images → `qa_pairs.jsonl` |
| `vqa/` | the VLM calls — the only stage that needs a GPU |
| `postprocess/` | predictions → summaries → report — the rules it applies are [`configs/postprocess/`](../../configs/postprocess/), not constants in the code |

**On the 15-minute budget below.** This document was written for the ToothFairy4
Grand Challenge submission, which started a **fresh container for every case**
and gave it **15 minutes** for all of the above. That submission is finished and
nothing enforces the deadline now, but the schedule it forced — starting vLLM
first and rendering while it loads, polling `/health` rather than sleeping,
sizing concurrency off the KV cache the server reports — is kept, because every
one of those is worth having whether or not a clock is running. Read the
timings as the reason the pipeline is shaped this way, not as a live constraint.

## Who does what

| | owner | produces |
|---|---|---|
| **A. Segmentation** | *(segmentation)* | `mask.nii.gz` — per-tooth FDI labels + jaw structures |
| **B. Facts** | *(segmentation)* | `facts.json` — which FDI numbers are present/absent |
| **B′. Facts audit** | *(segmentation)* | `facts.json` already corrected against the mask |
| **C. Report** | *(VQA)* | rendered images → VLM reads → `report.txt` |

**vLLM belongs entirely to C.** B/B′ is the handover point: everything
downstream needs to know which teeth actually exist, and whether they were in
the volume at all.

**Our input is the CBCT volume, the mask, and *audited* facts.**
`extract_facts.py` (v4, in `segmentation/`) now applies the audit corrections
at extraction time by default, so facts produced there are already audited.
It is spelled out below anyway, because it is a requirement on what we are
handed, not an implementation detail: see [The facts audit](#the-facts-audit).

## The pipeline

```
 t=0 ┌─────────────────────────────────────────────────────────────┐
     │ 1a. START vLLM  (background, GPU)                           │
     │     loads model weights, compiles kernels, allocates KV      │
     └─────────────────────────────────────────────────────────────┘
     ┌─────────────────────────────────────────────────────────────┐
     │ 1b. segmentation  ->  mask.nii.gz + facts.json (CPU/GPU)    │
     │ 1b'. facts AUDITED against the mask (~2 s)                  │
     │ 1c. render images from volume + mask + AUDITED facts  (CPU) │
     │       panoramic | 3D views | sinus  (parallel)              │
     │       then 32 tooth close-ups (needs panoramic + 3D)        │
     └─────────────────────────────────────────────────────────────┘
                              │
         vLLM ready ──────────┤ 2. INFERENCE starts as soon as the
                              │    server is up AND some images exist.
                              │    It does NOT wait for all 32 crops.
                              ▼
                        3. post-process -> report.txt
```

**The point of 1a running first:** loading the model needs the GPU and no CPU;
segmentation and rendering need CPU and (mostly) no GPU. Run them at the same
time and one hides inside the other. Run them in sequence and you pay for both.

## The facts audit

**Status: done.** `code/pipeline/segmentation/extract_facts.py` (v4) now runs
both corrections below at fact-extraction time and emits `fov.maxilla` and
`bridge_arches` by default — extracted facts are audited facts.
`infer.py`'s own `facts:audit` phase (`audit_facts.py`, the pre-v4 standalone
audit) is off by default as of 2026-08-21 and kept only as a fallback for
facts that arrive unaudited from somewhere else; pass `--audit-facts` to turn
it back on.

What does not change either way is the contract: **every generator downstream
assumes the facts it opens have already been audited against the mask.** The
rest of this section is the specification of what that means — read it as a
requirement on the input, whoever runs it.

### Why it exists

`facts.json` is extracted from the radiology **report**; `mask.nii.gz` is the
**acquisition**. They disagree in one systematic, damaging way: when the
maxilla was never in the scan volume, the report still enumerates the upper
teeth — usually as *absent*, because they are absent from the picture the
radiologist had — and the extraction records that verbatim. Nothing downstream
can tell that apart from a genuinely edentulous maxilla.

S0002 is the worked example. Its mask holds 252 mm³ of maxilla — 0.5 % of the
segmentation, nothing at all in the render — and no maxillary tooth label
whatsoever, while its facts said `fov.maxilla: "partial"` and listed 11–28 as
absent. That produced the caption *"No maxillary tooth is present — the maxilla
is fully edentulous."* Every clause about the maxilla is false, and the last is
a clinical claim about anatomy nobody imaged.

### What it corrects

Both rules measure the mask with `create_3d_renders.mask_label_volumes` — the
same function the renderer captions from, so the audit and the caption can
never disagree. **Bone and teeth are judged separately**, because the mask
routinely has one without the other (P045 has zero maxillary bone and all
sixteen upper tooth labels, and its report says exactly that).

| | condition | effect |
|---|---|---|
| **A** | maxillary bone < `MIN_ARCH_BONE_MM3` | `fov.maxilla := "excluded"` — renders as *"maxilla not included"*, and is what schema v6.6's `maxilla_included` answers from |
| **B** | bone below threshold **and** no maxillary tooth label either | every maxillary FDI (11–28, plus 51–65 primary codes) removed from `teeth_present` / `teeth_absent` / `crowns` / `implants` / `ian_close_teeth` |

It also writes `bridge_arches`, which upstream has no key for at all.

### Why it cannot be skipped

Neither correction is something upstream could have made — both are statements
about the **acquisition**, and only the mask can make them:

- `fov.maxilla := "excluded"` was **never** set by `extract_facts` v3; v4 sets
  it by default. Absent-teeth reporting gates on exactly that field, and
  **12 of the 40 validate cases change** because of it.
- the fixed-bridge rule reads `bridge_arches`, a key v3 never emitted; v4
  emits it by default.

### How it is wired today, and what survives the move

- **It runs before anything reads the facts.** Every generator reads that file —
  it decides which teeth get outlined and what the captions say — so auditing
  after a render would mean rendering twice. Upstream this is automatic; here it
  is why `facts:audit` is the first phase.
- **It is free.** ~2 s of CPU, entirely inside the ~200 s model-load shadow —
  which is also why moving it upstream costs this pipeline nothing.
- **It works on a copy.** `audit_facts.py` rewrites in place, so the runner
  copies first and never mutates the input it was handed. Keep that rule after
  the merge: `/input` is read-only on Grand Challenge regardless.
- **Failure is not fatal.** A failed audit logs `[WARN]` and the run continues
  on unaudited facts — they still describe the case, just less precisely. Once
  the audit is upstream, that fallback disappears with it: unaudited facts in,
  wrong captions out, silently.
- **Its stdout is printed, not captured.** This is the one stage that rewrites
  an input the rest of the run treats as given, and silence is exactly how
  *"the FOV correction never fired"* stays invisible for a whole run. Whoever
  owns it next should keep saying what it changed, for the same reason.

## Timings we have measured

On an A100. An A10G is ~2.5× slower on memory bandwidth, so treat these as a
floor, not a promise.

| step | time | notes |
|---|---|---|
| vLLM startup on a 24 GiB card (A30) | **159 s** | measured, cold cache: engine init 116 s of it |
| vLLM startup on A100, warm compile cache | ~28 s | the same work with the kernels already compiled |
| render all images | 71 s | panoramic 27 s ∥ 3D 52 s ∥ sinus 9 s, then crops 18 s |
| inference, 38 calls | 96 s | batched; concurrency ~10, auto-sized from the KV pool |
| post-process + report | < 5 s | pure string templates, no model |
| segmentation | 134 s | measured in the real container (103 s of it GPU) |
| fact extraction | 30 s | CPU |

### ⚠ The single biggest risk: the torch.compile cache

vLLM compiles CUDA kernels on first start and caches them. Cached, startup is
~28 s. Uncached, it is ~200 s — **every case**, because the container is fresh
every time. On a 900-second budget that is 20% of everything, spent doing the
same work over and over.

**Fix: run the model once while building the image, and ship the resulting cache
inside it.** Set `VLLM_CACHE_ROOT` to a path inside the image, start the server
once during the build, let it finish loading, stop it. The compiled kernels are
then baked in.

Grand Challenge also forbids writing to `/tmp` at runtime, so the cache must not
default there.

## Two things measured on 2026-08-13 that change the ordering

### 1. vLLM should start when segmentation releases the GPU, not at the end

In the production log the order was: segment (134 s) → facts (30 s) → render
(78 s) → **then** start vLLM, 248 s in. Startup is the longest single phase, and
it spent that whole time not running.

It cannot start at t=0 either: segmentation's forward pass is 103 s **on the
GPU**, and loading an 11 GiB model beside it on a 24 GiB card risks running out
of memory. But facts and rendering are pure CPU — 108 s of them:

```
segment 134 s (GPU) ──┬── START vLLM here
                      ├── facts 30 s ─ render 78 s   (CPU, 108 s)
                      └── vLLM loads underneath      ~159 s
```

Measured on the A30, one case end to end, every call succeeding:

```
vLLM startup   190 s  ####################       (cold compile cache)
images          71 s  #######                     entirely hidden inside it
inference       96 s                    ##########
post + report  0.1 s
TOTAL          287 s  = 32% of the 900 s budget
```

Image generation costs **nothing** on the critical path — it finishes 119 s
before the server is ready. With the compile cache baked in, startup drops by
roughly 130 s and the total lands near 160 s (18%).

### 2. Suspect the offline environment before suspecting compute

`--enforce-eager` already skips CUDA-graph capture, so a >10 minute startup is
unlikely to be compilation. With no network, an HF-hub metadata lookup or a
vLLM usage-stats POST does not fail fast — it **blocks until timeout**. Set:

```
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  VLLM_NO_USAGE_STATS=1  DO_NOT_TRACK=1
```

### ⚠ Do not shorten `--max-model-len`

32768 is correct and must stay. The context has to hold the prompt **and** the
reply, and the whole-jaw calls ask for 8192 output tokens on top of a ~12.5 k
character prompt. At `--max-model-len 8192` every one of those calls fails with
a 400 — but the per-tooth calls still succeed, so **the run exits 0 and writes a
report**, just with the whole-jaw findings silently missing. We hit exactly this
in testing; it does not look like a failure.

## Container requirements

Base must have **CUDA 12.8** — the platform runs driver 570, and CUDA 13 needs
driver 580+. A cu130 image will not start at all.

| | version | why |
|---|---|---|
| NVIDIA driver (theirs) | 570 | fixed by Grand Challenge |
| CUDA | **12.8** | anything 12.x works on driver 570; 13.x does not |
| vLLM | **0.19.0** | newest release still on CUDA 12.8 *and* supporting Qwen3.5 |
| torch | 2.10.0+cu128 | pulled in by vLLM 0.19.0 |

### Two environments in one image

Segmentation and VQA both want torch, and probably not the same version. To
avoid a conflict that only shows up at runtime, each half gets its own venv:

```dockerfile
# segmentation
RUN python3 -m venv /opt/seg   && /opt/seg/bin/pip   install -r seg_requirements.txt
# VQA + vLLM
RUN python3 -m venv /opt/vqa   && /opt/vqa/bin/pip   install -r vqa_requirements.txt
```

Each step is then run with its own interpreter (`/opt/seg/bin/python …`,
`/opt/vqa/bin/python …`). They never import each other — the handover is
`mask.nii.gz` and `facts.json` on disk — so they do not need to share anything.

Cost: two copies of torch, roughly +5 GB of image. Worth it against a version
conflict discovered on the leaderboard.

## Hardware budget

| | limit |
|---|---|
| GPU | A10G, 24 GiB VRAM (or T4 16 GiB — see below) |
| RAM | 32 GiB |
| time | **15 min per case** |
| network | **none** — everything baked into the image |
| user | non-root; `/input` read-only; `/tmp` must be empty |

Model footprint: `Qwen3.5-9B-AWQ` is **11.2 GiB** of weights, leaving ~12 GiB of
KV cache on a 24 GiB card. It will **not** work on a T4: the 4-bit kernels
(Marlin) need compute capability 8.0+, and a T4 is 7.5.

## Interface between the halves

What this side is handed — the CBCT volume, the mask, and **audited** facts:

```
/input/                       given by Grand Challenge
  <case>.nii.gz               the CBCT volume

  A/B/B' produce:
    mask.nii.gz               FDI-labelled segmentation
    facts.json                already audited against the mask:
                              {"structured": {"teeth_present": [11,12,...],
                                              "teeth_absent":  [16,25,...],
                                              "bridge_arches": [...]},
                               "fov": {"maxilla": "excluded" | "partial" | ...}}
                              - maxillary FDIs dropped, when neither bone nor
                                teeth are in the volume

  C produces:
/output/
  report.txt                  the radiology report
```

`facts.json` needs `teeth_present` above all: the renderer draws and labels only
those teeth, so a segmentation false positive never becomes a tooth in the
report. Everything else in the file is optional in the sense that a generator
will not crash without it — but two fields the generators genuinely depend on
are `fov.maxilla: "excluded"` and `bridge_arches`. `extract_facts.py` v4 emits
both by default; facts extracted with `--no-audit`, or handed in from
elsewhere unaudited, still need the audit run somewhere before use — in
`infer.py` if nowhere else. Nothing about that makes it optional.

## Open questions

1. **How long does segmentation take, and does it use the GPU?** If it does, it
   competes with vLLM's load instead of hiding under it, and the two can no
   longer start together.
2. ~~**Is `facts.json` derived from the mask?**~~ **Partly answered.** The mask
   and the facts both come from the segmentation component — the pools
   `dataset/source/{predictions,facts_all_622_cases_new}` — so both exist on
   Grand Challenge and passing both is the real input shape, not a research
   convenience. What is *not* answered is how tightly the facts track the mask,
   and the audit exists because the answer is "not tightly enough": the two
   corrections it makes are ones only the mask can support, and one of them
   changes 12 of 40 validate cases. Treat the facts as a claim to be checked,
   not as a derived view of the segmentation — and, once the merge lands, as a
   claim the segmentation side checks before handing it over.
3. **Peak RAM of segmentation + rendering together.** The cap is 32 GiB for the
   whole container, and both halves load the volume.
