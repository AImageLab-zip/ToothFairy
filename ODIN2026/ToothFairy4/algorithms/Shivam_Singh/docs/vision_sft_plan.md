# Vision-path LoRA SFT

What was done. **Why** it was done is the experiment log at the end — every
design choice here has an entry there, and the entry carries the measurement.

> **What is named here but not shipped.** This document is written against the
> research repo and cites it by path. The public release ships the training
> path that produced the checkpoint — `scripts/run_sft.sh` and the modules
> under `code/train/` — but not `docs/workflow.md` (the map of the research
> tree) or `code/train/visual_evidence/`, the teacher-evidence side arm the
> released checkpoint was **not** trained with. Those names are provenance for
> a decision, not instructions; every number below is stated here.

## 0. Dataset split

**Nothing in `dataset/validate` ever enters training.**

- **The training pool is drawn from `training` only.**
- **The held-out set for tuning is carved out of `training`**, not borrowed from
  validate — 24 whole cases, `outputs/training_results/sft_pool/heldout.txt`.
  Every decision that needs a number *during* the experiment reads that set.
  Validate is read once per arm, at the end.

## 2. Arms

What gets served, and what each one is:

| arm | checkpoint | trained on | Final, per case |
|---|---|---|---|
| **0a** | `Qwen3.5-9B-AWQ` | — untrained baseline | 0.3357 ± 0.1745 |
| **0b** | `Qwen3.5-9B` | — untrained bf16 control | not scored |
| **5** | `Qwen3.5-9B-AWQ-arm5` | tooth composite only | 0.3571 ± 0.1546 |
| **6-evidence** | `Qwen3.5-9B-AWQ-dental-cbct-sft` | + the nine global calls, teacher evidence on tooth rows only | **0.4094 ± 0.1632** |
| **6-uniform** | — | same rows, `visual_evidence` empty throughout | *training* |

Validate-40, local `qwen3-14b` judge, mean ± sd over the 40 cases. Arm 0b has no
official score — it was measured against 0a at the fact level only, mean F1
0.552 against 0.572.

Per-case sd is ~0.16, so a single arm's mean says little on its own. **Paired**
against the same cases is what decides:

| | Δ Final | 95% CI | better / worse |
|---|---|---|---|
| 5 − 0a | +0.0215 ± 0.1145 | −0.0140 to +0.0569 | 24 / 16 |
| 6e − 0a | **+0.0737** ± 0.1254 | **+0.0348 to +0.1125** | 27 / 11 |
| 6e − 5 | **+0.0522** ± 0.1260 | **+0.0132 to +0.0913** | 23 / 17 |

**Arm 5 alone is not distinguishable from the baseline** — its interval crosses
zero. Arm 6 is, and so is arm 6 over arm 5.

**6-uniform exists because 6-evidence was not one thing.** Its tooth rows carried
teacher prose and its arch rows carried `""`, so the two halves trained different
conditioning regimes; `visual_evidence` leads every fact block, so that reaches
every supervised decision. 6-evidence is superseded once 6-uniform is scored.

The *training*-side arm table — which tensors each one adapts — is
`code/train/lora_arms.py`. `assert_arm()` runs after `get_peft_model()` and
before the optimizer, and refuses to start unless the applied adapter matches the
declared parameter count.

## 3. Targets

### 3.0 What a row is

One row is one VLM call. Two kinds, both trained since arm 6.

**Tooth composite — 6,567 rows.** `create_tooth_detail.py` writes a 3×3 grid
(axial crown/mid/root, coronal post/mid/ant, sagittal R/mid/L) with the target
tooth outlined **red in every tile** and the mandibular canal filled translucent
yellow on lower teeth. 1343×1356 px → **1,764 vision tokens**, against 768 for
the entire panoramic. With the prompt text a call is **≈6,200 tokens**, uniform
across every sample. Five facts per call, six on 36–38/46–48 where
`mandible_canal` applies.

**Global — 4,234 rows** over nine call shapes: panoramic 2,228, 3D renders
1,746, sinus 260.

### 3.1 The label is the generated GT

`parse_reports_to_gt.py` turns the reference reports into `{case}_gt.json` —
**1,274 training, 93 validate** — in the same shape as `{case}_pred.json`:
`{case_id, global, teeth, _derivation}`, `teeth` carrying all 32 blocks in v7.1
vocabulary. How it is built and audited is `docs/workflow.md`, **THE GROUND
TRUTH**.

### 3.2 The loss mask comes from the GT's own nulls

Reports state findings by exception, and that cuts two ways.

**On a tooth that is there, silence is an answer and is supervised.** A present
tooth the report never calls carious is a tooth without caries. This is where the
~92%-negative prior comes from.

**On a tooth the report says is gone, the same silence says nothing.** Those
fields are `null`, and `structured_findings_evaluation.py` drops null-GT pairs,
so the loss mask and the metric read one definition. The rule is per *value*, not
per field: what the report **asserted** at an absent position survives, what only
**silence** produced becomes null. `bone_loss` is the carve-out — it comes from
an arch-level assertion about the dentition, not about the ridge.

Two accounting facts: **`unstated` never fires** — a position the report never
placed gets no tooth block, so the call is dropped as `no_gt_block` (7,122)
before any mask is consulted; and **`gated` fires only on arch facts** (3,090,
via `ARCH_GATES`), since `filling_quality` is already null wherever `with_endo`
is false.

### 3.3 `visual_evidence`

Empty and masked in all 47,763 fields, tooth and arch alike, so every supervised
decision is conditioned the same way and `--evidence-weight` has nothing left to
weight. The teacher-written evidence pass is off this path; its code, the
arithmetic behind the 0.04 weight, and what the perceivability screen kept are
all in `code/train/visual_evidence/README.md`.

### 3.4 Prompt parity

The training prompt is **token-identical** to what the server sends. The collator
renders through the pipeline's own `build_call_prompt()` and reproduces vLLM's
`string` content format via `sft_prompt.CONTENT_FORMAT`.
`check_prompt_parity.py` and the `STAGE=parity` gate diff `input_ids` against a
live `/tokenize`; re-run them after any change to the chat template, the server
flags, or `build_user_blocks`.

### 3.5 The pool, and the negative cap

**The pool is the whole training split** — 582 cases, `sft_pool/all_cases.txt`.
24 are held out into `heldout.txt`, stratified 6 per prefix to match validate's
10/10/10/10; the complement `train_minus_heldout.txt` (558) is what
`--case-list` takes. `select_sft_pool.py` writes all three.

The pool is **69% P** against validate's 25%, so **read the per-prefix breakdown
before believing any aggregate gain**.

**Quality filtering happens at the call, not the case.** `no_gt_block` per tooth,
`contradiction` and `arch-range` per FDI, `null`/`gated`/`refused`/the cap per
field. Nothing is dropped at case level: if it is hard to decide, mask it.

**Negatives are capped at 6:1 per field, by masking.** Every call stays, every
positive stays, only the loss changes. Seeded (`--seed 42`) so which negatives
are masked is identical across arms. Enums are left alone.

## 4. Training setup

| Component | Configuration |
|---|---|
| Runtime | Linux (SLURM); Python 3.12.13; PyTorch 2.11.0+cu128 (CUDA 12.8); transformers 5.9.0; peft 0.20.0; accelerate 1.13.0; datasets 5.0.1; Triton 3.6.0; flash-linear-attention 0.5.2 |
| Base model | Qwen3.5-9B, bf16, 9.41 B parameters. Trained on the bf16 checkpoint; the adapter is merged into `Qwen3.5-9B-AWQ` for serving |
| Adapter placement | LoRA on **214 modules**: vision blocks ×27 (`attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`), vision merger ×2, language full-attention layers ×8 (`self_attn.{q,k,v,o}_proj`), GatedDeltaNet layers ×24 (`linear_attn.{in_proj_qkv, in_proj_z, out_proj}`) |
| LoRA | rank 16; `lora_alpha` 32 (= 2r); dropout 0.05; `bias="none"`; **22,928,896 trainable parameters (0.24%)** |
| Optimiser | AdamW fused; β = (0.9, 0.999); ε = 1e-8; weight decay 0; gradient clipping 1.0 |
| Schedule | LR 1e-4, cosine, 3% warm-up; 2 epochs; per-device batch 1 × 16 gradient-accumulation steps (effective batch 16); **1,190 optimizer steps** |
| Precision | bfloat16; gradient checkpointing on both towers; `logits_to_keep` slices hidden states to the supervised span before `lm_head` |
| Sequence | `max_length` 8192; longer calls dropped and counted, never truncated |
| Loss | Token cross-entropy on the **values** of supervised fields only — not braces, field names, commas or `<\|im_end\|>`. Per-token weights: decision fields 1.0, `visual_evidence` 0 |
| Data | 582 cases → 10,801 rows; 24 cases held out → 558 cases / 10,207 rows; a further 30 cases / 687 rows held for eval loss → **9,520 training rows over 528 cases**. Negatives capped at 6:1 per field by masking, seed 42 |
| Hardware | 1 × NVIDIA A100 40 GB, 8 CPU, 64 GB RAM. ~12.5 h; peak 23.62 GiB |
| Merge | LoRA merged in fp32, cast to bf16, written into a copy of the AWQ checkpoint. The targeted tensors are bit-identical between the two checkpoints, so the merge is exact |
| Serving | vLLM 0.19.0, CUDA 12.8 container; `--max-model-len 32768`; `--limit-mm-per-prompt {"image": 3}`; `--gpu-memory-utilization 0.85`; prefix caching on |
| Inference | One request per call, JSON schema sent as OpenAI-standard `response_format`; `max_tokens` 8192; 3 parse retries. Sampling is the model's own `generation_config.json` — **temperature 1.0, top-p 0.8, top-k 20, presence penalty 1.5** — not pinned |

Prose form. The adapter is LoRA of rank 16 with a scaling factor of 32 and
dropout 0.05, applied to the vision tower, the vision–language merger, and the
language model's attention projections; the base weights are frozen throughout,
giving 22.9 M trainable parameters against a 9.41 B base. The per-device batch
size is 1 with 16 gradient-accumulation steps, an effective batch of 16. The
initial learning rate is 1 × 10⁻⁴ with cosine scheduling and a 3% warm-up ratio,
for 2 epochs, in bfloat16 with gradient checkpointing. Loss is token
cross-entropy restricted to the values of supervised fields. Training runs on one
NVIDIA A100 40 GB and takes about 12.5 hours for 1,190 optimizer steps. The
adapter is then merged into the AWQ checkpoint and served under vLLM 0.19.0.

**Decoding is stochastic and was never pinned.** `TEMPERATURE` defaults to
unset, so vLLM falls back to the checkpoint's `generation_config.json` and every
scored arm was sampled at temperature 1.0 / top-p 0.8 / top-k 20. Setting
`TEMPERATURE=0` makes a run replayable, but adopting it means re-scoring every
banked baseline at 0 as well.

## 5. Evaluation

Train on `training`, score on `validate` (40 cases, 93 `_gt.json`).

**Every reported number is `mean ± sd` over the cases**, computed from
`<run>/official_ranking/results.jsonl`, four decimal places, in six columns:
BLEU-4, METEOR, RadFact-P, RadFact-R, RadFact-F1, Final. Not
`summary.json -> aggregate`, which is a corpus-level statistic and not the mean
of the per-case values; the two differ visibly and must never share a table.

| | BLEU-4 | METEOR | RadFact-P | RadFact-R | RadFact-F1 | Final |
|---|---|---|---|---|---|---|
| only segmentation | 0.1435 ± 0.1739 | 0.3006 ± 0.1902 | 0.3443 ± 0.3113 | 0.3464 ± 0.2585 | 0.2907 ± 0.2333 | 0.2770 ± 0.1841 |
| qwen3.5-9B AWQ | 0.1506 ± 0.0982 | 0.3824 ± 0.1593 | 0.4318 ± 0.2904 | 0.4023 ± 0.2623 | 0.3530 ± 0.2168 | 0.3357 ± 0.1745 |
| + LoRA SFT arm 5 | 0.1542 ± 0.1033 | 0.3807 ± 0.1604 | 0.4834 ± 0.3040 | 0.4157 ± 0.2345 | 0.3795 ± 0.1918 | 0.3571 ± 0.1546 |
| + LoRA SFT arm 6 | 0.1553 ± 0.0990 | 0.3848 ± 0.1571 | 0.5319 ± 0.2736 | 0.4762 ± 0.2310 | 0.4442 ± 0.2004 | 0.4094 ± 0.1632 |

**Per-case sd is ~0.16 on Final, five times the gap between arms, so two means
are not a comparison.** Decide arm-vs-arm on the **paired** per-case delta and
its CI (§2). Every row must come from the same judge process on the same day —
the judge is non-deterministic at temperature=0.

Then, in order:

1. **`structured_findings_evaluation.py`**, per field and per call group, `PRED`
   beside `SUMMARY`. It explains *what* changed; it does not decide.
2. **The per-prefix breakdown**, because the pool is 69% P and validate is 25%.
3. **Arm 0**, claim counts per field, matched. Any precision gain arm 0
   reproduces is a threshold shift, not perception.
4. **Overlay-mention rate** — the fraction of `visual_evidence` strings citing
   the coloured outline. It is the check that caught the int4 failure.

**Do not read a per-field number without its positive count.** Inter-reader
percentages are coverage, not agreement.

## 6. Open

- **`NO_FACTS=1` — the overlay ablation, never run.** It changes the *pixels*,
  not just the captions, so re-rendering validate with it and re-scoring base +
  arm is the falsification test for "the arm learned to read the segmentation
  overlay rather than the anatomy". If the gain survives without the overlay it
  is anatomy; if it vanishes it is not. ~7 min of rendering plus one inference
  job. Note it cannot change what ships — the overlay is present at submission
  time too — so it is interpretation, not a gate.
- **Image staleness is not detectable by mtime.** Every generator fix invalidates
  earlier PNGs, and a checkout rewrites mtimes with no content change. The skip
  check should key on the generator's git commit plus the `schema.json` sha256
  written into the caption sidecar.
- **`--export=ALL,VAR=a,b,c` does not do what it looks like.** `--export` is
  itself comma-separated, so Slurm reads `VAR=a` and drops the rest. It cost the
  few-shot probe a silent one-case run. Pass case lists via a file and **print
  the count** at job start.

## Experiment log

Why each choice above was made, and what was measured. Paths not taken are here
too, because a null that cost a job is worth as much as a result.

### The few-shot probe — the operating point moved, nothing else did

662 calls, 24 cases. Claims 583 → 322 (−45%), precision 0.129 → 0.168, recall
0.347 → 0.250, Final Score 0.3241 → 0.3323, paired 95% CI **−0.038 to +0.050**.

Fewer claims, more of them right, more real findings missed, no measurable
movement in the score — a precision/recall trade along one ROC curve. Every arm
since is priced against that, and it is why §5 reads Final Score first and arm 0
third. Survey recall fell 0.40 → 0.28 while RadFact logical recall moved
+0.0023: the claims it dropped were ones the reports never made. **Trade-off
arguments belong on the official metric, not on the survey's recall column.**

### Why the vision path — int4 is blind to the overlay (`6e3f323`)

`Qwen3.5-9B-int4-AutoRound` answers fluently and invents pathology: `A008` tooth
11, a normal incisor, returns `is_remnant`, `complete_bony_inclusion`. bf16 on
the same container, prompt and temperature answers correctly. The mechanism is in
the evidence strings — int4 describes grayscale only, while bf16 opens *"The RED
OUTLINE encompasses a single-rooted tooth structure"*. Every generator draws
overlays to say which structure to read, so a model blind to them reads the wrong
object.

A perceptual defect producing over-calling, observed directly and invisible to
every text-level check. Whether the bf16 model over-calls for the same reason was
never established — that was arm 3's job, and arm 3 was cut.

### Resolution is not the constraint — a measured null

Is the feature absent from what the model can resolve, or present and
under-sampled? Different fixes, and no LoRA substitutes for pixels never
rendered. Job 554815, 16 min on one H100: 28 teeth — 20 `post_and_core`
positives and **8 root-treated controls**. Same prompt, same call builder, same
server; only the image differs, `zoom` being a tighter crop at twice the pixels.

**7 teeth fixed, 6 broken — exact McNemar p = 1.000.** No effect. What moved was
the **assertion rate, 25% → 43%**: the extra pixels did not make the feature
legible, they made the model likelier to claim it. Re-rendering will not fix
`post_and_core`.

**The 8 controls are what made it readable.** Without them the run reads as
"recall 0.30 → 0.45, resolution confirmed" and buys a pool re-render plus an arm.
n = 28 on one field, so this closes resolution *for `post_and_core`*, not
resolution generally. `zoom_probe.py` has since been deleted.

### Why the arm patterns are anchored

Two silent traps, both found before they cost a run. **The checkpoint is not
named like Qwen2-VL** — `model.visual.*` and `model.language_model.*`, not
`visual.*` and `model.layers.*`, which is the naming every LoRA recipe on the
internet was written against. A copied `target_modules` matches nothing and
`get_peft_model` reports 0 trainable parameters **without erroring**: the job
runs its wall clock and writes an empty adapter.

**The `mtp` head answers to the language pattern.** The multi-token-prediction
head carries its own `mtp.layers.0.self_attn.{q,k,v,o}_proj`, so an unanchored
`.*self_attn\.[qkvo]_proj` trains **nine** blocks where a language arm declares
eight — and a parameter-matched contrast is gone without a word in the log.

Hence `^`, `re.fullmatch`, and `assert_arm()`.

### The refusal list came from the wrong statistic

Every refusal used to rest on a low inter-reader Jaccard. **That is a coverage
number, not a disagreement.** Reader A writes *"post and core on 16"*; reader B
writes about the sinus and never mentions 16 — pairwise Jaccard 0, and nothing is
in dispute. **So the union is the label**, verified 341/341 across the
multi-reader training cases.

Re-derived on positive counts, only two refusals survive, and both on count:
`with_root_fracture` (4 positives in 4 cases — one case moves the whole field
under a whole-case split) and `furcation_involvement` (0 positives in 1,785
slots — the only thing it can teach is "emit `false`"). Both counts live in
`DEFAULT_REFUSED` beside the fields.

### Why the absent-tooth nulls matter more than any metric

Applying §3.2's rule removed **53,586 fabricated negatives** on training and
4,670 on validate — ~11 per absent tooth — while keeping 281 findings at
edentulous sites and changing **0** positions on a present tooth. Under the old
GT those were supervised negatives on positions with no referent, which is a
direct route to the "emit `false`" degenerate solution the cap exists against.

### Why negatives are masked rather than dropped

Field ratios differ wildly while a call carries every field at once, so dropping
enough calls to fix one field deletes another's positives. And dropping the
*all-negative* calls specifically is worse than useless: a tooth with no findings
is exactly what the model must learn to call normal, so removing those **is**
rebalancing toward positives, which is the thing forbidden two lines earlier.

### Why the curated pool was abandoned

The 144-case pool was three filters. `score_asis` is mask-derived and its
heaviest channel is presence, which `no_gt_block` already drops per tooth. The
audit rejects nothing — 0 ERRORs across all 1,274 stage-1 files. Prefix balance
became a stated risk instead of a filter, in exchange for 4.8× the data.

### The image is hoisted, so caption-before-image has never run

§3.4's check failed the first time it ran, and the failure was in the pipeline
rather than the collator. vLLM resolves a chat-template **content format** per
server by inspecting the template's Jinja AST, guesses `string` for Qwen3.5's,
and in that mode hoists every image placeholder to the front of the user turn,
joining the text parts after it. Host 6,126 tokens against 6,128 served; the two
extra tokens are newlines that appear in no message anywhere.

So the caption-before-image design `build_captioned_image_blocks` exists for —
*"so the model knows what it's looking at before it sees the pixels"* — **has
never reached the model, in any run in this repo**, the baseline included. It
also means nothing after the system turn was ever a common prefix, so
`--enable-prefix-caching` had almost nothing to hit.

**The collator matches the server, not the intent.** Restoring the intended order
would change the prompt every scored run has used and invalidate every baseline.
Making caption-before-image real is a separate measured experiment.

### The teacher pass that never saw an image

`qa_pairs.jsonl` stores image paths relative to the project dir so one payload
resolves on the host and at `/project` in the container.
`build_captioned_image_blocks()` resolves them against the **working directory**
and silently drops what it cannot find. The job set no working directory, so the
teacher wrote **9,301 fluent, specific-sounding strings having never seen an
image** — memory dressed as observation, manufactured by the harness.

It surfaced only because the screen refuses to judge perceivability with no image
and therefore made *zero* requests — and then reported "9301 string(s) kept",
because an unjudged string counted as neither pass nor fail.

**A silent degradation in the harness is indistinguishable from a model result**,
and the only defence is a component that refuses to proceed rather than one that
copes. `sft_prompt.py` and `check_prompt_parity.py` both cite this as the reason
they raise.

### AWQ quantizes almost nothing, so the merge is exact

`modules_to_not_convert` covers `visual`, `linear_attn`, `self_attn`,
`model.layers.0.` and `mtp` — only `model.language_model.layers.{1..31}.mlp.*`
are 4-bit, 279 quantized tensors of 961. The vision tower and every `self_attn`
are **bit-identical** to the bf16 checkpoint, same sha256 on
`model.visual.blocks.0.attn.qkv.weight`. So an arm is fit on exactly the tensors
it is merged into and the merge is numerically exact, not an approximation.

What stays open is behavioural: the adapter is fit with bf16 language MLPs
downstream and served with 4-bit ones. AWQ measured against bf16 on
byte-identical images at mean F1 **0.572 vs 0.552** — no measurable fact-level
cost — and every validate run *is* the transfer measurement, since it serves the
merged checkpoint.

### Why training never loads AWQ

Four jobs in 25 minutes on 2026-08-14, none reaching a forward pass:
`gptqmodel` missing, then `AwqMarlinLinear: out_features 4304 must be divisible
by 64`, then a Marlin JIT wanting ninja and a CUDA toolkit — and only then the
real question of whether backward works through Marlin kernels at all.

Installing `gptqmodel` made it worse rather than better. It **monkey-patches
Triton's autotuner globally at import**, and Qwen3.5's GatedDeltaNet layers run
on `fla`'s Triton kernels, whose `CachedAutotuner` then dies on a `_cache_lock`
only gptqmodel's patched `__init__` sets. Every forward hits it. And peft imports
gptqmodel during `get_peft_model` — `dispatch_awq()` calls
`is_gptqmodel_awq_layer()` per target module — so merely having it *installed*
broke a bf16 run too.

Three lesser traps, for anyone who retries this:

- `modules_to_not_convert` means **substring** to vLLM and **prefix-or-suffix**
  to transformers. The shipped list is written for vLLM, so under transformers
  none of its patterns fire and it tries to 4-bit all 287 linears. It surfaces as
  Marlin complaining about `out_features 4304` — the *vision* intermediate size,
  a tower never meant to be converted, which reads as a kernel problem and
  invites a useless backend override.
- Kernel selection is **per device**. On the A100 `auto` chose `AwqMarlinLinear`,
  whose `SUPPORTS_TRAINING` is `False`. It failed loudly only because Marlin
  JIT-compiles and there is no nvcc; a prebuilt Marlin would have run and
  returned **no gradient**.
- `gptqmodel 7.3.2` cannot be imported next to transformers 5.9.0.

### The memory budget, and one figure that was wrong

The budget: frozen bf16 base 19.31 GB, LoRA + optimizer ~0.13 GB, activations
with both towers checkpointed ~2.7 GB, GatedDeltaNet scan ~0.5 GB on `fla`, and
logits **~0.9 GB with `logits_to_keep`** against ~9 GB without — 6,200 positions
× a 248,320 vocab is what OOMs the job, not the weights. Total 25–28 GB.

**A stage-B probe reported 60.28 GiB and the budget was briefly declared wrong.
That figure is withdrawn.** `from_pretrained` leaves the model in **eval** mode
and `GradientCheckpointingLayer.__call__` gates on `self.gradient_checkpointing
and self.training`, so the probe ran with checkpointing silently inactive and
kept every activation. The Trainer calls `model.train()`; the real peak, printed
after `trainer.train()`, is **22.15 GiB**. The 25–28 GB arithmetic was right, and
a 40 GB card is enough — which matters because the a100 pool has 21 cards against
a100-80gb's 8.

Step time measured at **36.0 s** (78 steps, job 555445) on an 80 GB SXM4 card,
whose bandwidth is roughly 2× the 40 GB PCIe part — treat it as an optimistic
bound, not a transferable number.

### The overlay-mention zero point

The fraction of `visual_evidence` strings citing the coloured overlay, recorded
2026-08-14 **before any arm was scored** — a comparison without a zero point is
not a comparison. Pattern, case-insensitive over every string:

`\b(red|colou?red|highlight\w*|overlay|outline[sd]?|segmentation|mask(?:ed)?|contour\w*|delineat\w*)\b`

| arm | cases | strings | overlay-mention |
|---|---|---|---|
| **AWQ validate-40 — the submission model** | 40 | 4,967 | **27.56%** |
| bf16 validate-40 | 40 | 4,881 | 28.35% |
| AWQ held-out 22, tooth calls only | 22 | 2,492 | 27.85% |

**The three agreeing to within 0.8 points is what makes it usable** — the rate is
stable across quantisation *and* across case set, so it has no drift of its own
and a move is attributable to the arm. Read it as a **guard, not a target**: int4
failed by describing an overlay it could not see, so a large rise is as
suspicious as a collapse. A sharp fall would say the arm learned to answer
without looking at the outline the pipeline depends on.

(`outputs/training_results/overlay_baseline.json` holds the same numbers, but
`outputs/` is gitignored, so this table is the copy that survives.)

### Score every arm against the baseline, never against another arm

Arm 2 was first read as a large win because it was compared against **arm 1**,
which sits below baseline on nearly every field. Against the actual baseline it
is a wash — `with_fillings` +0.091 and `with_full_crown` +0.047, against
`eruption_state` −0.074 on N=405, `with_endo` −0.079, `canal.location` −0.058.
**The reference point flipped the conclusion, not the data.**

### One epoch is nearly as good as two

Arm 5's eval loss, twice per epoch: **0.2725 → 0.2531 → 0.2506 → 0.2469**,
monotonic, never rising. The second half of epoch 1 bought 7.1%; the whole of
epoch 2 bought **2.4%**. On a ~12 h run that is six hours for the last 2.4%, and
it also sits further inside the ≤2-epoch forgetting bound.

### Probe `lora_B`, not `lora_A`

peft initialises `lora_B` to zeros, so `dL/dA ∝ Bᵀ = 0` and **every `lora_A`
gradient is exactly zero at step 0 on every model**, quantized or not. A probe
reading `lora_A` reported "gradients do not reach the arm" on a plain bf16
checkpoint containing no quantized weight to blame, and burned a job on it. The
tell was that `.grad` was present but zero, not `None`. Read `lora_B` at step 0,
or `lora_A` after one optimizer step.

### Arm 5 → arm 6: the arch rows, and one freed constant

Arm 6 is arm 5 plus two target-build changes — the nine `global` calls built into
rows for the first time, and the two `maxilla_sinus_*.scope` constants freed from
`ARCH_REFUSED`. They separated cleanly:

| group | base | arm 5 | arm 6 | a6−a5 |
|---|---|---|---|---|
| `[3d]` | 0.5851 / 0.6101 | 0.5458 / 0.5800 | **0.6772 / 0.6753** | **+0.1315 / +0.0953** |
| `[sinus]` | 0.7807 | 0.5965 | **0.9298** | **+0.3333** |
| `[detail]` | 0.3743 / 0.6992 | 0.4730 / 0.7601 | **0.4893 / 0.7786** | +0.0163 / +0.0186 |
| `[panoramic]` | 0.4311 / 0.5576 | 0.4344 / 0.6164 | 0.4489 / 0.6072 | +0.0145 / −0.0093 |

`[3d]` recovered past baseline with no constant freed there, so that is the arch
rows alone. The whole `[sinus]` move is one field: `maxilla_sinus_*.scope` went
0.000 → 1.000 while `mucosa_state`, `sinus_content` and `intrasinusal_teeth` are
unchanged to three decimals.

**Be clear what that 1.000 is worth.** The GT is `partially_included` on 80 of 80
sides because that is `_resolve_scope`'s literal fallback in
`parse_reports_to_gt.py`, not an observation. The score says which constant the
model settled on — arm 5 emitted `fully_included` 38 times, arm 6 emits
`partially_included` 38 times. Worth having, since a wrong constant is 38 wrong
sentences, but it is **not** evidence the model can read sinus extent and must
not enter an aggregate as a recovered clinical finding.

And the tooth calls did not pay for it: `[detail]` +0.016 on validate and −0.008
on the held-out 22, flat within noise in both directions.

### Eval loss does not compare across arms

Arm 6's curve (0.2258 → 0.2132 → 0.2102 → 0.2072) looks far better than arm 5's
(0.2725 → 0.2531 → 0.2506 → 0.2469) and the difference means nothing. The two
eval sets share only 6 of 30 cases — the split samples a case pool that grew from
527 to 558 — and arm 6's is 37% arch rows by weight mass, which carry no evidence
prose. An arch loss of ~0.147 reproduces the whole gap with the tooth loss
unchanged. Compare arms on the fact-based eval or the official ranking, never on
the curve.
