# Training recipe

Every command is run from the repository root, with the package installed (`pip install -e .`)
or with `PYTHONPATH=src`. See [DATA.md](DATA.md) for what has to be on disk first.

Wall-clock on the machine the model was trained on (H100 80GB): caches a few hours, stage 1
minutes, stage 2 about 6 hours on two ranks, stage 3 about 3 hours, RFT about 6 hours.

## 1. Split

```bash
python -m toothfairy.data.make_splits
```

## 2. Feature caches

```bash
python -m toothfairy.cli.cache                      # both networks, every volume, resumable
```

This is the expensive, one-off step: two frozen segmentation networks over every volume.
Restarting the command skips volumes whose `.npz` already exists.

## 3. Curriculum

```bash
python -m toothfairy.cli.train --gpus 0             # single GPU
python -m toothfairy.cli.train --gpus 0,1           # stages 2 and 3 on two ranks
```

Three stages, each optimising one objective and starting from the previous checkpoint:

| stage | output dir | trains | objective |
|---|---|---|---|
| 1 perception | `models/sinus_perception` | aggregator + findings head (0.41M) | categorical claim loss + 0.5 x region–text SigLIP |
| 2 prefix | `models/sinus_prefix` | projector (2.18M) | per-slot token cross-entropy |
| 3 llm_lora | `models/sinus_llm_lora` | LoRA (29.1M) | per-slot token cross-entropy |

A stage whose directory already holds `done.flag` is skipped, so the same command resumes an
interrupted run. `--stage perception|prefix|llm_lora` runs one stage on its own.

With two ranks the DDP configs are picked up automatically. They differ from the single-GPU
configs in `grad_accum` only (4 -> 2), which keeps the effective batch at 4; without that the
effective batch would double and a change of batch size would be mixed into the change of
parallelism.

From stage 2 on there is no findings head, so claim information reaches the language model
only through the anatomical tokens.

## 4. Rejection-sampling fine-tuning

The LoRA is refined on its own samples: sample each slot several times, score each sample
against that slot's reference with an entailment judge, keep what clears the threshold, and
fine-tune on the survivors for one epoch.

Steps 4.2 and 4.3 need two things this repository does not vendor: an OpenAI-compatible
endpoint (the submitted model used a locally served `Qwen3-32B-AWQ`) and two third-party
checkouts, which are expected at these paths because the code puts them on `sys.path`:

| path | repository | used by |
|---|---|---|
| `tools/radfact_lite/` | [AImageLab-zip/radfact_lite](https://github.com/AImageLab-zip/radfact_lite) — the organisers' scorer | 4.2, for the normal-finding filter |
| `tools/RadFact/` | [microsoft/RadFact](https://github.com/microsoft/RadFact) | 4.3, for the entailment prompt and schema |

Without them those two steps fail at import. Steps 4.1, 4.4, 4.5 and 4.6 do not need either.

```bash
RFT=experiments/analysis/rft

# 4.1 sample the policy (GPU; shardable with --shard i --num-shards N)
python -m toothfairy.cli.rft rollout --all-regions --threshold 0.3 --maint-ratio 0 \
    --reward-gt sampled --out-dir $RFT

# 4.2 cache the official finding_filter verdicts for both sides, then fold the shards in
python -m toothfairy.cli.rft filter-cache --dumps $RFT/rft_groups.shard*.jsonl --out $RFT --sides both
python -m toothfairy.cli.rft filter-cache --out $RFT --merge

# 4.3 re-score the dump with the entailment judge, through that filter (no regeneration)
python -m toothfairy.cli.rft rescore --out-dir $RFT --pure-radfact --filter-cache $RFT/filter_cache.json

# 4.4 select what to train on — reads the re-scored groups, generates nothing
python -m toothfairy.cli.rft select --groups $RFT/rft_groups_llmjudge.jsonl \
    --maint-ratio 0.5 --exclude-cls --out-dir $RFT

# 4.5 one epoch of LoRA fine-tuning on the selection
python -m toothfairy.cli.rft finetune --selected $RFT/rft_selected.jsonl \
    --lr 2e-5 --epochs 1 --maint-weight 1 --out-dir models/sinus_rft

# 4.6 fold the fine-tuned LoRA onto the starting policy -> one scorable checkpoint
python -m toothfairy.cli.rft merge --base models/sinus_llm_lora --lora models/sinus_rft \
    --out models/sinus_rft_eval
```

Two arguments are worth stating explicitly. `--maint-ratio 0.5` at selection keeps
maintenance samples at half the finding samples; leaving it out lets the finding side drift.
`--maint-weight 1` at fine-tuning is **not** the default — the default `auto` picks a much
larger weight and quietly changes what is being compared.

## 5. Narrative rewriter

A separate QLoRA adapter, trained ground truth -> ground truth on the training split only. It
learns reporting style, not clinical content, and is applied to the slot outputs at the end.

```bash
python -m toothfairy.cli.rewrite pairs --split train --no-cls \
    --out experiments/analysis/rewrite/pairs_train.jsonl
python -m toothfairy.cli.rewrite train --pairs experiments/analysis/rewrite/pairs_train.jsonl \
    --no-cls --out-dir models/rewrite_qlora_nocls
```

`--no-cls` drops the two arch-summary slots, which is how the submitted adapter was trained.
It has to be passed to **both** commands: to `pairs` because it changes the data, and to
`train` because that is what records the fact in `adapter_best/train_meta.json`. Whatever
applies the adapter reads that file to decide whether to feed 36 or 38 slots — if it is
missing or says otherwise, the adapter is used on an input shape it was not trained on, and
the output still looks plausible enough that the mistake is invisible.

Judge the rewriter with RadFact, not with captioning. Having learned the style, it will
almost certainly win on BLEU and METEOR — that is the trap of this step. The question is
whether clinical content was lost or invented.

Evaluation is in [EVAL.md](EVAL.md).
