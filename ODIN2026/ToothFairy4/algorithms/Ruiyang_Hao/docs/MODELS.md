# Model artifacts

Weights are not stored in this source repository. By default, inference looks
under `models/`; set `EVIDENT_MODELS` to another root.

```text
models/
  stage1/
    fold_0/checkpoint_best.pth
  stage2/
    global/fold_0/
      checkpoint_best.pth
      thresholds.json
    relation/fold_0/
      checkpoint_best.pth
      thresholds.json
    tooth/fold_0/
      checkpoint_best.pth
      thresholds.json
  stage3/
    qwen_base/
    qwen_adapter/
      adapter_config.json
      adapter_model.safetensors
      tokenizer.json
```

Each Stage-2 checkpoint embeds the architecture plan needed to reconstruct its
branch. Threshold JSON files contain the released operating point for each branch
label. The Stage-3 base model and adapter must be available locally; inference is
offline and does not download model artifacts at runtime.
