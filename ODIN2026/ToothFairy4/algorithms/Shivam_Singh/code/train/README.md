# `code/train/` — LoRA SFT on the vision tower

## Files

| file | function |
|---|---|
| `sft.py` | entry point: stage table, interpreter choice, vLLM lifecycle |
| `select_sft_pool.py` | writes the pool case lists (`all_cases.txt`, `heldout.txt`, `train_minus_heldout.txt`) |
| `build_sft_targets.py` | qa_pairs + generated GT → `sft_targets.jsonl` (rows + per-field loss mask) |
| `sft_prompt.py` | one row → the messages the pipeline would send |
| `sft_collator.py` | one row → `input_ids`, labels, per-token weights |
| `dataset.py` | `RowDataset`, `load_rows` |
| `lora_arms.py` | the arm table and `assert_arm()` |
| `check_prompt_parity.py` | token-id equality gate, training text vs inference text |
| `trainer.py` | the training loop |
| `merge_vision_lora.py` | adapter → servable AWQ checkpoint |
| `dequantize_awq.py` | unpack AWQ 4-bit MLPs to bf16 |

## Running

One stage per submission via `scripts/run_sft.sh`; everything after `--` is
passed to that stage's module verbatim.

```bash
# 1. pool                                                    [CPU]
STAGE=pool scripts/run_sft.sh -- \
    --out-dir outputs/training_results/sft_pool

# 2. targets                                                 [CPU, ~10 s]
STAGE=targets scripts/run_sft.sh -- \
    --qa-jsonl outputs/training_results/payload_training_582/qa_pairs.jsonl \
    --gt-dir dataset/training/outputs/ground_truth \
    --include-arch --allow-empty-evidence \
    --supervise maxilla_sinus_left.scope maxilla_sinus_right.scope \
    --out outputs/training_results/vsft_pool_training/sft_targets.jsonl

# 3. parity gate                                             [GPU]
STAGE=parity sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/run_sft.sh -- \
    --rows outputs/training_results/vsft_pool_training/sft_targets.jsonl

# 4. train  (the arm-6-no-evidence config)                   [GPU, ~12 h, a100]
STAGE=train sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/run_sft.sh -- \
    --arm vision+language --rank 16 \
    --model models/Qwen3.5-9B \
    --rows outputs/training_results/vsft_pool_training/sft_targets.jsonl \
    --case-list outputs/training_results/sft_pool/train_minus_heldout.txt \
    --epochs 2 --lr 1e-4 --grad-accum 16 --eval-cases 30 \
    --out outputs/training_results/vsft_arm7

# 5. merge                                                   [CPU]
STAGE=merge sbatch --partition=cpu --qos=cpu scripts/run_sft.sh -- \
    --adapter outputs/training_results/vsft_arm7/adapter \
    --out models/Qwen3.5-9B-AWQ-arm7
```

Arms: `merger`, `vision+merger`, `vision+merger-r64`, `language`,
`vision+language`.

Env overrides: `STUDENT_MODEL`, `PORT`, `GPU_MEM_UTIL`,
`MAX_MODEL_LEN`, `MODEL_DIR`, `CONTAINER`, `SFT_PY`, `CONDA_ENV`.
