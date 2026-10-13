"""Config dataclasses for the report-generation model.

Experiments are expressed as config axes rather than as separate files:
  perception.backbone   — dental (default) | dental + tf2 hybrid
  train.findings_head   — findings head off | on
  train.use_siglip      — region-text contrastive alignment (default on)

The default schedule is a two-stage align -> generate run on top of a frozen segmentation
network: stage 1 = projector align (LoRA frozen, for freeze_llm_epochs epochs), stage 2 =
generate (LoRA on). The findings head and the contrastive loss are trained through both
stages. YAML overrides are shallow-merged; the resolved config is dumped next to the
checkpoint so a run can be reproduced from it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# DentalSegmentator seg-head indices (bg=0): 1 maxilla/upper-skull, 2 mandible,
# 3 upper_teeth, 4 lower_teeth, 5 canal. Teeth handled by tooth-query branch.
SEG_LABELS = {1: "maxilla", 2: "mandible", 3: "upper_teeth", 4: "lower_teeth", 5: "canal"}
# [38] = 4 non-teeth (maxilla·mandible·nerve_R·nerve_L) + 32 FDI + 2 CLS
N_TOKENS = 38


@dataclass
class PerceptionConfig:
    backbone: str = "dental"           # dental (6-class) | hybrid (dental non-teeth + tf2 teeth)
    model_dir: str = ("models/seg/Dataset112_DentalSegmentator_v100/"
                      "nnUNetTrainer__nnUNetPlans__3d_fullres")
    fold: int = 0
    checkpoint: str = "checkpoint_final.pth"
    feat_dim: int = 64                 # [encoder-stem; decoder-penultimate] 32+32
    cache_dir: str = "experiments/feature_cache/region_features"
    # tf2 per-tooth density bag (whole_pool 32x64, centre, mass, present; row i = FDI_ALL[i]).
    # With backbone="hybrid" these per-tooth vectors are injected into the teeth K/V.
    tf2_teeth_cache_dir: str = "experiments/feature_cache/tf2_tooth_density"
    # Inject the tf2 density channels (lo/hi/sat sub-pools) into the tooth query as extra
    # diagonal K/V tokens, alongside the whole-tooth tf2 vector. When True, tf2_teeth_cache_dir
    # must hold npz files that also contain the sub-pool centroids (_soft_c); a whole-only
    # cache behaves exactly like the baseline.
    tf2_density: bool = False

    # --- soft clip -------------------------------------------------------------- #
    # Upper soft-clip scale, which preserves metal and endodontic material (softclip_norm).
    # > 0 replaces CTNormalization.run; 0 keeps the hard upper clip (a gain-bias-only
    # ablation goes through the same hook).
    soft_clip_s: float = 0.0


@dataclass
class EncoderConfig:
    """aggregation(gated attention + tooth-query) + cross-label self-attention."""
    pe_k: int = 6                      # sinusoidal PE freqs per axis -> 3*2*pe_k dims
    d_model: int = 128
    n_heads: int = 4                   # tooth-query cross-attention heads
    gate_hidden: int = 128             # SiLU gated-attention hidden width
    dropout: float = 0.1
    cl_layers: int = 2                 # cross-label self-attention layers (1~2)
    cl_heads: int = 4
    cl_mlp_hidden: int = 256
    cl_dropout: float = 0.1
    use_presence_emb: bool = True      # presence (0/1) embedding for absent tokens
    use_in_proj: bool = True           # cross-label 1-layer fuse MLP
    # Drop the two arch-summary slots from the model: they are excluded from generation and
    # removed from the keys of the cross-label attention. Rationale: the model never emitted
    # anything for them (0 of 124 validation slots), and arch-level findings were moved onto
    # the bone tokens in the ground truth. False (default) is numerically identical to the
    # earlier behaviour.
    drop_arch_cls: bool = False


@dataclass
class RealizerConfig:
    llm_name: str = "Qwen/Qwen3.5-9B"
    quant: str = "4bit"                # 4bit (QLoRA, bitsandbytes; sm_120 verified) | bf16 | none
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    n_prefix_per_token: int = 1        # visual-prefix tokens emitted per anatomic token
    proj_hidden: int = 512             # visual-prefix MLP hidden, decoupled from LLM hidden
    dropout: float = 0.1               # visual-prefix projector dropout
    max_text_len: int = 512            # per-structure decode: region target token cap
    grad_checkpointing: bool = True
    use_liger: bool = True             # fused-linear-CE (large-vocab per-region OOM fix)
    system_prompt: str = ("You are an expert oral and maxillofacial radiologist. Report the "
                          "radiographic findings from the dental CBCT accurately and concisely, "
                          "using standard FDI tooth notation. A tooth number's first digit is its "
                          "quadrant (1 upper-right, 2 upper-left, 3 lower-left, 4 lower-right) and "
                          "second a position 1 to 8; use no other. Write each tooth number as two "
                          "digits with no dot (e.g., 33, not 3.3).")


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 2e-4
    weight_decay: float = 1e-4
    # The align and generate stages each run at most 200 epochs with early stopping. `epochs`
    # is the overall cap (align 200 + generate 200); where each stage actually ends is decided
    # by its own early stopping.
    epochs: int = 400
    batch_size: int = 1               # per-structure decode; effective batch comes from grad_accum
    grad_accum: int = 4
    early_stopping_patience: int = 5  # generate stage early-stop (0 = off)
    run_name: str = ""                # output subdir + wandb name

    # --- experiment axes ------------------------------------------------------- #
    findings_head: bool = False       # train the findings head alongside generation
    use_siglip: bool = True           # region-text contrastive alignment

    # --- decode path ----------------------------------------------------------- #
    # per_region: every anatomical token generates its own region text (the current path,
    #   with the EOS gate deciding which slots stay silent).

    # --- align -> generate, two stages over a frozen backbone ------------------- #
    # stage 1 = projector align (LoRA frozen) -> stage 2 = generate (LoRA on). Each stage runs
    # at most 200 epochs with early stopping. freeze_llm_epochs caps the align stage.
    freeze_llm_epochs: int = 200
    stage_early_stop: bool = True     # end align stage on a val plateau (cap above)
    stage_patience: int = 0           # 0 = early_stopping_patience
    # --- four-stage curriculum -------------------------------------------------
    #   Training is split into four stages so that each stage optimises a single objective
    #   instead of a mixture of losses.
    #     1) perception  — train the encoder and findings head on the gate loss only
    #                      (separate trainer, stage_a.py)
    #     2) prefix      — freeze 1) and the LLM, train only the soft prefix (projector)
    #                      on L_report
    #     3) perception_prefix — fine-tune perception + prefix on L_report (LLM still frozen)
    #     4) qlora       — freeze perception and prefix, train only the LLM QLoRA on L_report
    #   stage="align_generate" (default) keeps the older two-stage behaviour.
    #   Stages 2 to 4 run on plain L_report, with no findings head and no soft injection; the
    #   findings head belongs to stage 1 only. Claim information reaches the LLM solely through
    #   the encoder output (ctx).
    #     5) prefix_qlora — train the prefix (projector) and the LLM QLoRA *together* on
    #        L_report (perception stays frozen). The prefix stage aligns the projector against
    #        a base LLM whose LoRA is still the identity (lora_B = 0); once the LLM LoRA stage
    #        pushes the LLM away from that base, the projector cannot follow and the two end up
    #        mismatched. This stage closes that gap.
    stage: str = "align_generate"     # align_generate | prefix | perception_prefix | qlora | prefix_qlora
    init_from: str = ""               # previous-stage checkpoint (.pt), loaded strict=False
    # Learning rate applied only to the prefix (projector, region_type_emb, prob_proj). 0 means
    # a single learning rate (train.lr) for everything, as before. Giving an already converged
    # projector the same learning rate as a freshly started LoRA has repeatedly led to the
    # projector re-flattening (soft-prefix collapse) in this project, so it is lowered here.
    prefix_lr: float = 0.0

    # --- findings loss (when findings_head is on) ------------------------------ #
    lambda_gate: float = 0.5          # weight of the findings loss added to the per-region token CE
    gate_mode: str = "soft"           # soft (inject abnormality prob) | hard (no injection)
    # Write the per-axis probabilities into the prompt *as text*. This is a different mechanism
    # from the continuous vector injection (gate_mode: soft): diagnostics showed the information
    # already reaches the LLM but greedy decoding does not act on it, i.e. the bottleneck is the
    # decision rule, and for the LLM to learn that threshold the probabilities have to be in a
    # readable form. Applied uniformly to every slot and every axis, as probabilities. Requires
    # findings_head.
    prob_text: bool = False
    # Which claim vocabulary the probabilities are read in: merged = the 29 merged classes,
    # raw = the 38 classes before merging. This also fixes the output dimension of the findings
    # head — if it disagrees with the checkpoint, loading fails immediately.
    gate_class_weight: bool = True    # effective-number class-weight (train dense frequencies)
    gate_class_weight_beta: float = 0.99
    gate_state_weight: float = 1.0    # multiplier on the state axis (tooth presence/state)
    gate_focal_gamma: float = 0.0     # focal (1-p_t)^γ (0 = exactly CE/BCE)

    # --- region decode --------------------------------------------------------- #
    region_chunk_size: int = 0        # >0: decode regions in chunks of K (same gradient)

    # --- EOS gate: the model decides for itself which regions to speak in ------- #
    # Every imaged tooth (state != not_available) and every non-tooth region is fed to the LLM;
    # normal regions get an empty target and learn to emit EOS immediately, regions with findings
    # generate text. At inference the model gates itself.
    # False-EOS penalty: normal (EOS) regions dominate, so up-weight the loss of regions with
    # findings to avoid collapsing to "always EOS". Effective-number class weight over the
    # region type (normal vs finding) times λ (recall strength, tuned on a small grid).
    false_eos_weight_beta: float = 0.99
    false_eos_lambda: float = 1.0
    # Write an epoch{N}.pt after every epoch. When a full-data retraining has to decide "how
    # many epochs to stop at" without a validation split, measuring that sensitivity directly
    # with RadFact requires the intermediate weights (best/last alone cannot show it). With
    # LoRA only, each file is about 116MB.
    save_every_epoch: bool = False
    # Feed out-of-field (not_available) teeth as regions as well, with an empty target so they
    # learn to emit EOS. False keeps the older behaviour and its evaluation reproducibility;
    # only new arms turn this on. When enabling it, false_eos_lambda must be raised (about 1.6
    # restores the balance) — see dataset.py.
    eos_include_not_available: bool = False
    # Metric used for best-checkpoint and early-stop decisions. 'loss' = total validation loss
    # (older behaviour), 'finding' = token CE of the regions with findings only
    # (l_region_finding). With eos_include_not_available=True the normal-region tokens grow by
    # about 33%, so the total loss drops mechanically through dilution by easy EOS targets;
    # 'finding' is preferred in that case.
    best_metric: str = "loss"
    # Perception stage only (align_stage1): when best_metric=vision_gap, restrict the choice to
    # --- contrastive (when use_siglip is on) ----------------------------------- #
    lambda_siglip: float = 0.2
    siglip_text_encoder: str = "neuml/pubmedbert-base-embeddings"
    siglip_text_dim: int = 768        # PubMedBERT hidden size (checked against encoder.dim)
    siglip_proj_dim: int = 256
    siglip_hard_neg_mask: bool = False
    # Text-encoder input length. 128 (the old hard-coded value) was harmless when each slot
    # holds one or two sentences, but in arms that gather sentences onto a single token it
    # truncates 27.9% of the mandible text and loses 13.4% of all sentences. At 512 nothing
    # exceeds the limit once duplicates are removed.
    siglip_text_max_len: int = 512
    # All-gather the embeddings across ranks. SigLIP is a sigmoid pairwise loss, so building an
    # independent matrix per rank costs a factor of world_size in contrastive pairs. The
    # embeddings are small, (M, dim), so the gather cost is negligible.
    siglip_gather: bool = True

    # --- ground-truth sources -------------------------------------------------- #
    # Dense per-tooth categorical claim GT, read by build_axis_labels, the class weights and the
    # claim evaluation. Default = the merged per-patient claim GT (627 patients).
    dense_dir: str = "experiments/GT/categorical_GT/merged"
    # Source directory for the region target text the realizer generates (the anatomical
    # buckets teeth/maxilla/mandible/nerve), one file per report.
    anat_dir: str = "experiments/extraction/anatomical"


@dataclass
class GRPOConfig:
    """Rollout and rejection-sampling hyper-parameters, read by `toothfairy.rl.rollout` and the
    RFT pipeline. The block is called `grpo:` in the YAML for historical reasons — it was written
    for group-relative policy optimisation, and renaming the key would break existing configs.

    The supervised trainer (training/trainer.py) does **not** read this section. A config that
    omits these keys keeps every default, so behaviour is unchanged (additive, backward
    compatible).
    """
    run_name: str = "sinus_rft"
    seed: int = 42
    # Starting policy: the LLM LoRA checkpoint that rollouts are generated from.
    init_from: str = "models/sinus_llm_lora/best.pt"

    # --- rollout sampling ----------------------------------------------------- #
    n_finding_regions: int = 8         # 8 regions with findings
    n_normal_regions: int = 4          # 4 normal regions (silence = 1, speaking = 0)
    # For a patient with fewer than 8 regions with findings, keeping 4 normal regions inverts
    # the intended 2:1 ratio and disables the safeguard, so the number of normal regions is
    # reduced to match and the 2:1 ratio is preserved.
    preserve_ratio_when_short: bool = True
    group_size: int = 6                # G
    temperature: float = 1.0           # exploration sampling (greedy = zero group variance)
    top_p: float = 0.95
    max_new_tokens: int = 96
    # repetition_penalty is deliberately **not** used: it makes the sampling policy differ from
    # the log-probability policy, which breaks GRPO.

    # --- optimisation --------------------------------------------------------- #
    lr: float = 1e-5                   # much smaller than the 2e-4 used for supervised training
    weight_decay: float = 0.0
    grad_accum: int = 1                # patients per optimizer step
    max_grad_norm: float = 1.0
    total_steps: int = 1000
    advantage_norm: str = "std"        # std (divide by the group standard deviation) | none
    adv_std_eps: float = 1e-4          # below this std a group counts as zero-variance

    # --- KL (reference = a second frozen adapter holding the LLM LoRA weights) -- #
    kl_beta: float = 0.04
    kl_estimator: str = "k3"           # k3 (low variance, >= 0) | k1 (unbiased, noisy)
    kl_delta_clamp: float = 10.0       # upper bound that keeps exp() from blowing up

    # --- runtime / memory ------------------------------------------------------ #
    logprob_micro_batch: int = 6       # completions per log-prob forward (= G: one prompt pass)

    # --- reward --------------------------------------------------------------- #
    # Reference used for reward scoring = the **union of all reports** for the patient.
    #   union   — the region texts of every report for that patient, concatenated with
    #             duplicate sentences removed and order preserved. The reference of the final
    #             RadFact evaluation is already a union, so reward and evaluation then score
    #             against the same target. The finding/normal split (the 8:4 sampling) follows
    #             the same definition.
    #   sampled — the older behaviour (the single report drawn for that epoch). Kept **only as
    #             a control**. Under it, 13.7% of the normal slots in the training set are in
    #             fact findings in another report of the same patient, so the rule "empty
    #             target, so speaking scores 0" suppresses real findings (the reward rises
    #             while RadFact does not).
    reward_gt: str = "union"           # union (default) | sampled (control)
    # A **tooth** presence statement ("Tooth 31 present") may be written or omitted, so it is
    #   neutralised in scoring. In the ground truth only 45.5% of state=normal teeth record
    #   presence, which put opposite targets (silence 1.0 / speaking 0.0) on identical inputs.
    #   Arch-level presence statements are **not** neutralised. See rl.presence.
    reward_neutralize_presence: bool = True
    # After presence statements are removed, GT regions from which the rule-based extractor can
    #   build no claim at all are dropped from scoring: there, claim F1 degenerates to empty vs
    #   empty = 1.0, so any output would score full marks. On the validation union this covers
    #   56 tooth regions and 131 non-tooth regions.
    #   Warning: most of the non-tooth cases are field-of-view or negated findings that the
    #   extractor does not implement, so those regions drop out of RL training.
    reward_drop_claimless: bool = True

    # --- validation / checkpointing -------------------------------------------- #
    eval_every: int = 50               # step interval of the greedy val reward (stopping)
    eval_patients: int = 8
    eval_patience: int = 6             # number of validation evaluations without improvement
    save_every: int = 25               # interval (steps) for writing resume.pt, in case of a crash


@dataclass
class ReportModelConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    realizer: RealizerConfig = field(default_factory=RealizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    @property
    def n_tokens(self) -> int:
        return N_TOKENS               # fixed at 38 anatomical tokens

    @property
    def d_model(self) -> int:
        return self.encoder.d_model


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> ReportModelConfig:
    """Build a ReportModelConfig from defaults <- YAML file <- kwargs."""
    base = asdict(ReportModelConfig())
    if path is not None:
        y = yaml.safe_load(Path(path).read_text()) or {}
        base = _merge(base, y)
    if overrides:
        base = _merge(base, overrides)
    realizer_over = base.get("realizer", {})
    perception_over = dict(base.get("perception", {}))
    return ReportModelConfig(
        perception=PerceptionConfig(**perception_over),
        encoder=EncoderConfig(**base.get("encoder", {})),
        realizer=RealizerConfig(**{**realizer_over,
                                   "lora_targets": tuple(realizer_over.get(
                                       "lora_targets", RealizerConfig().lora_targets))}),
        train=TrainConfig(**base.get("train", {})),
        grpo=GRPOConfig(**base.get("grpo", {})),
    )


def dump_config(cfg: ReportModelConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = dataclasses.asdict(cfg)
    d["realizer"]["lora_targets"] = list(cfg.realizer.lora_targets)
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
