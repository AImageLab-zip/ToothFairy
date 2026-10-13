"""LLM backend, visual-prefix projector and chat-template assembly.

Shared generation infrastructure. The heavy transformers/peft/Qwen stack is imported LAZILY
inside build_llm_backend, so importing the architecture (and running the smoke tests) works
without it. The chat helpers below expose the same interface
(get_input_embeddings / forward(inputs_embeds, attention_mask, labels) -> .loss/.logits); it is
used to verify visual-prefix injection, loss and gradient flow offline.
"""
from __future__ import annotations


import torch
import torch.nn as nn




def _maybe_apply_liger(llm_name: str):
    """Apply Liger fused-linear-cross-entropy for the model's type (kernel swap only).

    The per-region decode batches all of a patient's regions into one LLM forward; with a
    ~250k vocab (Qwen3.5/Gemma4) the [tokens × vocab] logits tensor OOMs a 32GB card. Liger's
    fused-linear-CE computes the SAME CE without materialising those logits (~0.1% bf16
    reduction-order diff; gradient flow unchanged). Lazy import -> envs without liger just skip
    it. Returns the patched model_type, or None."""
    try:
        from transformers import AutoConfig
        from liger_kernel.transformers import monkey_patch as _mp
    except Exception:                                  # liger not installed in this env
        return None
    hf = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
    cands = []
    for mt in [getattr(hf, "model_type", None),
               getattr(getattr(hf, "text_config", None), "model_type", None)]:
        if not mt:
            continue
        cands.append(mt)
        if "_unified" in mt:                           # gemma4_unified_text → gemma4_text patcher
            cands.append(mt.replace("_unified", ""))
    for mt in cands:
        fn = getattr(_mp, f"apply_liger_kernel_to_{mt}", None)
        if fn is None:
            continue
        try:
            fn(fused_linear_cross_entropy=True, cross_entropy=False)
        except TypeError:                              # patcher without those kwargs
            fn(fused_linear_cross_entropy=True)
        return mt
    return None


def _maybe_torch_gdn_fallback() -> bool:
    """With `TF4_TORCH_GDN_FALLBACK=1`, **disable the accelerated kernels** of the Qwen3.5
    gated-delta-net and fall back to the pure-torch implementation shipped with transformers
    (identical maths, slower).

    On the stack this was validated on (sm_90, torch 2.11+cu128, triton >= 3.4) both accelerated
    paths break
    training:
      * the `causal_conv1d` extension **segfaults**. Minimal repro: `causal_conv1d_fn(randn,
        randn)` -> SIGSEGV. An unmodified trainer dies at exactly the same point, so this is an
        environment problem and not a model problem.
      * `flash-linear-attention` raises a **RuntimeError** in backward: "Triton >= 3.4.0 on
        Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (#640) — install
        tilelang". FLA itself refuses to run because its results are wrong on Hopper, so avoiding
        the path is right — working around it would not be.
    Disabled here: causal_conv1d_fn/update, chunk/fused_recurrent_gated_delta_rule and
    FusedRMSNormGated (transformers has an equivalent torch implementation of each and picks it
    by reading these globals when the module is constructed).
    The real fix is an environment change: rebuild or remove causal_conv1d for this GPU and
    install tilelang. The default keeps the previous behaviour (kernels in use).
    """
    import os as _os
    mode = (_os.environ.get("TF4_TORCH_GDN_FALLBACK") or "").strip().lower()
    if mode not in ("1", "all", "conv"):
        return False
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as _m
    except Exception:                                  # not applicable to other families
        return False
    _m.causal_conv1d_fn = None                         # → F.silu(conv1d(x))
    _m.causal_conv1d_update = None                     # → torch_causal_conv1d_update
    if mode == "conv":
        # Partial fallback: disable causal_conv1d only and keep FLA (gated delta rule,
        # FusedRMSNormGated). The two failures have different causes:
        #   causal_conv1d — the installed .so **does contain an sm_90 cubin**, yet the first call
        #     segfaults immediately (bf16/fp16/fp32 alike; the import itself is fine). So it is
        #     not a missing architecture; the cause is unknown.
        #   FLA — this was FLA's own guard against Hopper + triton >= 3.4, and **installing
        #     tilelang lifts it.**
        # The expensive kernel is the linear-attention one, so this mode recovers most of the
        # speed.
        return True
    _m.chunk_gated_delta_rule = None                   # → torch_chunk_gated_delta_rule
    _m.fused_recurrent_gated_delta_rule = None         # → torch_recurrent_gated_delta_rule
    _m.FusedRMSNormGated = None                        # → Qwen3_5RMSNormGated
    return True


def build_llm_backend(cfg) -> tuple[nn.Module, object, int]:
    """Load Qwen/Gemma + shared LoRA (lazy heavy imports). Returns (model, tokenizer, hidden).

    cfg = RealizerConfig. quant: bf16 | 4bit (QLoRA via bitsandbytes — verified on RTX 5090
    sm_120) | none. use_liger: apply fused-linear-CE so large-vocab per-region batches fit.
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if _maybe_torch_gdn_fallback():
        import os as _os
        _mode = (_os.environ.get("TF4_TORCH_GDN_FALLBACK") or "").strip().lower()
        print(f"[realizer] Qwen3.5 GDN fallback (TF4_TORCH_GDN_FALLBACK={_mode}) — "
              + ("causal_conv1d disabled only; FLA still used through tilelang"
                 if _mode == "conv" else
                 "pure-torch transformers implementation instead of causal_conv1d/FLA"), flush=True)

    if getattr(cfg, "use_liger", True):
        patched = _maybe_apply_liger(cfg.llm_name)
        if patched:
            print(f"[realizer] Liger fused-linear-CE applied to '{patched}' "
                  f"(large-vocab per-region OOM fix, kernel-only)", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg.llm_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"trust_remote_code": True}
    import os as _os, json as _json
    # TF4_STREAM_LLM_LOAD=1: stream the shards to the GPU one at a time to lower peak host RAM.
    # Meant for inference on a host with little RAM (the evaluation node capped it at 32GB).
    # With the variable unset (training) the load path is unchanged.
    stream_load = _os.environ.get("TF4_STREAM_LLM_LOAD") == "1"
    # A pre-quantised 4bit checkpoint (quantization_config in config.json) is loaded exactly as
    # stored instead of being re-quantised (~7GB on disk, small shards -> safe to mmap). The
    # training path loads the bf16 weights and converts them to 4bit at load time.
    prequant = False
    _cfg_json = _os.path.join(cfg.llm_name, "config.json")
    if _os.path.isfile(_cfg_json):
        try:
            prequant = "quantization_config" in _json.load(open(_cfg_json))
        except Exception:
            prequant = False
    if cfg.quant == "4bit":
        if not prequant:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=_torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        if stream_load:
            load_kwargs["low_cpu_mem_usage"] = True
            load_kwargs["device_map"] = {"": 0}
    elif cfg.quant == "bf16":
        load_kwargs["torch_dtype"] = _torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.llm_name, **load_kwargs)

    if cfg.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.grad_checkpointing)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_targets),
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # multimodal wrappers (e.g. Gemma4Unified) nest the LM width under config.text_config
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        hidden = model.config.text_config.hidden_size
    return model, tok, hidden


class VisualPrefixProjector(nn.Module):
    """d_model anatomical token -> n_prefix LLM-hidden visual-prefix embeddings.

    The MLP hidden width (proj_hidden) is DECOUPLED from the LLM hidden: the output must equal
    the LLM hidden (visual prefix lives in the token-embedding space) but the intermediate need
    not. proj_hidden=512 keeps this ~1.9M params (memorisation risk on 564 patients) with no
    info bottleneck (input is only d_model).
    """

    def __init__(self, d_model: int, hidden: int, n_prefix: int,
                 proj_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_prefix = n_prefix
        self.hidden = hidden
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, hidden * n_prefix),
        )
        # Align the visual prefix with the LLM embedding distribution and remove the shared
        # (flat) component so the prefix actually carries vision. Without it the region prefixes
        # re-flatten (pairwise cosine ~0.95) and the decode collapses into declaring slots absent.
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B,N,d_model) -> (B, N*n_prefix, hidden)."""
        B, N, _ = tokens.shape
        p = self.proj(tokens)                                   # (B,N,hidden*n_prefix)
        p = p.view(B, N * self.n_prefix, self.hidden)
        return self.out_norm(p)                                 # align to LLM embedding space


# --------------------------------------------------------------------------- #
# VLM chat-template assembly.
# The visual prefix (= vision tokens) is placed in the user turn of the chat template together
# with the instruction, and the assistant generates the target. These helpers take the realizer
# object (self) and use self.tokenizer / system_prompt / _embed_text / llm.
# --------------------------------------------------------------------------- #
QWEN_PRE = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
QWEN_MID = "\n{instr}<|im_end|>\n<|im_start|>assistant\n"
QWEN_END = "<|im_end|>"


def chat_text_ids(realizer, text: str) -> torch.Tensor:
    """text -> token ids (special tokens kept, no BOS added)."""
    return realizer.tokenizer(text, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]


def _chat_segments(realizer, instruction: str):
    """(pre_ids, mid_ids, end_ids) around the vision & target positions, derived from the
    model's OWN chat template (model-agnostic: Qwen2.5/Qwen3.5/Gemma4/...). pre = system +
    user-turn header; mid = instruction + assistant header; end = turn terminator.

    Two sentinels (vision-position in the user turn, target-position as the assistant content)
    are rendered by apply_chat_template with the FULL turn, then split out. Rendering the target
    AS assistant content (not add_generation_prompt) is essential: Gemma4 emits a `thought`
    channel after `add_generation_prompt`, but places real assistant content directly after
    `model\n`. A tokenizer that carries no chat template falls back to the Qwen layout."""
    tok = realizer.tokenizer
    if not getattr(tok, "chat_template", None):
        return (chat_text_ids(realizer, QWEN_PRE.format(sys=realizer.system_prompt)),
                chat_text_ids(realizer, QWEN_MID.format(instr=instruction)),
                chat_text_ids(realizer, QWEN_END))
    VS, TS = "\x00VIS\x00", "\x00TGT\x00"
    sys_p = realizer.system_prompt
    user = VS + "\n" + instruction

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            return None

    msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + \
           [{"role": "user", "content": user}, {"role": "assistant", "content": TS}]
    text = render(msgs)
    if text is None or VS not in text or TS not in text:     # system role rejected → fold into user
        text = render([{"role": "user", "content": (sys_p + "\n\n" if sys_p else "") + user},
                       {"role": "assistant", "content": TS}])
    before, rest = text.split(VS, 1)
    mid_text, end_text = rest.split(TS, 1)
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    end = enc(end_text) if end_text.strip() else torch.tensor([tok.eos_token_id], dtype=torch.long)
    return enc(before), enc(mid_text), end


def assemble_chat_sample(realizer, vision: torch.Tensor, instruction: str,
                         target_ids: torch.Tensor, device, dtype):
    """One sample: [sys+user header][vision][instr+assistant header][target][eos]
    -> (embeds (L,hidden), labels (L,)). The context (headers, vision, instruction) is masked to
    -100, so only the target and the EOS are trained on."""
    pre, mid, end = (x.to(device) for x in _chat_segments(realizer, instruction))
    target_ids = target_ids.to(device)
    ep, em, et, ee = (realizer._embed_text(x).to(dtype) for x in (pre, mid, target_ids, end))
    ev = vision.to(dtype)
    embeds = torch.cat([ep, ev, em, et, ee], dim=0)
    n_ctx = ep.shape[0] + ev.shape[0] + em.shape[0]
    labels = torch.cat([
        torch.full((n_ctx,), -100, dtype=torch.long, device=device),
        target_ids, end,                                       # train on target + turn-end EOS
    ], dim=0)
    return embeds, labels


def assemble_chat_prompt(realizer, vision: torch.Tensor, instruction: str, device, dtype):
    """Prompt for generation: [sys+user header][vision][instr+assistant header]
    -> embeds (L,hidden)."""
    pre, mid, _end = (x.to(device) for x in _chat_segments(realizer, instruction))
    ep, em = (realizer._embed_text(x).to(dtype) for x in (pre, mid))
    return torch.cat([ep, vision.to(dtype), em], dim=0)








def pad_and_stack(seqs, labs, hidden, device, dtype):
    """List of variable-length (embeds, labels) -> right-padded batch (inp, attn, labels)."""
    B = len(seqs)
    L = max(s.shape[0] for s in seqs)
    inp = torch.zeros(B, L, hidden, device=device, dtype=dtype)
    attn = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        n = s.shape[0]
        inp[i, :n] = s
        attn[i, :n] = 1
        labels[i, :n] = l
    return inp, attn, labels
