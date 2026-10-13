"""Region-conditioned realizer — independent per-region decode.

**Each region prefix generates the target text of that region on its own** (independent
decode, so the visual prefix cannot be ignored). One patient = several (region, target)
pairs -> all of them are flattened and batched into a single LLM forward (variable length,
right-padded + masked).

Standard VLM layout: the visual prefix (vision tokens) is placed in the user turn of the
chat template together with the per-region instruction, and the assistant produces the

Design:
- frozen base LLM + one shared LoRA + region-type embedding + region instruction + token CE.
- soft gate: the categorical abnormality probabilities of that tooth (SOFT_DIM, i.e. without
  normal/none) are mapped into d_model by prob_proj and added to the FDI vision token
  (graded). hard gate: nothing is injected. (gate_mode flag)
- region_id -> instruction: 'maxilla'|'mandible'|'nerve'|'teeth'|'fdi:<n>'|'teeth_arch:<arch>'.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .llm_backend import (build_llm_backend, VisualPrefixProjector,
                          assemble_chat_sample, assemble_chat_prompt, pad_and_stack)

# region-type id (index into region_type_emb), determined by the region_id prefix.
# cls = arch-level summary token (upper/lower share one type; the arch is told apart by
# the slot and the instruction).
REGION_TYPES = {"maxilla": 0, "mandible": 1, "nerve": 2,
                "teeth": 3, "fdi": 4, "teeth_arch": 5, "cls": 6}
N_REGION_TYPES = len(REGION_TYPES)

# per-region instruction (VLM user turn; wording may be adapted to the domain)
_REGION_INSTR = {
    "maxilla": "Describe the radiographic findings in the maxilla and maxillary sinuses.",
    "mandible": "Describe the radiographic findings in the mandible.",
    "nerve": "Describe the course and relationships of the mandibular canal "
             "(inferior alveolar nerve).",
    "teeth": "Describe the radiographic findings for the dentition.",
}


def region_type_id(region_id: str) -> int:
    head = region_id.split(":")[0]
    if head in ("nerve_right", "nerve_left"):        # both canals share one type emb (side: slot)
        head = "nerve"
    elif head in ("cls_upper", "cls_lower"):         # both arch CLS share one type (arch: slot)
        head = "cls"
    return REGION_TYPES[head]






def _instruction_with_probs(region_id: str, prob_text: str | None) -> str:
    """Structure instruction + (if given) the per-axis probability block. **Training and
    inference go through this same function** — if the block is attached on one side only,
    inference sees a different distribution from the one trained on. If this wording is
    changed, every caller that builds the same block has to change with it.
    """
    instr = region_instruction(region_id)
    if not prob_text:
        return instr
    return (f"{instr}\n\nClassifier probabilities for this structure "
            f"(each axis sums to 1; single value = probability of the positive class):\n"
            f"{prob_text}")


def region_instruction(region_id: str) -> str:
    head, _, rest = region_id.partition(":")
    if head == "fdi":
        return f"Describe the radiographic findings for tooth {rest} (FDI notation)."
    if head in ("cls_upper", "cls_lower"):           # arch-level summary (broad dental findings)
        arch = "maxillary" if head == "cls_upper" else "mandibular"
        return f"Summarize the arch-level dental findings in the {arch} arch."
    if head == "teeth_arch":
        arch = "maxillary" if rest == "upper" else "mandibular"
        return f"Describe the arch-level dental findings in the {arch} arch."
    if head in ("nerve_right", "nerve_left"):        # right and left canals are separate regions
        side = "right" if head == "nerve_right" else "left"
        return (f"Describe the course and relationships of the {side} mandibular canal "
                f"(inferior alveolar nerve).")
    return _REGION_INSTR[head]


class Realizer(nn.Module):
    """anatomical token(s) -> region-type emb (+ soft-gate probability injection) -> vision
    tokens -> chat template (region instruction) -> per-region LLM decode.
    decode_regions (training CE) / generate_region."""

    def __init__(self, cfg, d_model: int, n_soft: int):
        super().__init__()
        rc = cfg.realizer
        self.llm, self.tokenizer, hidden = build_llm_backend(rc)
        self.hidden = hidden
        self.d_model = d_model
        drop = rc.dropout if hasattr(rc, "dropout") else 0.1
        self.projector = VisualPrefixProjector(d_model, hidden, rc.n_prefix_per_token,
                                               proj_hidden=rc.proj_hidden, dropout=drop)
        self.region_type_emb = nn.Embedding(N_REGION_TYPES, d_model)
        # soft gate: per-tooth abnormality probs ([SOFT_DIM]) -> d_model, added to FDI vision
        self.prob_proj = nn.Linear(n_soft, d_model)
        nn.init.zeros_(self.prob_proj.weight)         # no injection at first; learned later
        nn.init.zeros_(self.prob_proj.bias)
        self.system_prompt = getattr(rc, "system_prompt", "")

    def _embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def _region_prefix(self, tokens: torch.Tensor, type_id: int,
                       prob: torch.Tensor | None) -> torch.Tensor:
        """tokens (n,d_model) -> vision tokens (n*n_prefix, hidden). The region-type emb is
        always added; prob (soft gate, when given) is added to the FDI token."""
        dev = tokens.device
        x = tokens + self.region_type_emb(torch.tensor(type_id, device=dev))   # (n,d)
        if prob is not None:
            x = x + self.prob_proj(prob)              # (n,d) + (d,) broadcast
        return self.projector(x.unsqueeze(0))[0]      # (n*n_prefix, hidden)

    def decode_regions(self, specs: list[dict], return_count: bool = False):
        """specs: [{tokens(n,d_model), region_id(str), prob([SOFT_DIM]|None), target_ids(T,)}].
        chat: every region is assembled as a [user(vision+instruction)][assistant(target)] chat and
        right-padded + masked into one batched LLM forward -> token CE (mean). Context is -100, so
        only target+EOS is trained. Empty specs give 0 loss.
        With return_count=True returns (mean_loss, n_loss_tokens) — for token-weighted
        accumulation over region sub-batches."""
        if not specs:
            z = self.region_type_emb.weight.sum() * 0.0       # 0 loss (grad-safe, empty batch)
            return (z, 0) if return_count else z
        device = self.region_type_emb.weight.device
        dtype = self.llm.get_input_embeddings().weight.dtype

        seqs, labs = [], []
        for s in specs:
            vision = self._region_prefix(s["tokens"], region_type_id(s["region_id"]),
                                         s.get("prob"))
            e, l = assemble_chat_sample(
                self, vision,
                _instruction_with_probs(s["region_id"], s.get("prob_text")),
                s["target_ids"], device, dtype)
            seqs.append(e); labs.append(l)
        inp, attn, labels = pad_and_stack(seqs, labs, self.hidden, device, dtype)
        loss = self.llm(inputs_embeds=inp, attention_mask=attn, labels=labels).loss
        if return_count:
            return loss, int((labels != -100).sum().item())
        return loss




    @torch.no_grad()
    def generate_region(self, tokens: torch.Tensor, region_id: str,
                        prob: torch.Tensor | None = None,
                        max_new_tokens: int = 128, prob_text: str | None = None,
                        **gen_kwargs):
        """Single-region inference: chat prompt (vision + region instruction) -> llm.generate."""
        vision = self._region_prefix(tokens, region_type_id(region_id), prob)   # (P,hidden)
        device = vision.device
        dtype = self.llm.get_input_embeddings().weight.dtype
        e = assemble_chat_prompt(self, vision,
                                 _instruction_with_probs(region_id, prob_text),
                                 device, dtype).unsqueeze(0)
        attn = e.new_ones(1, e.shape[1], dtype=torch.long)
        return self.llm.generate(inputs_embeds=e, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, **gen_kwargs)
