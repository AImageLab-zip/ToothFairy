"""ReportModel — cached perception → encoder → [findings head] → realizer + losses.

Each region prefix generates its own region text (per-structure decode), so the visual prefix
cannot be ignored. Training runs in two stages on a frozen segmentation backbone: prefix alignment
first, then generation. The findings head is optional. The region layout is per-FDI.

Flow:
  encoder(batch) → ctx [B,38,d], presence
  [findings_head] → teeth_logits [B,32,AUX_DIM]  (only when the findings head is enabled)
  slot ↔ region → specs (region tokens, target, soft prob) → realizer.decode_regions → L_region
  loss = L_region + [lambda_gate · L_findings] + [lambda_siglip · L_contrastive]

The findings loss is scaled by `lambda_gate` in all three paths (forward, chunked training, eval),
which keeps them mutually consistent; `lambda_gate` is a hyperparameter that has to be tuned for
each config rather than carried over from another training schedule.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ..config import ReportModelConfig
from ..schema.claims import SOFT_DIM, logit_slices, region_logit_slices, \
    MAXILLA_AXES, MANDIBLE_AXES, NERVE_AXES, N_AXES
from ..encoder import (RegionEncoder, NONTEETH_SLOT, UPCLS_SLOT, LOCLS_SLOT,
                       FDI_TO_SLOT, FDI_TO_ROW)
from ..heads import FindingsHead
from ..generation import Realizer, region_type_id
from ..losses import gate_loss, axis_soft_probs, ContrastiveHead


def build_encoder(cfg: ReportModelConfig, device=None) -> nn.Module:
    """Build the perception aggregator: cached region bags -> (ctx, presence).

    Both segmentation networks are frozen and run once, so the aggregator reads their
    cached, mask-weighted patch bags.
    """
    ec = cfg.encoder
    return RegionEncoder(
        feat_dim=cfg.perception.feat_dim, pe_k=ec.pe_k, d_model=cfg.d_model,
        gate_hidden=ec.gate_hidden, n_heads=ec.n_heads, dropout=ec.dropout,
        n_tokens=cfg.n_tokens, cl_layers=ec.cl_layers, cl_heads=ec.cl_heads,
        cl_mlp_hidden=ec.cl_mlp_hidden, cl_dropout=ec.cl_dropout,
        use_presence_emb=ec.use_presence_emb, use_in_proj=ec.use_in_proj,
        use_tf2=(cfg.perception.backbone == "hybrid"),
        use_tf2_density=cfg.perception.tf2_density,
        drop_arch_cls=getattr(ec, "drop_arch_cls", False))


# --------------------------------------------------------------------------- #
# SigLIP: gather embeddings across ranks
# --------------------------------------------------------------------------- #
def _gather_embeddings(V: torch.Tensor, zt: torch.Tensor, texts: list[str]):
    """Collect (vision, text, source text) embeddings from all DDP ranks so the SigLIP matrix
    grows by world_size.

    SigLIP is a sigmoid pairwise loss, so building an independent matrix per rank costs as many
    contrastive pairs as there are ranks (gradient accumulation does not help either — each step
    still builds its own matrix). The embeddings are (M, dim) and tiny, so gathering them is
    negligible. M differs per rank (the number of non-empty regions varies per volume), so pad to
    the largest M, gather, and then keep only the valid rows.

    **The rank's own slice is put back as the local tensor so the gradient path survives** (the
    OpenCLIP convention): all_gather detaches, and without this no gradient at all would reach the
    vision tower. The text side is frozen and needs no gradient.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return V, zt, texts
    world = dist.get_world_size()
    if world < 2:
        return V, zt, texts
    rank = dist.get_rank()

    m = torch.tensor([V.shape[0]], device=V.device, dtype=torch.long)
    sizes = [torch.zeros_like(m) for _ in range(world)]
    dist.all_gather(sizes, m)
    sizes = [int(x.item()) for x in sizes]
    mmax = max(sizes)

    def _pad(t):
        if t.shape[0] == mmax:
            return t
        pad = t.new_zeros((mmax - t.shape[0], *t.shape[1:]))
        return torch.cat([t, pad], dim=0)

    vb, tb = [torch.zeros_like(_pad(V)) for _ in range(world)], \
             [torch.zeros_like(_pad(zt)) for _ in range(world)]
    dist.all_gather(vb, _pad(V).contiguous())
    dist.all_gather(tb, _pad(zt).contiguous())
    vb[rank] = _pad(V)                                   # restore the gradient path (own slice)

    txt_buf = [None] * world
    dist.all_gather_object(txt_buf, texts)

    Vs = torch.cat([vb[r][:sizes[r]] for r in range(world)], dim=0)
    Ts = torch.cat([tb[r][:sizes[r]] for r in range(world)], dim=0)
    out_txt: list[str] = []
    for r in range(world):
        out_txt.extend((txt_buf[r] or [])[:sizes[r]])
    return Vs, Ts, out_txt


class ReportModel(nn.Module):
    def __init__(self, cfg: ReportModelConfig, device=None):
        super().__init__()
        self.cfg = cfg
        # `encoder.drop_arch_cls` has to be switched on in two places: removal of the cross-label
        # keys (encoder) and **not feeding the arch summary slots at all** (dataset-wide switch).
        # Setting only one of them makes training feed 38 slots while evaluation queries 36.
        from ..data.dataset import set_drop_arch_cls
        set_drop_arch_cls(bool(getattr(cfg.encoder, "drop_arch_cls", False)))
        d_model = cfg.d_model
        self.encoder = build_encoder(cfg, device)
        self.realizer = Realizer(cfg, d_model=d_model, n_soft=SOFT_DIM)
        # findings head (optional): when off, aux_logits=None and there is no findings loss.
        self.findings_head = (FindingsHead(d_model)
                              if cfg.train.findings_head else None)
        self._logit_slices = logit_slices()
        self._nt_slices = {"maxilla": region_logit_slices(MAXILLA_AXES),
                           "mandible": region_logit_slices(MANDIBLE_AXES),
                           "nerve": region_logit_slices(NERVE_AXES)}
        # Per-axis findings class weights, injected by the trainer from the dense label frequencies
        # just before training (a dict is not moved by model.to, so the findings loss moves them
        # onto logits.device). None = unweighted.
        self.class_weights: dict[str, torch.Tensor] | None = None
        # false-EOS penalty: loss weight per region type (normal = empty target vs finding).
        # Injected by the trainer from the training GT frequencies (effective number x lambda).
        # None = unweighted (w=1, i.e. not configured).
        self.region_type_weights: dict[str, float] | None = None
        # Weight of the findings state axis (tooth presence/state = FDI localisation). A buffer, so
        # model.to moves it along.
        aw = torch.ones(N_AXES)
        aw[0] = float(cfg.train.gate_state_weight)
        self.register_buffer("axis_weights", aw)
        # region-text contrastive (optional): the text encoder is injected by the trainer so the
        # heavy load stays out of construction.
        self.use_siglip = bool(cfg.train.use_siglip)
        self.text_encoder = None
        self.contrastive = (ContrastiveHead(self.realizer.hidden, cfg.train.siglip_text_dim,
                                            cfg.train.siglip_proj_dim)
                            if self.use_siglip else None)

    # --- encode -------------------------------------------------------------- #
    def _encode(self, batch):
        """aggregate + cross-label → ctx (B,N,d); with the findings head on, teeth aux logits
        (B,32,AUX_DIM)."""
        ctx, _presence = self.encoder(batch)                       # (B,N,d)
        aux_logits = self.findings_head.teeth_logits(ctx) if self.findings_head is not None else None
        return ctx, aux_logits

    # --- slot → region tokens (+ soft prob) ---------------------------------- #
    def _slot_tokens(self, ctx_i: torch.Tensor, region_id: str,
                     prob_i: torch.Tensor | None, soft: bool):
        """ctx_i (N,d) for a single patient. region_id → (tokens(n,d), prob([SOFT_DIM]|None))."""
        head = region_id.split(":")[0]
        if head in NONTEETH_SLOT:
            return ctx_i[NONTEETH_SLOT[head]].unsqueeze(0), None
        # The arch summary slots (upper/lower) are generation targets as well: one broad dental
        # summary per arch (complete dentition, diffuse periodontitis, ...). They are text-only —
        # no soft probability is injected and they are outside the findings head, because broad
        # dental statements have no categorical axis.
        if region_id == "cls_upper":
            return ctx_i[UPCLS_SLOT].unsqueeze(0), None
        if region_id == "cls_lower":
            return ctx_i[LOCLS_SLOT].unsqueeze(0), None
        # The per-arch teeth tokens are not generation targets: the split GT has no such bucket
        # (its text was regex-derived and mis-routed), and periodontal staging moved into the
        # maxilla/mandible claims.
        if head == "fdi":                            # that FDI slot + (if soft) abnormal probs
            f = int(region_id.split(":")[1])
            slot = FDI_TO_SLOT[f]
            prob = prob_i[FDI_TO_ROW[f]] if (soft and prob_i is not None) else None
            return ctx_i[slot].unsqueeze(0), prob
        raise ValueError(f"unknown region_id {region_id!r}")

    # --- axis probabilities rendered as text --------------------------------- #
    def _prob_text_on(self) -> bool:
        return bool(getattr(self.cfg.train, "prob_text", False)) and self.findings_head is not None

    def _nonteeth_logits_cached(self, ctx: torch.Tensor):
        """Compute the non-teeth head outputs once per forward and reuse them."""
        return self.findings_head.nonteeth_logits(ctx)

    def _slot_prob_text(self, region_id: str, aux_i, nt_i) -> str | None:
        """Axis probabilities for this slot as text. Arch summary slots have no categorical axes,
        so they get None (nothing rendered)."""
        from ..schema.claims import render_probs
        head = region_id.split(":")[0]
        if head == "fdi":
            return render_probs(AXES, aux_i[FDI_TO_ROW[int(region_id.split(":")[1])]])
        if nt_i is None:
            return None
        if region_id == "maxilla":
            return render_probs(MAXILLA_AXES, nt_i["maxilla"])
        if region_id == "mandible":
            return render_probs(MANDIBLE_AXES, nt_i["mandible"])
        if region_id == "nerve_right":
            return render_probs(NERVE_AXES, nt_i["nerve"][0])
        if region_id == "nerve_left":
            return render_probs(NERVE_AXES, nt_i["nerve"][1])
        return None

    def _soft_on(self) -> bool:
        """Soft gate injection = gate_mode soft AND a findings head exists (aux_logits needed)."""
        return (self.cfg.train.gate_mode == "soft") and (self.findings_head is not None)

    def _region_weights(self) -> tuple[float, float]:
        """(w_normal, w_finding) for the false-EOS penalty. Not injected = (1,1) = uniform."""
        rw = self.region_type_weights
        if rw is None:
            return 1.0, 1.0
        return float(rw.get("normal", 1.0)), float(rw.get("finding", 1.0))

    def _build_specs(self, ctx: torch.Tensor, aux_logits, region_targets: list[dict]) -> list[dict]:
        """ctx (B,N,d), aux_logits (B,32,AUX_DIM)|None, region_targets[B]={region_id: target_ids}
        → specs flattened over all (sample, region). An empty dict is skipped, with no loss."""
        soft = self._soft_on()
        ptext = self._prob_text_on()
        nt_all = self._nonteeth_logits_cached(ctx) if ptext else None
        specs: list[dict] = []
        for i, targets in enumerate(region_targets):
            if not targets:
                continue
            prob_i = axis_soft_probs(aux_logits[i], self._logit_slices) if (soft and aux_logits is not None) else None
            nt_i = {k: v[i] for k, v in nt_all.items()} if nt_all is not None else None
            for rid, tgt in targets.items():
                tokens, prob = self._slot_tokens(ctx[i], rid, prob_i, soft)
                spec = {"tokens": tokens, "region_id": rid, "prob": prob, "target_ids": tgt}
                if ptext:
                    spec["prob_text"] = self._slot_prob_text(rid, aux_logits[i], nt_i)
                specs.append(spec)
        return specs

    # --- findings loss ------------------------------------------------------- #
    def _gate_loss(self, ctx: torch.Tensor, aux_logits: torch.Tensor, batch) -> torch.Tensor:
        gamma = float(self.cfg.train.gate_focal_gamma or 0.0)
        nt = batch.get("nonteeth_labels")
        nonteeth_logits = self.findings_head.nonteeth_logits(ctx) if nt is not None else None
        return gate_loss(aux_logits, nonteeth_logits, batch["fdi_labels"], nt,
                         self._logit_slices, self._nt_slices, self.class_weights,
                         self.axis_weights, gamma)

    # --- contrastive loss ---------------------------------------------------- #
    def _siglip_loss(self, ctx: torch.Tensor, aux_logits, batch) -> torch.Tensor:
        """SigLIP between the per-(sample, region) visual prefix and that region's GT text. The
        vision side is the mean-pooled projector output (no LLM forward, so it is cheap in memory);
        the text side is a frozen encoder (no grad)."""
        soft = self._soft_on()
        vz, texts, prob_cache = [], [], {}
        for i, targets in enumerate(batch["region_targets_text"]):
            if not targets:
                continue
            if soft and aux_logits is not None and i not in prob_cache:
                prob_cache[i] = axis_soft_probs(aux_logits[i], self._logit_slices)
            prob_i = prob_cache.get(i) if soft else None
            for rid, txt in targets.items():
                if not txt or not txt.strip():             # empty EOS-gate target (normal region)
                    continue
                tokens, prob = self._slot_tokens(ctx[i], rid, prob_i, soft)
                vision = self.realizer._region_prefix(tokens, region_type_id(rid), prob)  # (P,hidden)
                vz.append(vision.mean(dim=0)); texts.append(txt)
        if len(vz) < 2:                                            # no pair to contrast
            return ctx.new_zeros(())
        V = torch.stack(vz)                                        # (M,hidden)
        with torch.no_grad():
            zt = self.text_encoder.encode(texts).to(V.device).float()   # (M,text_dim) frozen
        if getattr(self.cfg.train, "siglip_gather", False):
            V, zt, texts = _gather_embeddings(V, zt, texts)
        return self.contrastive(V, zt, texts, self.cfg.train.siglip_hard_neg_mask)




    # --- forward ------------------------------------------------------------- #
    def forward(self, batch):
        cs = self.cfg.train.region_chunk_size or 0
        if cs > 0 and not torch.is_grad_enabled():                  # chunked eval → bounded memory
            return self.eval_loss_chunked(batch, cs)
        ctx, aux_logits = self._encode(batch)
        specs = self._build_specs(ctx, aux_logits, batch["region_targets"])
        l_region = self._weighted_region_loss(specs)
        out = {"l_region": l_region}
        loss = l_region
        if self.findings_head is not None and "fdi_labels" in batch:
            l_gate = self._gate_loss(ctx, aux_logits, batch)
            out["l_gate"] = l_gate
            loss = loss + self.cfg.train.lambda_gate * l_gate
        if self.contrastive is not None and self.text_encoder is not None and torch.is_grad_enabled():
            l_sig = self._siglip_loss(ctx, aux_logits, batch)
            out["l_siglip"] = l_sig
            loss = loss + self.cfg.train.lambda_siglip * l_sig
        if aux_logits is not None:
            out["aux_logits"] = aux_logits
        out["loss"] = loss
        return out

    def _weighted_region_loss(self, specs: list[dict]) -> torch.Tensor:
        """Token-weighted CE with a per-region-type weight: finding (non-empty target) vs normal
        (empty target = EOS). With w=(1,1) this is identical to a plain uniform mean. Used by the
        non-chunked path (forward, chunk=0)."""
        w_n, w_f = self._region_weights()
        fin = [s for s in specs if s["target_ids"].numel() > 0]
        nor = [s for s in specs if s["target_ids"].numel() == 0]
        lf, nf = self.realizer.decode_regions(fin, return_count=True)
        ln, nn = self.realizer.decode_regions(nor, return_count=True)
        Nw = w_f * nf + w_n * nn
        if Nw <= 0:
            return lf * 0.0 + ln * 0.0                          # grad-safe 0
        return (w_f * lf * nf + w_n * ln * nn) / Nw

    # --- region sub-batching (bounded per-region memory) --------------------- #
    def _weighted_groups(self, batch, fin, nor, w_f, w_n, ntok):
        """[(metas, weight)] — the findings at `w_f`, the normals at `w_n`.

        Findings and normals are decoded as separate groups so that the false-EOS weight can be
        applied to one of them; splitting them here keeps the region sub-batching bounded.
        """
        return [(fin, w_f), (nor, w_n)]

    def _region_meta(self, batch):
        """[(i, region_id, target_ids)] flattened — same traversal as _build_specs, without ctx."""
        metas = []
        for i, targets in enumerate(batch["region_targets"]):
            if not targets:
                continue
            for rid, tgt in targets.items():
                metas.append((i, rid, tgt))
        return metas

    def _specs_for(self, ctx, aux_logits, metas):
        """Decode specs for the given metas from THIS ctx/aux (recomputed per chunk). Equivalent to
        _build_specs."""
        soft = self._soft_on()
        ptext = self._prob_text_on()
        nt_all = self._nonteeth_logits_cached(ctx) if ptext else None
        prob_cache, specs = {}, []
        for (i, rid, tgt) in metas:
            if soft and aux_logits is not None and i not in prob_cache:
                prob_cache[i] = axis_soft_probs(aux_logits[i], self._logit_slices)
            prob_i = prob_cache.get(i) if soft else None
            tokens, prob = self._slot_tokens(ctx[i], rid, prob_i, soft)
            spec = {"tokens": tokens, "region_id": rid, "prob": prob, "target_ids": tgt}
            if ptext:
                nt_i = {k: v[i] for k, v in nt_all.items()} if nt_all is not None else None
                spec["prob_text"] = self._slot_prob_text(rid, aux_logits[i], nt_i)
            specs.append(spec)
        return specs

    def _end_token_len(self):
        """Length of the chat turn terminator (constant) — needed for the N_total token weight.
        Cached after the first call."""
        if getattr(self, "_end_len_cache", None) is None:
            from ..generation.llm_backend import _chat_segments
            self._end_len_cache = int(_chat_segments(self.realizer, "x")[2].numel())
        return self._end_len_cache

    def train_step_chunked(self, batch, chunk_size, loss_scale):
        """Memory-bounded training step (calls backward internally). Same loss composition as
        forward(): L_region (chunked) + lambda_gate · L_findings + lambda_siglip · L_contrastive.
        SigLIP runs once over the whole batch ctx (cross-patient, no LLM forward); only the
        l_region decode is chunked. Returns a dict of detached losses — **the trainer must not call
        backward on the return value**, because every term has already gone through
        recompute-encode → backward → free."""
        tc = self.cfg.train
        dev = next(self.parameters()).device
        l_gate_d = None
        if self.findings_head is not None and "fdi_labels" in batch:  # findings: encode once, back
            ctx0, aux0 = self._encode(batch)
            l_gate = self._gate_loss(ctx0, aux0, batch)
            gl = tc.lambda_gate * l_gate * loss_scale
            # When the findings head is frozen and used only as a probability source there are no
            # trainable parameters, so the loss carries no grad_fn and backward would crash. The
            # value is still recorded so the logs show what the head is seeing.
            if gl.requires_grad:
                gl.backward()
            l_gate_d = l_gate.detach()
            del ctx0, aux0, l_gate
        l_sig_d = None
        if self.contrastive is not None and self.text_encoder is not None:  # cross-patient (batch)
            ctx_s, aux_s = self._encode(batch)
            l_sig = self._siglip_loss(ctx_s, aux_s, batch)
            (tc.lambda_siglip * l_sig * loss_scale).backward()
            l_sig_d = l_sig.detach()
            del ctx_s, aux_s, l_sig
        metas = self._region_meta(batch)
        if not metas:
            l_region_d = torch.zeros((), device=dev)
        else:
            # false-EOS penalty: make each chunk homogeneous in region type (finding vs normal =
            # empty target), then weight it. Token-weighted normalisation (N_w) keeps the Liger
            # fused CE and is gradient-exact (w=1 reproduces the unweighted path).
            w_n, w_f = self._region_weights()
            end_len = self._end_token_len()
            fin = [m for m in metas if m[2].numel() > 0]
            nor = [m for m in metas if m[2].numel() == 0]
            def _ntok(m):
                return int(m[2].numel()) + end_len
            _groups = self._weighted_groups(batch, fin, nor, w_f, w_n, _ntok)
            N_w = max(sum(wt * sum(_ntok(m) for m in g) for g, wt in _groups), 1e-8)
            acc = 0.0
            for group, wt in _groups:
                for c in range(0, len(group), chunk_size):
                    ctx_c, aux_c = self._encode(batch)              # recompute (fresh graph)
                    specs_c = self._specs_for(ctx_c, aux_c, group[c:c + chunk_size])
                    mean_loss, n_tok = self.realizer.decode_regions(specs_c, return_count=True)
                    ((mean_loss * n_tok * wt / N_w) * loss_scale).backward()
                    acc += float(mean_loss.item()) * n_tok * wt
                    del ctx_c, aux_c, specs_c, mean_loss
            l_region_d = torch.tensor(acc / N_w, device=dev)
        loss_d = (l_region_d + (tc.lambda_gate * l_gate_d if l_gate_d is not None else 0.0)
                  + (tc.lambda_siglip * l_sig_d if l_sig_d is not None else 0.0))
        out = {"l_region": l_region_d, "loss": loss_d}
        if l_gate_d is not None:
            out["l_gate"] = l_gate_d
        if l_sig_d is not None:
            out["l_siglip"] = l_sig_d
        return out

    @torch.no_grad()
    def eval_loss_chunked(self, batch, chunk_size):
        """Memory-bounded eval loss (no backward): encode once, decode in chunks. loss = l_region +
        lambda_gate · l_gate (SigLIP is skipped at eval, so validation is driven by l_region)."""
        ctx, aux = self._encode(batch)
        out = {}
        if aux is not None:
            out["aux_logits"] = aux
        if self.findings_head is not None and "fdi_labels" in batch:
            out["l_gate"] = self._gate_loss(ctx, aux, batch)
        metas = self._region_meta(batch)
        if not metas:
            out["l_region"] = torch.zeros((), device=ctx.device)
        else:
            w_n, w_f = self._region_weights()                      # same weights as training
            end_len = self._end_token_len()
            fin = [m for m in metas if m[2].numel() > 0]
            nor = [m for m in metas if m[2].numel() == 0]
            def _ntok(m):
                return int(m[2].numel()) + end_len
            _groups = self._weighted_groups(batch, fin, nor, w_f, w_n, _ntok)
            N_w = max(sum(wt * sum(_ntok(m) for m in g) for g, wt in _groups), 1e-8)
            # Track the finding and normal terms separately. The weighted sum l_region drops
            # mechanically as soon as more normal regions are fed (eos_include_not_available=True),
            # purely by dilution with easy EOS tokens, and that dilution grows as training proceeds
            # (normal CE → 0), which distorts best-checkpoint and early-stopping decisions.
            # l_region_finding is exported separately so it can be used as the selection criterion.
            acc = 0.0
            parts = {}
            for name, group, wt in (("finding", fin, w_f), ("normal", nor, w_n)):
                sub, ntok = 0.0, 0
                for c in range(0, len(group), chunk_size):
                    ml, nt = self.realizer.decode_regions(
                        self._specs_for(ctx, aux, group[c:c + chunk_size]), return_count=True)
                    sub += float(ml.item()) * nt
                    ntok += nt
                acc += sub * wt
                parts[name] = (sub, ntok)
            out["l_region"] = torch.tensor(acc / N_w, device=ctx.device)
            for name in ("finding", "normal"):
                sub, ntok = parts[name]
                out[f"l_region_{name}"] = torch.tensor(
                    sub / ntok if ntok else 0.0, device=ctx.device)
                out[f"ntok_{name}"] = torch.tensor(float(ntok), device=ctx.device)
        out["loss"] = out["l_region"] + (self.cfg.train.lambda_gate * out["l_gate"]
                                         if "l_gate" in out else 0.0)
        return out

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
