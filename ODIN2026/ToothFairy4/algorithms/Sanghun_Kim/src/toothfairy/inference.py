"""End-to-end report generation: one CBCT volume -> one English radiology report.

This is the inference path, driven by `toothfairy.cli.report`. There is no ground truth and no
feature cache here, so the perception features are extracted from the volume **on the fly**:

  1. **dental segmentation network** (frozen), one pass -> non-teeth and arch region features
     (LPS -> RAS flip).
  2. **tf2 (ToothFairy2_Teeth) segmentation network** (frozen), one pass -> per-tooth density
     features (RAS -> flip).
     - hybrid + argmax: whole pooling.
     - hybrid + density: soft pooling plus lo/hi/sat sub-pools.
  3. **hybrid encoding** -> findings head -> per-structure decode. Each anatomical token
     (38 = 4 non-teeth + 2 arch summaries + 32 FDI teeth) is generated separately with the EOS
     gate self-gating (a normal slot produces an empty output) -> bullets concatenated in a
     fixed order.
  4. **Narrative rewrite**: a rewrite-only LoRA is stacked on the same 4-bit base model and
     moves the 36 slots into prose (the report LoRA stays loaded; only the active adapter is
     swapped). If that fails the concatenated bullets are emitted, so a report always comes out.

WARNING: dental and tf2 use different left/right flip conventions (dental flips LPS volumes,
   tf2 flips RAS volumes) -> each backbone gets its own preparation step
   (tf2_tooth_density.prepare_tf2_volume and _dental_ras in this file).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from toothfairy.config import load_config
from toothfairy.models import ReportModel
from toothfairy.perception.backbone import build_predictor
from toothfairy.perception.cache import extract_region_bag
from toothfairy.perception.tf2_tooth_density import (
    build_tf2_predictor, prepare_tf2_volume, extract_tooth_density_bag)
from toothfairy.perception.softclip_norm import install_softclip_ctnorm
from toothfairy.schema.claims import FDI_ALL
from toothfairy.losses import axis_soft_probs

# Fixed anatomical order for concatenation. per_region = 4 non-teeth + 2 arch summaries +
# 32 FDI teeth (38). joint leaves out the arch summaries (36).
_NONTEETH_ORDER = ["maxilla", "mandible", "nerve_right", "nerve_left"]
_CLS_ORDER = ["cls_upper", "cls_lower"]
_JOINT_TOKENS = _NONTEETH_ORDER + [f"fdi:{f}" for f in FDI_ALL]
_PER_REGION_TOKENS = _NONTEETH_ORDER + _CLS_ORDER + [f"fdi:{f}" for f in FDI_ALL]
_LPS = ("L", "P", "S")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHAT_TOK = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
_ROLE = re.compile(r"^\s*(assistant|user|system)\b", re.IGNORECASE)

def _clean_gen(txt: str) -> str:
    txt = _THINK.sub(" ", txt)
    txt = _CHAT_TOK.sub(" ", txt)
    txt = _ROLE.sub("", txt.strip())
    for marker in ("<|im_start|>", "\nuser\n", "\nassistant\n", "\nsystem\n"):
        if marker in txt:
            txt = txt.split(marker)[0]
    return " ".join(txt.replace("\n", " ").split()).strip()


def _region_order_key(rid: str):
    if rid in _NONTEETH_ORDER:
        return (0, _NONTEETH_ORDER.index(rid))
    if rid in _CLS_ORDER:
        return (1, _CLS_ORDER.index(rid))
    if rid.startswith("fdi:"):
        f = int(rid.split(":")[1])
        return (2, FDI_ALL.index(f) if f in FDI_ALL else 99)
    return (3, 0)


def assemble_report(region_texts: dict) -> str:
    """{region_id: generated_text} (non-empty entries only) -> bullets in the fixed order."""
    lines = []
    for rid in sorted(region_texts, key=_region_order_key):
        t = " ".join(region_texts[rid].replace("\n", " ").split()).strip()
        if not t:
            continue
        for seg in t.split(" - "):
            seg = seg.lstrip("-").strip()
            if seg:
                lines.append(f"- {seg}")
    return "\n".join(lines).strip()


# ------------------------------------------------- QLoRA narrative rewriter (slot-labelled)
# WARNING: the three definitions below are copied **verbatim** from
#    `src/toothfairy/pipeline/rewrite_train.py`. If the prompt differs from the training prompt by even
#    one character, the mapping the adapter learned (slots -> report) is not the one applied here.
#    Change this whenever the training module changes — on adapter load,
#    `_assert_rewrite_prompt_matches_training()` compares against the training prompt stored in
#    `rewrite_adapter/train_meta.json` and warns on a mismatch.
_QLORA_SYSTEM = (
    "You are a dental and maxillofacial radiologist. You are given the findings of a\n"
    "dental CBCT, already extracted and grouped by anatomical structure. Write the\n"
    "findings section of the report.\n\n"
    "Use ONLY the findings given. Do not add, invent, infer, or drop any finding.\n"
    "Every tooth number in the input must be recoverable from your report — written\n"
    "explicitly, or covered by an explicit range you write. Do not introduce tooth\n"
    "numbers that are not in the input.\n\n"
    "Output ONLY the report text — no preamble, no bullet points, no headings."
)
# Slot order follows the convention of the reference reports (mandible -> lower teeth ->
# mandibular canal -> maxilla -> upper teeth; 91.2% of the validation reports).
_QLORA_SLOT_ORDER = (["mandible", "cls_lower"]
                     + [f"fdi:{f}" for f in FDI_ALL if f // 10 in (3, 4)]
                     + ["nerve_right", "nerve_left", "maxilla", "cls_upper"]
                     + [f"fdi:{f}" for f in FDI_ALL if f // 10 in (1, 2)])


def _qlora_user_prompt(slots: dict) -> str:
    lines = [f"[{k}] {l}" for k in _QLORA_SLOT_ORDER for l in slots.get(k, []) if l.strip()]
    return "Findings by structure:\n" + "\n".join(lines) + "\n\nWrite the report."


def _assert_rewrite_prompt_matches_training(adapter_dir: Path) -> None:
    """Check that `_QLORA_SYSTEM` matches the system prompt the adapter was trained with.

    `train_meta.json` ships next to the adapter, so the check also works where the training
    module is not available. If the prompt differs by even one character,
    the mapping the adapter learned (slots -> report) is not applied as trained, yet the output
    still looks plausible, so the mismatch cannot be spotted by eye.

    WARNING: a mismatch does **not** stop the run — warning and continuing beats emitting an
    empty report.
    """
    meta = adapter_dir / "train_meta.json"
    try:
        trained = json.loads(meta.read_text(encoding="utf-8")).get("system_prompt")
    except Exception as exc:                                   # older adapters may ship no meta
        print(f"[pipeline] skipping rewrite prompt check ({meta.name}: {exc})", flush=True)
        return
    if trained and trained != _QLORA_SYSTEM:
        print("[pipeline] WARNING: rewrite system prompt differs from the training prompt — the "
              "adapter's learned mapping is not applied as trained. Match src/toothfairy/pipeline/rewrite_train.py.",
              flush=True)


def _slots_from_region_texts(region_texts: dict) -> dict:
    """{rid: "a - b"} -> {rid: ["a", "b"]}, the shape of the training input (a list of lines per
    slot). Must split bullets by the same rule as `assemble_report` so both paths see the same
    content."""
    slots = {}
    for rid, t in region_texts.items():
        segs = [s.lstrip("-").strip()
                for s in " ".join(t.replace("\n", " ").split()).split(" - ")]
        segs = [s for s in segs if s]
        if segs:
            slots[rid] = segs
    return slots


def _install_input_norm(cfg) -> None:
    """Install the input normalisation used during training — without it the report is silently
    produced from different features.

    Training reads a pre-built feature cache, and that cache was written by
    `pipeline/cache_dental.py` / `pipeline/cache_teeth.py` with `install_softclip_ctnorm(s)`
    installed. Inference has no cache and extracts from the volume on the fly, so **the same
    hook has to be installed here** for the inputs to match training.

    Measured on one volume — mean relative difference of the pooled features against the cache:

        | normalisation           | dental             | tf2 whole_pool |
        |-------------------------|--------------------|----------------|
        | standard CTNormalization| 0.021 ~ 0.057      | 0.0147         |
        | soft-clip s=1000        | 0.000008 ~ 0.00003 | 0.000017       |

    So without the hook the features are off by 2~6%; with it they agree to float noise.

    WARNING: some training configs do not record `soft_clip_s` — training only reads the cache, so
    the value was not needed there, and it survives only in the cache's `_meta.json`
    ("soft-clip CTNormalization (s=1000.0)"). The submission config therefore states it
    **explicitly**. The guard below refuses to continue silently when the cache path ends in
    `_sc` (a soft-clip cache) but the value is empty.
    """
    s = float(getattr(cfg.perception, "soft_clip_s", 0.0) or 0.0)
    sc_cache = any(str(getattr(cfg.perception, k, "")).rstrip("/").endswith("_sc")
                   for k in ("cache_dir", "tf2_teeth_cache_dir"))
    if sc_cache and s <= 0:
        raise ValueError(
            "config.perception.cache_dir points at a soft-clip cache (_sc) but soft_clip_s is "
            "empty. Set it in the submission config to the value the training cache was built "
            "with (soft_clip_s in the cache _meta.json; 1000.0 for the released model).")
    if s > 0:
        install_softclip_ctnorm(s)
    print(f"[pipeline] input norm: {'soft-clip s=%g' % s if s > 0 else 'CTNormalization'}",
          flush=True)


class ReportGenerator:
    """Loads the frozen dental (+tf2) segmentation networks and ReportModel once, then
    produces one report per volume."""

    def __init__(self, dental_pred, tf2_pred, cfg, device, model_dir, ckpt_name):
        self.model = None                       # loaded lazily after segmentation (lower peak VRAM)
        self.tok = None
        self.dental_pred = dental_pred
        self.tf2_pred = tf2_pred                # None -> dental-only path
        self.cfg = cfg
        self.device = device
        self._model_dir = Path(model_dir)
        self._ckpt_name = ckpt_name
        self.density = bool(getattr(cfg.perception, "tf2_density", False))
        self._rewrite_adapter_loaded = False
        self._rewrite_no_cls = False

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(cls, model_dir, device: str = "cuda",
                        dental_subdir: str = "dental_segmentator",
                        tf2_subdir: str = "tf2_segmentator",
                        ckpt_name: str = "best.pt") -> "ReportGenerator":
        """Load only the segmentation networks. ReportModel is loaded lazily inside the first
        generate call, once segmentation is done, so that the transient 33-channel patch softmax
        of segmentation and the language model are never resident at the same time (lower peak
        VRAM)."""
        model_dir = Path(model_dir)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")

        cfg = load_config(model_dir / "config.resolved.yaml")
        _install_input_norm(cfg)

        dental_pred = build_predictor(str(model_dir / dental_subdir), fold=0,
                                      checkpoint_name="checkpoint_final.pth", device=str(dev))
        tf2_pred = None
        if cfg.perception.backbone == "hybrid":
            tf2_pred = build_tf2_predictor(str(model_dir / tf2_subdir), fold=5,
                                           checkpoint_name="checkpoint_final.pth", device=str(dev))
        return cls(dental_pred, tf2_pred, cfg, dev, model_dir, ckpt_name)

    def _ensure_model(self):
        """Lazily load ReportModel. Called after segmentation, so peak VRAM is the larger of
        the two stages rather than their sum."""
        if self.model is not None:
            return
        import gc
        gc.collect()                             # free segmentation host arrays (DRAM)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        model = ReportModel(self.cfg).to(self.device).eval()
        ck = torch.load(self._model_dir / self._ckpt_name, map_location=self.device)
        # The final submission used a larger findings schema (31 vs 22 inputs).
        # Preserve its unused probability projector when findings are disabled.
        # An active findings head must still match the checkpoint schema exactly.
        prob_weight = ck["trainable_state"].get("realizer.prob_proj.weight")
        proj = model.realizer.prob_proj
        if prob_weight is not None and prob_weight.shape != proj.weight.shape:
            if model.findings_head is not None or prob_weight.shape[0] != proj.out_features:
                raise RuntimeError("Checkpoint probability projector does not match the findings schema")
            model.realizer.prob_proj = torch.nn.Linear(
                prob_weight.shape[1], proj.out_features, bias=proj.bias is not None,
                device=proj.weight.device, dtype=proj.weight.dtype).eval()
        _, unexpected = model.load_state_dict(ck["trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"{self._ckpt_name} unexpected keys: {list(unexpected)[:5]}")
        try:
            model.realizer.llm.config.use_cache = True
        except Exception:
            pass
        self.model = model
        self.tok = model.realizer.tokenizer

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, volume, max_new_tokens: int = 512, rep_penalty: float = 1.3) -> str:
        # Per-stage timing — printed only when `TF4_TIMING=1` (default behaviour is unchanged).
        # The evaluation instance was an A10G with a **15 minute per-case limit**. To reason about
        # the margin, the volume-proportional part (segmentation) has to be separated from the
        # fixed part (model load, encode, 38-slot decode, rewrite).
        _tm = [("start", time.perf_counter())] if os.environ.get("TF4_TIMING") else None
        def _mark(lab):
            if _tm is not None:
                _tm.append((lab, time.perf_counter()))
        def _dump():
            if _tm is None:
                return
            base = _tm[0][1]
            parts = [f"{l} {t - _tm[i][1]:.1f}s" for i, (l, t) in enumerate(_tm[1:])]
            print(f"[timing] {' · '.join(parts)} · total {_tm[-1][1] - base:.1f}s", flush=True)
        pooling = "soft" if self.density else "argmax"
        with tempfile.TemporaryDirectory(prefix="tf4_") as td:
            tdp = Path(td)
            orig_nii = self._write_orig(volume, tdp)
            dental_nii = self._dental_ras(orig_nii, tdp)          # dental: flip LPS volumes
            region_bag = extract_region_bag(self.dental_pred, dental_nii)
            tf2_bag = None
            if self.tf2_pred is not None:
                tf2_nii, flipped = prepare_tf2_volume(orig_nii, tdp)  # tf2: flip RAS volumes
                tf2_bag = extract_tooth_density_bag(self.tf2_pred, tf2_nii, pooling=pooling)
                if flipped:
                    Path(tf2_nii).unlink(missing_ok=True)

        _mark("segmentation+features")
        batch = self._bag_to_batch(region_bag)
        if tf2_bag is not None:
            batch["teeth_tf2"] = self._tf2_to_batch(tf2_bag)

        # Release the segmentation transient (33-channel patch softmax) before the lazy model load
        # so the two never coexist (lower peak VRAM).
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._ensure_model()
        _mark("llm load")

        batch = _to_device(batch, self.device)
        ctx, aux = self.model._encode(batch)
        _mark("encode")
        aux_i = aux[0] if aux is not None else None

        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        gkw = dict(do_sample=False, repetition_penalty=rep_penalty, num_beams=1,
                   eos_token_id=list({im_end, self.tok.eos_token_id}),
                   pad_token_id=(self.tok.pad_token_id or im_end))

        # per-structure decode: every anatomical token is queried, the EOS gate silences normal
        # and out-of-field slots, and only the non-empty outputs are concatenated.
        soft = self.model._soft_on()
        prob_i = (axis_soft_probs(aux_i, self.model._logit_slices)
                  if (soft and aux_i is not None) else None)
        region_texts = {}
        # If the encoder drops the two arch summary slots, do not query them: asking for a slot
        # the model was never trained on is out-of-distribution input and silently yields
        # spurious text.
        tokens = _PER_REGION_TOKENS
        if bool(getattr(self.cfg.encoder, "drop_arch_cls", False)):
            tokens = [t for t in tokens if t not in _CLS_ORDER]
        # If the model was trained with the per-axis probabilities written into the prompt **as
        # text** (train.prob_text), inference has to do the same: a prompt that differs between
        # training and inference is out-of-distribution input.
        ptext = self.model._prob_text_on()
        nt_i = ({k: v[0] for k, v in
                 self.model.findings_head.nonteeth_logits(ctx[:1]).items()} if ptext else None)
        for rid in tokens:
            tokens, prob = self.model._slot_tokens(ctx[0], rid, prob_i, soft)
            ptxt = self.model._slot_prob_text(rid, aux_i, nt_i) if ptext else None
            gen = self.model.realizer.generate_region(tokens, rid, prob,
                                                      max_new_tokens=200,
                                                      prob_text=ptxt, **gkw)
            txt = _clean_gen(self.tok.decode(gen[0], skip_special_tokens=True))
            if txt:                                                # empty output = EOS gate = skip
                region_texts[rid] = txt
        _mark("38-slot decode")
        # Stack the rewrite-only LoRA on the same 4-bit base model and move the slots into
        # prose. On any failure emit the concatenated bullets — never an empty report.
        out = self._rewrite_qlora(region_texts, gkw)
        _mark("qlora rewrite")
        _dump()
        if out:
            return out
        print("[pipeline] QLoRA rewrite failed -> emitting the slot bullets", flush=True)
        return assemble_report(region_texts)

    # ------------------------------ QLoRA rewrite (slot labels -> narrative report)
    def _rewrite_qlora(self, region_texts: dict, gkw: dict) -> str:
        """Move the slots into a narrative report with the `model/rewrite_adapter` adapter.

        A second adapter is stacked on the **same base model** as the report LoRA; the active
        adapter is switched only for the duration of this generation and then restored, so the
        report generation path is unaffected. The adapter is read once, on the first call.

        WARNING, known limitation: the adapter was trained on **ground-truth slots**, whereas the
        actual input is **model slots** (with omissions and errors). The risk is hallucination —
        plausibly filling the gaps — so it must not be judged on captioning metrics: having
        learned the reference style, it will almost always win BLEU/METEOR regardless.
        """
        adapter_dir = self._model_dir / "rewrite_adapter"
        if not adapter_dir.is_dir():
            print(f"[pipeline] no rewrite adapter at: {adapter_dir}", flush=True)
            return ""
        llm = self.model.realizer.llm
        if not self._rewrite_adapter_loaded:
            _assert_rewrite_prompt_matches_training(adapter_dir)
            llm.load_adapter(str(adapter_dir), adapter_name="rewrite")
            self._rewrite_adapter_loaded = True
            # The adapter declares this itself. `no_cls: true` means it was trained without the
            # two arch summary slots, so the inference input must also be 36 tokens — otherwise
            # it is out-of-distribution input. Older adapters without the flag keep all 38 slots.
            try:
                self._rewrite_no_cls = bool(json.loads(
                    (adapter_dir / "train_meta.json").read_text(encoding="utf-8")).get("no_cls"))
            except Exception:
                self._rewrite_no_cls = False
            print(f"[pipeline] rewrite adapter no_cls={self._rewrite_no_cls}", flush=True)
            print(f"[pipeline] loaded rewrite adapter: {adapter_dir}", flush=True)

        slots = _slots_from_region_texts(region_texts)
        if getattr(self, "_rewrite_no_cls", False):
            slots = {k: v for k, v in slots.items() if k not in ("cls_upper", "cls_lower")}
        if not slots:
            return ""
        msgs = [{"role": "system", "content": _QLORA_SYSTEM},
                {"role": "user", "content": _qlora_user_prompt(slots)}]
        try:
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               return_tensors="pt", enable_thinking=False)
        except TypeError:
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(self.device)

        active = getattr(llm, "active_adapters", None) or ["default"]
        llm.set_adapter("rewrite")
        try:
            # Same generation settings as at training time: greedy, max_new_tokens 640, no penalty.
            out = llm.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               max_new_tokens=640, do_sample=False, num_beams=1,
                               eos_token_id=gkw["eos_token_id"],
                               pad_token_id=gkw["pad_token_id"])
        finally:
            llm.set_adapter(list(active)[0] if not isinstance(active, str) else active)
        return _clean_gen(self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


    # ------------------------------------------------------------- internals
    def _build_specs(self, ctx_i, aux_i, region_ids):
        soft = self.model._soft_on()
        prob_i = (axis_soft_probs(aux_i, self.model._logit_slices)
                  if (soft and aux_i is not None) else None)
        ptext = self.model._prob_text_on()
        nt_i = ({k: v[0] for k, v in
                 self.model.findings_head.nonteeth_logits(ctx_i.unsqueeze(0)).items()}
                if ptext else None)
        specs = []
        for rid in region_ids:
            tokens, prob = self.model._slot_tokens(ctx_i, rid, prob_i, soft)
            spec = {"tokens": tokens, "region_id": rid, "prob": prob}
            if ptext:
                spec["prob_text"] = self.model._slot_prob_text(rid, aux_i, nt_i)
            specs.append(spec)
        return specs

    def _bag_to_batch(self, bag) -> dict:
        def pack(c):
            return _pad_one(bag.pooled[c], bag.mass[c], bag.center[c], self.cfg.perception.feat_dim)
        regions = {c: pack(c) for c in (1, 2, 5)}
        teeth = {"upper": pack(3), "lower": pack(4)}
        return {"regions": regions, "teeth": teeth, "pid": ["_submission"]}

    def _tf2_to_batch(self, bag) -> dict:
        """tf2 ToothDensityBag -> teeth_tf2 batch (B=1, 16 teeth per arch), mirroring
        dataset._load_tf2. In density mode the lo/hi/sat sub-pools (pooled, mass, center) are
        carried as well and injected as the diagonal K/V of the tooth query."""
        pool = bag.whole_pool.astype(np.float32)
        mass = bag.whole_mass.astype(np.float32)
        center = bag.center.astype(np.float32)
        present = bag.present.astype(bool)
        dens = None
        if self.density:
            dens = {"lo": (bag.lo_pool, bag.lo_mass, bag.lo_center),
                    "hi": (bag.hi_pool, bag.hi_mass, bag.hi_center),
                    "sat": (bag.sat_pool, bag.sat_mass, bag.sat_center)}

        def arch(sl):
            d = {"pooled": torch.from_numpy(pool[sl])[None],
                 "mass": torch.from_numpy(mass[sl])[None],
                 "center": torch.from_numpy(center[sl])[None],
                 "present": torch.from_numpy(present[sl])[None]}
            if dens is not None:
                for b, (p, m, c) in dens.items():
                    d[f"{b}_pooled"] = torch.from_numpy(p.astype(np.float32)[sl])[None]
                    d[f"{b}_mass"] = torch.from_numpy(m.astype(np.float32)[sl])[None]
                    d[f"{b}_center"] = torch.from_numpy(c.astype(np.float32)[sl])[None]
            return d
        return {"upper": arch(slice(0, 16)), "lower": arch(slice(16, 32))}

    def _write_orig(self, volume, tmpdir: Path) -> str:
        if isinstance(volume, (str, Path)) and not str(volume).lower().endswith((".nii", ".nii.gz")):
            import SimpleITK
            volume = SimpleITK.ReadImage(str(volume))
        if hasattr(volume, "GetSize"):                            # SimpleITK image
            import SimpleITK
            src = tmpdir / "volume_in.nii.gz"
            SimpleITK.WriteImage(volume, str(src))
            return str(src)
        return str(volume)

    def _dental_ras(self, orig_nii: str, tmpdir: Path) -> str:
        img = nib.load(orig_nii)
        ax = nib.aff2axcodes(img.affine)
        if tuple(ax) != _LPS:
            return orig_nii
        lr = next(i for i, c in enumerate(ax) if c in ("L", "R"))
        data = np.flip(np.asanyarray(img.dataobj), axis=lr).copy()
        out = tmpdir / "volume_dental_ras.nii.gz"
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out))
        return str(out)


# --------------------------------------------------------------------- utils
def _pad_one(pooled: np.ndarray, mass: np.ndarray, center: np.ndarray, feat_dim: int) -> dict:
    n = int(pooled.shape[0])
    P = max(n, 1)
    out_pooled = np.zeros((1, P, feat_dim), np.float32)
    out_mass = np.zeros((1, P), np.float32)
    out_center = np.zeros((1, P, 3), np.float32)
    out_mask = np.zeros((1, P), bool)
    if n:
        out_pooled[0, :n] = pooled
        out_mass[0, :n] = mass
        out_center[0, :n] = center
        out_mask[0, :n] = True
    return {"pooled": torch.from_numpy(out_pooled), "mass": torch.from_numpy(out_mass),
            "center": torch.from_numpy(out_center), "mask": torch.from_numpy(out_mask)}


def _to_device(batch: dict, device):
    out = {}
    for k, v in batch.items():
        if k in ("regions", "teeth"):
            out[k] = {c: {kk: t.to(device) for kk, t in d.items()} for c, d in v.items()}
        elif k == "teeth_tf2":
            out[k] = {arch: {kk: t.to(device) for kk, t in d.items()} for arch, d in v.items()}
        else:
            out[k] = v
    return out
