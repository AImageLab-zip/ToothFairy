# Segmentation

Stage 1 of the challenge pipeline: CBCT volume -> labelled mask -> structured
facts. Report generation (the rest of this repo) is the second half, and
consumes the two files this stage hands it.

- `segmentation.py` / `task1_inference.py` — the U-Mamba2/nnU-Net inference
  wrapper used for the Grand Challenge submission: the L-first left/right
  orientation fix and patch-bounded inference memory are `segmentation.py`'s;
  `task1_inference.py` is adapted from U-Mamba2 (CC BY-NC 4.0 — see
  `THIRD_PARTY.md`). Both expect `code/container/config.py` on the Python
  path for `NNUNET_MODEL_FOLDER` / `NNUNET_CHECKPOINT` / `NNUNET_FOLDS` /
  `USE_MIRRORING`.
- `extract_facts.py` (v4) — mask -> `facts.json`, and/or corrects an existing
  `facts.json` against the mask. Extraction and audit are the same code now;
  see its module docstring for the two modes.
- `audit_facts.py` — the standalone audit `extract_facts.py` v4 folded in.
  Kept as `infer.py`'s fallback for facts arriving unaudited from elsewhere.
- `facts.py`, `orientation.py` — the label-scheme seam and reorientation
  helpers the above share.

You can still swap in your own segmenter or hand in a mask/`facts.json` from
elsewhere — nothing downstream cares how the two files below were produced,
only that they match this interface.

## The interface

Everything downstream needs exactly two files per case:

```
mask.nii.gz     FDI-labelled segmentation: per-tooth labels plus jaw
                structures (mandible, maxilla, mandibular canal, sinuses).

facts.json      {"structured": {"teeth_present": [11, 12, 13, ...],
                                "teeth_absent":  [16, 25, ...]}}
```

`teeth_present` is the load-bearing field. The renderers draw and label **only**
the teeth it names, and `create_tooth_detail.py` builds one composite crop per
present tooth. A segmentation false positive that reaches this list becomes a
tooth in the report; one that never reaches it is never asked about. That is why
the field is an input to *rendering* and not merely metadata.

Everything else in `facts.json` is optional as input to the pipeline. Two fields
are not optional as input to the **generators**, though:

| field | who needs it |
|---|---|
| `fov.maxilla: "excluded"` | the maxilla arch gate — see `../postprocess/source_rules.py` |
| `bridge_arches` | the bridge source rule, same file |

`extract_facts.py` writes both by default (`--no-audit` restores the older,
unaudited v3 output). If you supply facts from elsewhere, either write those
two fields yourself or accept the defaults postprocess falls back to.

## Where the mask is actually read

Nothing here reads it — these do:

- `../preprocess/create_panoramic.py` — curved reconstruction, and the tooth
  outlines, filtered by `teeth_present`
- `../preprocess/create_3d_renders.py` — surface renders of jaws, teeth, canal
  and sinuses
- `../preprocess/create_tooth_detail.py` — the per-tooth composite crops
- `../preprocess/create_sinus_detail.py` — the maxillary sinus views
