#!/usr/bin/env python3
"""
run_vqa_inference.py

VLM inference matching the qa_pairs.jsonl format produced by the updated
build_vqa_pairs.py (schema v6.1): each case record has

  record["global"][call_key]           = {"images": {...}, "captions": {...}, "questions": {...}}
  record["dental_elements"]["tooth_N"] = {"images": {...}, "captions": {...}, "questions": {...}}

There's no more single "global" + per-tooth "detail" call in the OLD sense
-- "global" now holds SEVERAL calls (grouped by which images they need: one
per 3D view, one sinus call per side, and the panoramic split into a
mandible half and a maxilla half; mandible and maxilla facts are pooled
together here and collapse into one call whenever they happen to need the
same image, e.g. the mandible's condyle/canal facts and the maxilla's
upper-wisdom-tooth fact all only need "3d_left"), and "dental_elements"
holds one call per tooth. Every call's images come with a caption (unless
build_vqa_pairs.py was run with --no-captions), which gets woven into the
prompt right before each image -- MedThinkVQA-style "per-image caption,
then the image".

All calls in "global" are merged into one flat prediction dict
(prediction["global"]) since their facts are logically part of the same
whole-case picture -- and because the panoramic call's
dental_arch_findings_{arch} is what WOULD gate which teeth get a detail call
(see model_absent_teeth), which needs the merged view to be readable. That
gate is off (TRUST_MODEL_ABSENCE = False), and its being off is what makes
every call in a case independent of every other, so they are all issued
concurrently -- see run_calls_concurrently().

Reads qa_pairs.jsonl (one record per case), calls vLLM via the
OpenAI-compatible chat completions API, writes one {case_id}_pred.json
per case.

Usage:
  python code/pipeline/vqa/run_vqa_inference.py \
      --vqa-jsonl test_5/outputs/qa_pairs.jsonl \
      --out-dir test_5/outputs/predictions \
      --vllm-url http://localhost:8000/v1 \
      --model qwen3.5-vl

  # Options:
  LIMIT=2 python code/pipeline/vqa/run_vqa_inference.py ...     (smoke test)
  python code/pipeline/vqa/run_vqa_inference.py ... --dry-run   (no vLLM calls)
  python code/pipeline/vqa/run_vqa_inference.py ... --resume    (skip inference for cases with an
                                                    existing _pred.json; still writes a
                                                    missing summary/report from it)
  python code/pipeline/vqa/run_vqa_inference.py ... --case-ids A004 A059
  python code/pipeline/vqa/run_vqa_inference.py ... --max-concurrency 16
                                                   (calls in flight per case; 1 =
                                                    the old sequential behaviour.
                                                    Env: MAX_CONCURRENCY)
  python code/pipeline/vqa/run_vqa_inference.py ... --summaries-out-dir <dir>
                                                   (also persist the postprocessed
                                                    {case_id}_summary.json fed to the
                                                    report LLM -- same output as
                                                    postprocess_pred.py's own CLI)
"""

import os, sys, json, base64, argparse, re, time, traceback
import io, threading, contextlib, functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

from postprocess_pred import postprocess_prediction  # noqa: E402

# The LLM report writer is not here, and not reachable from here. This file
# writes predictions and (optionally) postprocessed summaries; reports are
# written by synthesize_report.py's deterministic templates, in the pipeline
# and in the competition container alike. --reports-out-dir used to hook the
# LLM writer of the baseline arms in through a lazy import, and outlived it:
# the pipeline stopped passing the flag, the module stopped shipping, and the
# flag stayed in the CLI where it failed with ModuleNotFoundError for anyone
# who found it. The safety argument for the template -- it cannot invent a
# clinical error the VQA did not already make -- is worth more than a flag
# nothing passed.

# ── Image encoding ───────────────────────────────────────────────────────

def _b64(path: str) -> str:
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def _img_block(path: str) -> dict:
    """Return an OpenAI-compatible image_url content block."""
    ext = Path(path).suffix.lower()
    mt = {"png": "image/png", "jpg": "image/jpeg",
          "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mt};base64,{_b64(path)}"},
    }


def build_captioned_image_blocks(images: Dict[str, str], captions: Dict[str, str]) -> List[dict]:
    """
    Build the interleaved [caption text, image] blocks for one call's image
    set -- MedThinkVQA style: each image is immediately preceded by a short
    text block naming it and giving its caption (orientation markers, slice
    position, etc.), so the model knows what it's looking at before it sees
    the pixels, rather than being handed a bare stack of images.

    Only images whose file actually exists on disk are included, silently
    skipping the rest -- a partial image set (e.g. build_vqa_pairs.py kept a
    call alive with only 2 of its 3 images present) still gets sent with
    whatever's available.
    """
    blocks = []
    i = 0
    for key, path in images.items():
        if not path or not Path(path).exists():
            continue
        i += 1
        caption = captions.get(key)
        label = f"Image {i} ({key})" + (f": {caption}" if caption else ":")
        blocks.append({"type": "text", "text": label})
        blocks.append(_img_block(path))
    return blocks


# ── vLLM call with retry ─────────────────────────────────────────────────

# Whether the declared json_schema is enforced by the sampler or left as
# prose in the prompt. On by default; STRUCTURED_OUTPUTS=0 restores the
# prompt-only behaviour every pre-2026-08 run actually had (see
# call_vllm_messages), which is the only honest way to reproduce one.
STRUCTURED_OUTPUTS = os.environ.get("STRUCTURED_OUTPUTS", "1").lower() \
    not in ("0", "false", "no", "")

# Decoding temperature. Empty (the default) sends none and takes the server's,
# which is what every run in this repo before 2026-08-14 did -- and which makes
# an arm comparison unreadable, because two runs of the SAME model differ.
#
# THIS WAS NOT A THEORETICAL PROBLEM. arm 1 was scored against a baseline drawn
# a day earlier at unknown sampling, and the difference could not be attributed
# to the adapter at all. TEMPERATURE=0 makes a scoring run replayable, which is
# the property a baseline has to have.
#
# It is NOT set to 0 by default, deliberately. Every banked number in this
# project -- the 0.572/0.552 AWQ-vs-bf16 figures, §5's 27.6% overlay zero point,
# the aksssr_v7_validate baseline -- was produced at the server default, and an
# arm scored at 0 against a baseline scored at the default swaps one confound
# for another. Adopting 0 means re-running the baseline at 0 too; the payload
# replay in scripts/aksssr_pipeline.sh makes that ~17 min rather than a re-render.
TEMPERATURE = os.environ.get("TEMPERATURE", "").strip()

# A truncated object is RESENT (see infer_call), and at temperature 0 a resend
# is a byte-identical replay -- the retry stops being a retry. So a retry after
# a parse failure draws at this instead, keeping the mechanism alive without
# giving up a reproducible first draw.
RETRY_TEMPERATURE = float(os.environ.get("RETRY_TEMPERATURE", "0.7"))


def _sampling(attempt: int) -> dict:
    """Temperature for this attempt, or {} to take the server's default."""
    if TEMPERATURE == "":
        return {}
    t = float(TEMPERATURE)
    if attempt > 0 and t == 0.0:
        return {"temperature": RETRY_TEMPERATURE}
    return {"temperature": t}


def call_vllm_messages(client, model: str, messages: list,
                       max_tokens: int = 4096, retries: int = 3,
                       json_schema: Optional[dict] = None) -> Optional[str]:
    """
    Call vLLM (OpenAI-compatible chat completions) with retry logic, taking a
    FULL messages list.

    Split out of call_vllm() so a caller can express turn structures the
    two-turn [system, user] shape cannot -- a multi-turn layout
    [system, ex_user, ex_assistant, ..., query_user], for instance. The
    pipeline itself still goes through call_vllm() below and is unaffected:
    same retry, same structured decoding, same reasoning_content fallback,
    same Optional[str] return.

    json_schema (from qa_pairs.jsonl, built by build_vqa_pairs.py) CONSTRAINS
    the sampler to the declared shape. Without it the schema is prompt-only,
    and the model sometimes answers an object-typed field with a string
    describing the object ("object {eruption_state: 'fully_erupted'}") --
    valid JSON, wrong shape, which then breaks postprocessing downstream. It
    constrains only the generated turn, so any answers already present in the
    prompt are untouched.

    IT IS SENT AS `response_format`, NOT `guided_json`, AND THAT IS A FIX.
    vLLM's request models are pydantic `extra="allow"`, so an unknown key in
    extra_body is accepted, logged nowhere the pipeline reads, and dropped.
    `guided_json` was deprecated in vLLM 0.11 (forwarded to
    `structured_outputs`) and REMOVED by 0.22 -- which is the version inside
    extraction.sqsh. So every run in this repo that thought it was decoding
    under a grammar was in fact prompt-only, silently, and normalize_pred.py
    has been repairing shape violations that the sampler should never have
    been able to emit. `response_format` is the OpenAI-standard spelling,
    is a first-class client parameter rather than an extra_body key, and is
    understood by every vLLM in play (0.11 -> 0.22, and 0.19.0 in the
    submission container); the client validates its shape locally and the
    server 400s on a schema its grammar backend cannot compile, so this
    failure mode is loud in both directions.

    STRUCTURED_OUTPUTS=0 turns it off, for A/B-ing the constraint itself
    against the arms' predecessors -- not for routine use.
    """
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    fmt = {}
    if json_schema and STRUCTURED_OUTPUTS:
        # `schema` is the OpenAI key; vLLM reads it through an alias onto its
        # own `json_schema` field. Sending it under any other name is the
        # same silent no-op guided_json just was.
        fmt["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "vqa_call", "schema": json_schema},
        }

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                extra_body=extra_body,
                **_sampling(attempt),
                **fmt,
            )
            msg = resp.choices[0].message
            content = msg.content
            reasoning = getattr(msg, "reasoning_content", None)

            if content:
                return content
            if reasoning:
                print(f"    [DEBUG] content empty, using reasoning_content instead",
                      file=sys.stderr)
                return reasoning
            print(f"    [DEBUG] both content and reasoning_content empty",
                  file=sys.stderr)
            return None
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    [retry {attempt+1}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def call_vllm(client, model: str, system: str, user_blocks: list,
              max_tokens: int = 4096, retries: int = 3,
              json_schema: Optional[dict] = None) -> Optional[str]:
    """One system turn + one user turn -- the shape every pipeline call uses."""
    return call_vllm_messages(
        client, model,
        [{"role": "system", "content": system},
         {"role": "user", "content": user_blocks}],
        max_tokens=max_tokens, retries=retries, json_schema=json_schema)


class IncompleteObject(ValueError):
    """The response parsed, but to fewer facts than the model was writing.

    Deliberately NOT a json.JSONDecodeError -- parse_json catches that one to
    run its repairs, and this is the case where a repair would be the bug.
    """


def _unclosed_brackets(frag: str) -> str:
    """The closing sequence `frag` still needs, or "" if it needs none.

    Scans outside string literals, honouring escapes, so a brace inside
    visual_evidence prose does not count.
    """
    stack, in_str, esc = [], False, False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ("{" if ch == "}" else "["):
                stack.pop()
    if in_str:
        return ""        # cut mid-string: genuinely truncated, do not repair
    return "".join("}" if c == "{" else "]" for c in reversed(stack))


def parse_json(text: Optional[str]) -> dict:
    """Extract JSON from model output, tolerating markdown fences / preamble.

    Two repairs, both narrow, both logged, and one refusal: a trailing tail
    that starts with "," raises IncompleteObject rather than returning the
    short object (see below). Other trailing text after a complete
    object is discarded rather than fed to the parser. And it closes an
    object the model finished but did not terminate. Putting example
    answers in the prompt made this common: every example's assistant turn ends
    with "}", and after the last fact's own "}" the model reads the pattern as
    finished and stops one character short. Measured on job 552631, on a
    prompting arm that has since been dropped -- 4 of 16 calls with examples, 0
    of 16 without, and all four held every field the call asked for and parsed
    on appending a single "}". The repair outlived the arm because the failure
    is the model's, not the arm's.

    The repair is narrow and self-validating: only the closers the bracket
    stack is actually missing are appended, nothing is added inside a string
    (a response cut mid-string is truncation, and is left to fail), and the
    result must parse or the original error stands. It logs, because a
    silently repaired answer is indistinguishable from a clean one and this
    is the model telling us something about the prompt.
    """
    if not text:
        raise ValueError("Empty response")
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")

    # Take the FIRST complete object and ignore whatever follows it, rather
    # than regexing to the last "}" in the response. Job 552639 produced a
    # 19,988-character answer whose first 1,890 characters were the complete,
    # correct object and whose remaining 18,000 were the model looping on
    # "Wait, 43 is Patient's Right. 38 is Patient's Left. Yes. OK." A greedy
    # match swallows that tail; raw_decode stops at the object's own end.
    try:
        obj, end = json.JSONDecoder().raw_decode(cleaned, start)
        trailing = cleaned[end:].strip()
        if trailing:
            # A tail that opens with a comma is not commentary -- it is the
            # REST OF THIS OBJECT. The model closed the brace after the first
            # fact and kept going: `{"tooth_46_morphology": {...}}, ...` where
            # five facts were asked. raw_decode happily returns fact 1, the
            # call is recorded as answered, and four facts vanish into a
            # [WARN] nobody diffs. That is the "first complete object" defect,
            # and it is a truncation, not a parse success -- so refuse it and
            # let infer_call resend. Anything else after a complete object is
            # the model talking to itself and is still safe to discard.
            if trailing.startswith(","):
                raise IncompleteObject(
                    f"object closed after {len(obj)} field(s) with "
                    f"{len(trailing)} chars still to come: {trailing[:90]!r}")
            print(f"[WARN] parse_json: discarded {len(trailing)} chars of "
                  f"trailing text after the JSON object: {trailing[:90]!r}",
                  file=sys.stderr)
        return obj
    except json.JSONDecodeError:
        pass

    frag = cleaned[start:].rstrip()
    closing = _unclosed_brackets(frag)
    if closing:
        try:
            out = json.loads(frag + closing)
            print(f"[WARN] parse_json: response ended unterminated, closed it "
                  f"with {closing!r} ({len(frag)} chars)", file=sys.stderr)
            return out
        except json.JSONDecodeError:
            pass
    return json.loads(frag)      # raises with the original, honest error


# ── Prompts ───────────────────────────────────────────────────────────────

# Shared "how to read a tooth" block, injected into every call that judges
# per-tooth findings (the per-tooth detail calls and the panoramic tooth
# survey; NOT the 3D/sinus calls, which ask nothing about tooth structure).
#
# It exists because the two commonest VLM errors on this task are the same
# mistake -- reading a gray level without reading WHERE it sits:
#   • enamel (bright by nature) read as a crown/restoration
#   • the root canal (dark by nature) read as caries
# Neither is settled by brightness ALONE. The block therefore states the
# normal structure first and then gives the discriminator for each pair, and
# every discriminator reads brightness together with shape and position --
# brightness is a real cue (metal streaks and is far brighter than enamel),
# it is just never a threshold on its own, since tooth-coloured restoratives
# are made to match enamel's radiopacity and land in the same gray range.
# "Brighter than enamel -> it is a restoration" was the earlier wording and it
# silently missed every composite and ceramic restoration; the opposite
# overcorrection -- ignore brightness, read shape only -- throws away the one
# cue that does settle metal. Keep it short: ~34 calls per case.
HOW_TO_READ_A_TOOTH = """\
HOW TO READ A TOOTH
────────────────────
Normal structure -- none of this is a finding on its own:
- ENAMEL: the bright cap over the CROWN only. It is the densest tissue in the
  body, so a healthy crown is ALREADY the brightest part of the tooth.
- DENTIN: the mid-gray bulk beneath the enamel, continuing into the root.
- PULP CHAMBER / ROOT CANAL: the DARK line down the CENTRE of the tooth, from
  the crown to each root apex.

A BRIGHT area can be enamel or restorative material, and a DARK one can be the
normal root canal or a defect. Tell them apart by brightness, SHAPE and
POSITION together:

Enamel or restoration?
- Enamel: caps the crown only, thins away smoothly at the neck, follows the
  cusp shape, and the dark canal still shows through beneath it.
- Restoration: geometry a tooth does not have -- a uniform shell or block
  with a SHARP, straight margin against tooth substance, a flat floor, or a
  cavity outline; it hides/replaces the structure beneath it, and metal
  streaks. A crown covers the whole clinical crown down to a defined neck
  margin.
- Bright but tooth-shaped, tapering at the neck, canal visible under it
  -> ENAMEL, report no restoration.

Canal or defect?
- Canal (normal): dark, thin, CENTRAL, smooth-walled, continuous from the
  pulp chamber to the apex, and similar in each root of the same tooth.
- A BRIGHT line following that same central path -> root canal filling (RCT).
- Defect: dark that is WIDER than the canal, ragged or asymmetric between
  roots of one tooth, breaking the tooth's outline, or reaching the OUTER
  surface -- occlusal, proximal contact point, cervical -- eating inward.
- Thin, central, symmetric = anatomy. Wide, ragged, off-centre or open to the
  surface = defect (caries, resorption, fracture).

Also:
- Crown above bone -> fully erupted; partly below -> partially erupted/impacted
- Linear shadow along the root -> fracture
- Dark halo at the apex -> periapical lesion
"""

HOW_TO_READ_THE_PANORAMIC = """\
HOW TO READ THE PANORAMIC
──────────────────────────
SCALE FIRST. This is the whole arch flattened into one image, so a single tooth
is a few dozen pixels wide -- roughly a fortieth of the detail the per-tooth
crops carry. Report only what is unmistakable AT THIS SCALE. A feature you have
to zoom into, or reason your way to, is not a finding here.

WHAT THIS IMAGE CAN SETTLE
- Whether a tooth is there at all, and roughly where the bone crest sits.
- Material INSIDE the canal: a thin bright line running the length of the canal
  toward the apex, or a THICK, SHORT bright rod in its upper part that stops
  well above the apex (see _definitions.root_canal_fill).
- Material ON the crown: a sharply-bordered opacity clearly brighter than the
  enamel of its neighbours, with a straight or geometric edge no tooth has.
- Gross loss of tooth substance: a dark notch that breaks the crown's outline
  and eats inward from the outer surface, or a root left with no crown at all.
- A third molar's tipping and how much of its crown is still under bone.

WHAT IT CANNOT SETTLE -- and the enum does not ask you to
- WHICH restoration it is. A full crown, a large filling and a bridge abutment
  retainer are one bright capped tooth at this scale, so they share the single
  value "restoration"; the per-tooth crops separate them.
- A post from a root canal filling. Both are material in the canal and both
  read "root_canal_treatment" here, for the same reason.
- Caries from a broken-down root remnant. Both are lost tooth substance and
  both read "defect".
- Early or small caries, hairline root fractures, fine periapical detail.
- Anything on a tooth the reformat has smeared or doubled (see below).

PANORAMIC ARTEFACTS THAT IMITATE FINDINGS -- none of these is a finding
- CERVICAL BURNOUT: a dark wedge at the neck, between the crown's enamel and
  the root, caused by the beam passing through less tissue there. It appears on
  MANY teeth at once and sits at the same level on each. Cervical caries is a
  break in ONE tooth's outline; a dark band shared by a run of neighbouring
  teeth is burnout.
- GHOST SHADOWS: a blurred, magnified copy of a dense structure from the
  OPPOSITE side, projected across the image and usually higher than the real
  one. Common over the rami and the midline.
- SUPERIMPOSITION: the spine in the midline, the hard palate over the upper
  roots, and the opposite arch over the rami, all darkening or brightening
  areas that belong to no tooth.
- REFORMAT SMEAR: the curve is fitted to the arch, so a tooth outside it is
  stretched, thinned or duplicated. Sharp horizontal density steps mark where
  the curve left the anatomy.
"""

# Global call keys that judge per-tooth findings and therefore get a reading
# block -- the panoramic tooth survey (dental_arch_findings_{arch}, implants,
# bridges). The 3D and sinus calls do not, nor does pano_others_{arch}
# (primary teeth, bone quality, periodontal resorption): none of them turns
# on telling enamel from a crown or a canal from caries.
#
# THESE GET HOW_TO_READ_THE_PANORAMIC, NOT HOW_TO_READ_A_TOOTH. They used to
# get the latter, which is written for the composite crops and asks for detail
# this image does not contain: "the dark canal still shows through beneath it",
# "in axial views a crown fills nearly the whole cross-section", the red target
# outline. Measured on the panoramic, one mandibular tooth gets ~48 of the
# image's 768 visual tokens, against 1764 for the single tooth in a composite
# -- so that block was spending 533 tokens arguing for a level of scrutiny the
# pixels cannot support, and inviting exactly the over-claiming the survey
# shows (panoramic-source precision 0.16-0.25). The panoramic block below
# instead states what this image can and cannot settle, and names the four
# artefacts that imitate findings on it.
#
# The "panoramic|{arch}" spellings are the pre-v6.1 call keys, kept so an
# older qa_pairs.jsonl still gets the block instead of silently missing it.
TOOTH_READING_CALLS = {
    "pano_tooth_findings_mandible", "pano_tooth_findings_maxilla",
    "panoramic|mandible", "panoramic|maxilla",
}

# ── What the model is actually being shown ────────────────────────────────
#
# One intro per CALL, built from that call's own images_needed names, because
# the four image kinds are not the same kind of picture and the old single
# wording was wrong for three of them. It claimed:
#
#   "You are given N CBCT image(s) ... Read each caption before its image, and
#    use all the images together to answer."
#
#   - "CBCT image" is right only for the sinus grids and the tooth composites,
#     which really are grayscale slices out of the volume. A 3D render is a
#     SURFACE MODEL of the segmentation masks -- no grayscale, no slice, and
#     nothing in it to window or measure density on. The panoramic is a curved
#     reformat of the volume, closer to a rendering than to a slice.
#   - "use all the images together" was addressed to a model holding exactly
#     ONE image: every fact in the current schema needs one image, so that
#     sentence asked it to combine a set of size 1 for every global call, and
#     "up to 1 CBCT images ... axial, coronal and sagittal slices" told the
#     per-tooth call it had three when it has one composite carrying nine
#     panels.
#
# FDI TAGS ARE LIKEWISE NOT UNIVERSAL. The panoramic prints an FDI tag per
# tooth in the tooth's own colour, and so do the sinus details; the 3D renders
# print no tags at all (their colour key is per STRUCTURE -- mandible green,
# canal yellow), and the tooth composite has one target outlined in red, the
# mandibular canal filled with a translucent wash of that same yellow on lower
# teeth, and its FDI in the title bar. Telling a 3D call to "match the tag to its tooth by
# COLOUR" names a marker that is not in the picture, which is the same failure
# mode as a caption describing an outline that is not drawn.
IMAGE_KIND_INTROS = {
    "3d": "a 3D surface reconstruction built from this case's CBCT "
          "segmentation masks -- a model of the segmented structures, not a "
          "grayscale slice",
    "panoramic": "a panoramic view reconstructed from this case's CBCT volume "
                 "(a curved reformat, flattening the whole arch into one image)",
    "sinus": "a grid of grayscale CBCT slices through one maxillary sinus",
    "tooth": "a 3x3 grid of grayscale CBCT slices through this tooth, one "
             "plane per row (axial, coronal, sagittal)",
}

# The image-quality note. EVERY call gets one -- none of these four pictures is
# a photograph of the patient: each is generated, and each leans on the same
# automatic segmentation somewhere. What differs is HOW MUCH of the picture the
# segmentation is responsible for, so the opening clause is per kind and the
# instruction that follows is shared.
#
# The failure this is aimed at is a model that explains an impossible shape
# instead of declining to read it. The system prompt already forbids inventing
# anatomy that is absent; this covers the other half -- anatomy that is present
# but wrong, or too degraded to call -- and gives it a specific way to say so
# ("unable to evaluate" + null) rather than leaving the model to choose between
# a guess and a malformed answer.
IMAGE_KIND_CAVEAT_LEADS = {
    "3d": "this render IS the segmentation, drawn as a surface, with no CBCT "
          "grayscale underneath it, so it shows exactly what was segmented "
          "and nothing else",
    "panoramic": "the grayscale is a curved reformat of the CBCT volume, which "
                 "can smear or duplicate structures the curve does not follow, "
                 "and every tooth outline and FDI tag drawn on it comes from "
                 "the segmentation",
    "sinus": "the grayscale is straight from the CBCT volume, but the green "
             "cross-marker that locates the sinus comes from the segmentation",
    "tooth": "the grayscale is straight from the CBCT volume, but the red "
             "outline that names the target tooth comes from the segmentation",
}

IMAGE_QUALITY_INSTRUCTION = (
    "and an automatic segmentation can be wrong. If a structure is hard to "
    "recognise, or what you see does not make anatomical sense, do not "
    "rationalise it into something plausible and do not fall back on what a "
    "normal case would look like: say \"unable to evaluate\" in "
    "visual_evidence and answer null for the other fields of that finding. "
    "Never assume anything that is not in the image."
)


def build_image_caveats(kinds) -> str:
    """The per-kind image-quality note(s) for one call, deduplicated in order."""
    out = []
    for kind in dict.fromkeys(kinds):
        lead = IMAGE_KIND_CAVEAT_LEADS.get(kind)
        if lead:
            out.append(f"NOTE ON IMAGE QUALITY: this image is generated -- "
                       f"{lead} -- {IMAGE_QUALITY_INSTRUCTION}")
    if not out:
        out.append(f"NOTE ON IMAGE QUALITY: this image is generated, "
                   f"{IMAGE_QUALITY_INSTRUCTION}")
    return "\n\n".join(out)

# Which kinds carry per-tooth FDI tags, and WHICH WARNING each one earns --
# they are not the same picture. The panoramic outlines every tooth in its own
# colour and prints the matching tag in a margin band, so colour is a genuine
# cross-check there and position is the trap (tags are nudged sideways to avoid
# overlap). The sinus details tag teeth but do NOT outline them, so there is no
# outline to match a colour against and the tag's position is all there is --
# telling that call to "match tag to tooth by COLOR" would name a marker the
# image does not contain, the same failure a caption describing an outline that
# is not drawn would make.
FDI_TAG_BLOCK_OUTLINED = """
READING THE FDI TAGS
─────────────────────
An FDI tag is printed in EXACTLY THE SAME COLOR as the outline of the tooth
it names. Match tag to tooth by COLOR, not by position, and check the number
before you use it -- naming a finding against the neighbouring tooth is the
easiest error here, and a plausible wrong number is worse than none.
"""

FDI_TAG_BLOCK_TAGGED = """
READING THE FDI TAGS
─────────────────────
Each tooth the slice cuts through is tagged with its FDI number, in its own
colour. The teeth themselves are NOT outlined, so there is no outline to match
a colour against -- the tag sits on the tooth it names. Check the number before
you use it: naming a finding against the neighbouring tooth is the easiest
error here, and a plausible wrong number is worse than none.
"""

FDI_TAG_BLOCKS = {
    "panoramic": FDI_TAG_BLOCK_OUTLINED,
    "sinus": FDI_TAG_BLOCK_TAGGED,
}


def image_kind(name: str) -> str:
    """Classify one images_needed name: 3d | panoramic | sinus | tooth | other."""
    if name.startswith("3d_"):
        return "3d"
    if name.startswith("panoramic"):
        return "panoramic"
    if name.startswith("sinus_"):
        return "sinus"
    if name.startswith("tooth_"):
        return "tooth"
    return "other"


def build_images_intro(image_names) -> str:
    """
    The "what you are looking at" paragraph for one call.

    Singular and plural are written separately rather than with an "(s)": the
    plural half is the only one that may ask for the images to be combined, and
    a call with one image must not be told to combine anything.
    """
    names = list(image_names)
    kinds = [image_kind(n) for n in names]
    tail = "\n\n" + build_image_caveats(kinds)
    if len(names) == 1:
        what = IMAGE_KIND_INTROS.get(kinds[0])
        if what:
            return (f"You are given ONE image for this case: {what}. It is "
                    f"preceded by a caption giving its view and what every "
                    f"marker in it means -- read the caption before the "
                    f"image.{tail}")
        return ("You are given ONE image for this case, preceded by a caption "
                "giving its view and what every marker in it means -- read the "
                f"caption before the image.{tail}")
    distinct = {k for k in kinds if k in IMAGE_KIND_INTROS}
    if len(distinct) == 1:
        what = IMAGE_KIND_INTROS[distinct.pop()]
        lead = f"You are given {len(names)} images for this case, each {what}."
    else:
        lead = f"You are given {len(names)} images for this case."
    return (f"{lead} Each is preceded by a caption giving its view and what "
            f"every marker in it means. Read each caption before its image, "
            f"and use the images together to answer.{tail}")


def build_guidance_section(guidance: Optional[str]) -> str:
    """The shared-definitions block, header and all -- or nothing.

    The header used to sit in the templates, so a call whose facts reference
    no _definitions still printed "HOW THESE FINDINGS ARE DEFINED" over the
    word "(none)". That reads as a section the model failed to receive. The
    sinus calls are the ones that hit it: since the visual-evidence rule
    stopped being prepended to every block (v7.1), they reference nothing.
    """
    guidance = (guidance or "").strip()
    if not guidance:
        return ""
    return ("HOW THESE FINDINGS ARE DEFINED\n"
            "───────────────────────────────\n"
            f"{guidance}\n\n")


def build_fdi_tag_block(image_names) -> str:
    """
    The tag-reading warning, only for calls whose images actually tag teeth,
    and in the variant that matches how THIS call's images tag them.

    A call mixing both kinds takes the outlined wording: it is the stricter of
    the two (match by colour, not position) and the panoramic is the image a
    mixed call would be reading tooth identity off.
    """
    kinds = [image_kind(n) for n in image_names]
    for kind in ("panoramic", "sinus"):
        if kind in kinds:
            return FDI_TAG_BLOCKS[kind]
    return ""


CATEGORY_SYSTEM = """\
You are an expert dental/maxillofacial radiologist interpreting CBCT scans.
Answer ONLY in valid JSON — no prose, no markdown fences, no extra keys.
Every field in OUTPUT SCHEMA must appear as a key, even if null or empty.
Use exactly the choice strings given; never invent values.
Every answer object starts with "visual_evidence": describe what you actually
see first, then the other fields — all of which must agree with it.

REPORT ONLY WHAT IS IN THE IMAGE.
Never infer, assume, or imagine anatomy you cannot point to in a specific
image. If a structure is not shown — outside the acquisition, not segmented,
or hidden behind something else — say exactly that in visual_evidence and
answer null, rather than describing what would normally be there. A structure
you expect but cannot see is evidence of ABSENCE, not permission to assume
presence.
"""

CATEGORY_USER_TEMPLATE = """\
{images_intro}

FDI: 11-18 upper right, 21-28 upper left, 31-38 lower left, 41-48 lower right.
{fdi_tags}{tooth_reading}
{guidance}OUTPUT SCHEMA (return a JSON object with exactly these keys)
─────────────────────────────────────────────────────────────
{output_schema}

QUESTIONS
─────────
{questions}
"""

TOOTH_SYSTEM = """\
You are an expert dental radiologist interpreting per-tooth CBCT images.
Answer ONLY in valid JSON — no prose, no markdown fences, no extra keys.
Every field in OUTPUT SCHEMA must appear as a key, even if null or empty.
Use exactly the choice strings given; never invent values.
Every answer object starts with "visual_evidence": describe what you actually
see first, then the other fields — all of which must agree with it.

REPORT ONLY WHAT IS IN THE IMAGE.
Never infer, assume, or imagine anatomy you cannot point to in a specific
image. If a structure is not shown — outside the acquisition, not segmented,
or hidden behind something else — say exactly that in visual_evidence and
answer null, rather than describing what would normally be there. A structure
you expect but cannot see is evidence of ABSENCE, not permission to assume
presence.
"""

TOOTH_USER_TEMPLATE = """\
{images_intro}

ALWAYS FOCUS ON TOOTH {fdi} — the tooth outlined in RED in every panel.
Locate that outline in each panel before you answer.

The caption above says what that outline is. Two consequences for your
answers: a crown, filling, post or implant on tooth {fdi} sits just OUTSIDE
the outline, capping or filling it, and is still a finding for tooth {fdi}
— never answer "no crown" because the bright cap is outside the red line;
and the neighbouring teeth, opposing arch and surrounding bone, which ARE
worth reading as context for brightness, bone level and eruption, never
carry a finding that belongs to tooth {fdi}.
{tooth_reading}
{guidance}OUTPUT SCHEMA (return a JSON object with exactly these keys)
─────────────────────────────────────────────────────────────
{output_schema}

QUESTIONS
─────────
{questions}
"""


# ── VLM Inference ─────────────────────────────────────────────────────────

def build_user_blocks(images: Dict[str, str], captions: Dict[str, str],
                      q_data: dict, user_template: str,
                      extra_fmt: Optional[dict] = None) -> Optional[list]:
    """
    The content of ONE user turn: the interleaved [caption, image] blocks
    followed by the rendered question text. Returns None when none of the
    call's images exist on disk (nothing to send).

    Split out of infer_call so anything that needs a turn in the pipeline's
    exact shape gets it from here instead of re-deriving the template's format
    keys and drifting the moment the template changes. build_call_prompt()
    below is the call-entry-shaped face of it.

    "guidance" (schema v6.1's shared _definitions vocabulary) is defaulted
    rather than required: a qa_pairs.jsonl built before that field existed
    still renders, just without the definitions block.
    "tooth_reading" defaults to empty: only the calls that actually judge
    per-tooth structure pass HOW_TO_READ_A_TOOTH in via extra_fmt, so the
    3D and sinus calls render the template with the section absent.
    Only the images that were actually FOUND on disk describe the call --
    same existence test build_captioned_image_blocks applies, so a call whose
    sinus image never rendered is not told it is looking at one, and the
    count in the intro always matches the number of images actually sent.
    """
    image_blocks = build_captioned_image_blocks(images, captions)
    if not image_blocks:
        return None

    present_names = [k for k, p in images.items() if p and Path(p).exists()]

    fmt_kwargs = {
        "output_schema": q_data.get("output_schema", "{}"),
        "questions": q_data.get("questions", ""),
        "guidance": build_guidance_section(q_data.get("guidance")),
        "n_images": len(image_blocks) // 2,  # each image is a text+image pair
        "images_intro": build_images_intro(present_names),
        "fdi_tags": build_fdi_tag_block(present_names),
        "tooth_reading": "",
    }
    if extra_fmt:
        fmt_kwargs.update(extra_fmt)

    return image_blocks + [{"type": "text", "text": user_template.format(**fmt_kwargs)}]


def build_call_prompt(call_data: dict, user_template: str,
                      extra_fmt: Optional[dict] = None):
    """
    Render one qa_pairs.jsonl call entry into (user_blocks, json_schema).

    The call-entry-shaped face of build_user_blocks(): anything that needs the
    EXACT prompt the pipeline would send can get it without re-deriving it --
    check_prompt_parity.py's token-id comparison of the training and serving
    paths is the one that matters, and --dump-prompt diffs these blocks against
    a stored model_input dump. Re-deriving the prompt from schema.json instead
    would let the two drift apart on the next schema edit, which is precisely
    what a parity check must not do.

    Returns (None, None) if none of the call's images exist on disk.
    """
    q_data = call_data.get("questions", {})

    user_blocks = build_user_blocks(call_data.get("images", {}),
                                    call_data.get("captions", {}),
                                    q_data, user_template, extra_fmt=extra_fmt)
    if user_blocks is None:
        return None, None

    # Older qa_pairs.jsonl files have no "json_schema" key -- fall back to
    # prompt-only schema enforcement rather than failing.
    return user_blocks, q_data.get("json_schema")


def infer_call(client, model: str, call_data: dict, system: str, user_template: str,
               extra_fmt: Optional[dict] = None, max_tokens: int = 8192,
               debug: bool = True, label: str = "",
               parse_retries: int = 3) -> dict:
    """
    Run one VLM call for one entry from qa_pairs.jsonl (a "call" dict with
    "images", "captions", and "questions"). Generic across the "global" and
    "dental_elements" call shapes -- they both have the same {"images",
    "captions", "questions"} structure now.

    Returns {} if none of the call's images exist on disk (nothing to send).

    A call whose answer parses to a truncated object is RESENT, up to
    parse_retries times, and never stored short. call_vllm's own retry cannot
    do this: it retries transport failures, and a half-written object is a
    perfectly successful HTTP 200. Sampling is not deterministic across
    requests at the server's default temperature, so a resend is a real
    second draw rather than a replay; if every draw truncates, the exception
    propagates and run_calls_concurrently records the call as unanswered --
    which the coverage line then shows, instead of a silent one-fact answer.
    """
    user_blocks, json_schema = build_call_prompt(call_data, user_template, extra_fmt)
    if user_blocks is None:
        print(f"    [WARN] {label}: no images found on disk, skipping call", file=sys.stderr)
        return {}

    for attempt in range(1, parse_retries + 1):
        raw = call_vllm(client, model, system, user_blocks, max_tokens=max_tokens,
                        json_schema=json_schema)

        if debug:
            preview = raw[:500] if raw else repr(raw)
            print(f"    [DEBUG] {label} raw response ({len(raw) if raw else 0} chars): {preview}",
                  file=sys.stderr)

        try:
            return parse_json(raw)
        except IncompleteObject as e:
            if attempt == parse_retries:
                raise
            print(f"    [retry {attempt}/{parse_retries - 1}] {label}: "
                  f"incomplete answer, resending -- {e}", file=sys.stderr)


def merge_dicts(dicts: List[dict]) -> dict:
    """Flatten several calls' results into one dict (fields are disjoint by design)."""
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return merged


# ── Case Processing ───────────────────────────────────────────────────────

ALL_FDIS = [11,12,13,14,15,16,17,18,
            21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38,
            41,42,43,44,45,46,47,48]

# Category group(s) that feed into the merged "global" prediction. Just
# "global" now -- mandible/maxilla/dental_arch facts are pooled together at
# the build_vqa_pairs.py stage so cross-section facts needing the same
# images share one call instead of being split into separate category groups.
GLOBAL_GROUPS = ["global"]

# Whether the panoramic read's absence call (see model_absent_teeth) may
# cancel a per-tooth call for a tooth that HAS a composite image. Off: the segmentation that produced
# the image wins, and every imaged tooth is asked about. On: the model's call
# wins and those teeth are skipped, saving calls at the cost of losing the
# detail findings for any tooth it wrongly declared absent (measured on the
# v4 training run: 4-6 segmented teeth per case).
TRUST_MODEL_ABSENCE = False

# Cases whose --resume fill-in failed, collected across the whole run so the
# final summary can state how many summaries are missing instead of leaving
# the discrepancy (N predictions vs M summaries) for someone to notice by hand.
FILL_FAILURES: List[tuple] = []

# Which arch each dental_arch_findings_* fact covers, for model_absent_teeth.
ARCH_FINDINGS_FIELDS = {
    "dental_arch_findings_mandible": [f for f in ALL_FDIS if f >= 31],
    "dental_arch_findings_maxilla":  [f for f in ALL_FDIS if f <= 28],
}


def model_absent_teeth(global_pred: dict) -> set:
    """
    The FDIs the panoramic read says are NOT in the mouth.

    Schema v6.1 has no absent_teeth_* fact any more; the panoramic read is
    dental_arch_findings_{arch}. TWO CONVENTIONS are read here, because a
    prediction file outlives the schema that produced it:

      v6.2 (fixed keys) -- all 16 of the arch's FDIs are answered every time
        and absence is the VALUE "absent". This is the reliable form: a tooth
        the reader never got to is no longer indistinguishable from one that
        is not in the mouth.
      v6.1 (present teeth only) -- the KEY SET was the answer, so absence is
        the complement of the keys within the arch's range.

    Applying both is safe in either direction: a v6.2 answer has no missing
    keys, so the complement is empty, and a v6.1 answer has no "absent"
    values. A partial v6.2 answer (a key genuinely dropped despite the schema
    requiring it) falls back to the v6.1 reading, which is the behaviour that
    was there before -- conservative, and the per-tooth call still runs
    because TRUST_MODEL_ABSENCE is False.

    Either reading is only taken for an arch whose fact actually came back
    with findings: an unanswered or empty call means "nobody looked", and
    reading it as "all 16 teeth of this arch are missing" would cancel every
    per-tooth call in the arch. Older predictions' absent_teeth* lists are
    still honoured so a v4-era qa_pairs/prediction pair keeps working.
    """
    absent: set = set()

    for key in ("absent_teeth", "absent_teeth_mandible", "absent_teeth_maxilla"):
        value = global_pred.get(key)
        if isinstance(value, list):
            absent |= {f for f in value if isinstance(f, int)}

    for field, arch_fdis in ARCH_FINDINGS_FIELDS.items():
        fact = global_pred.get(field)
        findings = fact.get("findings") if isinstance(fact, dict) else None
        if not isinstance(findings, dict) or not findings:
            continue
        answered = set()
        for raw, value in findings.items():
            try:
                fdi = int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            answered.add(fdi)
            if isinstance(value, str) and value.strip().lower() == "absent":
                absent.add(fdi)
        absent |= {f for f in arch_fdis if f not in answered}

    return absent


def write_summary_for_case(prediction: dict, summaries_out_dir: str) -> Optional[dict]:
    """
    Postprocess one case's prediction and persist the intermediate
    {case_id}_summary.json -- the exact compact JSON that synthesize_report.py
    renders. Without this the summary only ever exists in memory, so a bad
    report cannot be traced back to whether postprocessing or the extraction
    was at fault. Same content and filename as postprocess_pred.py's CLI, so
    the two are interchangeable and these files can be hand-corrected.

    Returns the summary, or None if postprocessing failed (logged, never fatal
    -- a summary is a debugging artifact).
    """
    case_id = prediction.get("case_id", "?")
    try:
        summary = postprocess_prediction(prediction)
        out_path = os.path.join(summaries_out_dir, f"{case_id}_summary.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2))
        print(f"  [{case_id}] summary -> {out_path}", file=sys.stderr)
        return summary
    except Exception as e:
        print(f"    [ERROR] {case_id} postprocess/summary: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# ── Batched calling ───────────────────────────────────────────────────────
#
# Every call in a case is an independent request against the same server, but
# they used to be issued one at a time, one in flight, so vLLM's continuous
# batching had nothing to batch and the GPU idled between requests. Measured on
# the 40-case v6.9 validate run (logs/aksssr_v6_553351.log): 1092 calls, 2h34m
# of inference, 4.0 min/case, of which 781 tooth calls at 9.5 s each accounted
# for 80.5% of the time.
#
# What makes it safe to issue ALL of a case's calls together -- global and
# per-tooth -- rather than only the tooth ones is TRUST_MODEL_ABSENCE being off.
# The only thing a tooth call ever needed from the global calls is
# model_absent_teeth(), and with the flag off that set cannot cancel a tooth
# call: the segmentation that produced the composite wins, so `to_process` is
# just "every tooth with an image", which is known from qa_pairs.jsonl before
# any request is sent. Turn TRUST_MODEL_ABSENCE on and the dependency comes
# back, so the code below keeps the barrier for that case rather than assuming
# it away -- two batched phases instead of one.
DEFAULT_MAX_CONCURRENCY = 8


class _ThreadRoutedStderr:
    """sys.stderr proxy that buffers writes per worker thread.

    Every diagnostic this file emits -- infer_call's raw-response preview,
    parse_json's repair warnings, the retry lines -- is written by whichever
    thread happens to be running. Left alone they interleave line by line and
    stop being attributable to a call, and those logs are exactly what the
    project reads its parse-failure and repair rates out of (the 31-against-1
    unterminated-object asymmetry between two prompting arms was counted from
    them).

    Routing instead of locking each print keeps a call's whole diagnostic
    block together, and needs no signature change in parse_json or anything
    else that already writes to stderr.
    """

    def __init__(self, real):
        self._real = real
        self._local = threading.local()

    def _buf(self):
        return getattr(self._local, "buf", None)

    def write(self, s):
        buf = self._buf()
        if buf is None:
            return self._real.write(s)
        return buf.write(s)

    def flush(self):
        if self._buf() is None:
            self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)

    @contextlib.contextmanager
    def capture(self):
        self._local.buf = io.StringIO()
        try:
            yield self._local.buf
        finally:
            self._local.buf = None


_STDERR_ROUTER: Optional[_ThreadRoutedStderr] = None
_EMIT_LOCK = threading.Lock()


def _install_stderr_router() -> _ThreadRoutedStderr:
    """Install the router on first use, not at import.

    Other tools import this module for build_call_prompt() and parse_json();
    replacing sys.stderr as an import side effect would change their logging
    too, for no reason. Only the batched path needs it.
    """
    global _STDERR_ROUTER
    if _STDERR_ROUTER is None:
        _STDERR_ROUTER = _ThreadRoutedStderr(sys.stderr)
        sys.stderr = _STDERR_ROUTER
    return _STDERR_ROUTER


def run_calls_concurrently(jobs: List[Tuple[str, Callable[[], dict]]],
                           max_workers: int,
                           progress_every: int = 8,
                           progress_prefix: str = "") -> List[dict]:
    """Run (label, thunk) jobs against the server, returning results IN ORDER.

    Each thunk returns a parsed call result; an exception becomes {} with an
    [ERROR] line, which is what the sequential loops did per call. Results are
    returned in submission order regardless of completion order, because
    merge_dicts() and the tooth loop below both depend on it.

    max_workers <= 1 runs them sequentially and without the stderr router, so
    the old behaviour -- and the old log format -- is recoverable exactly with
    --max-concurrency 1.
    """
    if not jobs:
        return []

    if max_workers <= 1:
        results = []
        for i, (label, fn) in enumerate(jobs, 1):
            try:
                results.append(fn())
            except Exception as e:
                print(f"    [ERROR] {label}: {e}", file=sys.stderr)
                results.append({})
            if progress_every and (i % progress_every) == 0:
                print(f"    {progress_prefix}{i}/{len(jobs)} calls", file=sys.stderr)
        return results

    router = _install_stderr_router()
    real_stderr = router._real

    def _worker(label: str, fn: Callable[[], dict]):
        with router.capture() as buf:
            try:
                result = fn()
            except Exception as e:
                print(f"    [ERROR] {label}: {e}", file=sys.stderr)
                result = {}
        return result, buf.getvalue()

    results: List[Optional[dict]] = [None] * len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, label, fn): i
                   for i, (label, fn) in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                result, logged = fut.result()
            except Exception as e:                      # a worker itself died
                label = jobs[i][0]
                result, logged = {}, f"    [ERROR] {label}: {e}\n"
            results[i] = result
            done += 1
            with _EMIT_LOCK:
                if logged:
                    real_stderr.write(logged)
                if progress_every and (done % progress_every) == 0:
                    real_stderr.write(f"    {progress_prefix}{done}/{len(jobs)} calls\n")
                real_stderr.flush()

    return [r if r is not None else {} for r in results]


def process_case(client, model: str, record: dict, out_dir: str,
                  dry_run: bool = False, debug: bool = True,
                  summaries_out_dir: Optional[str] = None,
                  max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> dict:
    """
    Process one case:
      1. Call vLLM once per call in "global", merging every call's result
         into one flat prediction["global"] dict.
      2. Call vLLM once per tooth in "dental_elements" that (a) the model
         didn't call absent, AND (b) actually has at least one image.
      1+2 are issued TOGETHER, up to max_concurrency in flight, because with
         TRUST_MODEL_ABSENCE off nothing in step 2 depends on step 1 -- see
         run_calls_concurrently(). Results are still assembled in call order.
      3. If summaries_out_dir is given, write the postprocessed
         {case_id}_summary.json -- the intermediate synthesize_report.py
         renders -- see write_summary_for_case().
    """
    case_id = record["case_id"]
    prediction = {"case_id": case_id, "global": {}, "teeth": {}}

    # Passed straight through from qa_pairs.jsonl -- the mask's own account of
    # which arches the acquisition covers. Nothing here reads it; it exists so
    # postprocess_pred.py can tell a tooth with no composite because the
    # segmentation has no label for it (absent) from one with no composite
    # because its whole arch was skipped (unknown). See build_vqa_pairs.py.
    if record.get("coverage"):
        prediction["coverage"] = record["coverage"]

    # ── Build every call this case needs ─────────────────────────────────
    tooth_records = record.get("dental_elements", {})
    has_image = [fdi for fdi in ALL_FDIS
                 if tooth_records.get(f"tooth_{fdi}", {}).get("images")]

    global_jobs: List[Tuple[str, Callable[[], dict]]] = []
    for group_name in GLOBAL_GROUPS:
        group_calls = record.get(group_name, {})
        print(f"  [{case_id}] {group_name} ({len(group_calls)} call(s))...", file=sys.stderr)

        for call_key, call_data in group_calls.items():
            label = f"{group_name}/{call_key}"
            global_jobs.append((label, functools.partial(
                infer_call, client, model, call_data,
                CATEGORY_SYSTEM, CATEGORY_USER_TEMPLATE,
                extra_fmt={"tooth_reading": (
                    "\n" + HOW_TO_READ_THE_PANORAMIC
                    if call_key in TOOTH_READING_CALLS else "")},
                max_tokens=8192, debug=debug, label=label)))

    def tooth_job(fdi: int) -> Tuple[str, Callable[[], dict]]:
        label = f"tooth {fdi}"
        return (label, functools.partial(
            infer_call, client, model, tooth_records.get(f"tooth_{fdi}", {}),
            TOOTH_SYSTEM, TOOTH_USER_TEMPLATE,
            extra_fmt={"fdi": fdi, "tooth_reading": "\n" + HOW_TO_READ_A_TOOTH},
            max_tokens=4096, debug=debug, label=label))

    # ── Run them ─────────────────────────────────────────────────────────
    # A tooth only gets a composite because create_tooth_detail.py found its
    # label in the mask, so an image outranks a panoramic read that calls the
    # tooth absent -- the model reading a 2D projection is the weaker source.
    # The gate therefore only ever skips teeth that have no image anyway
    # (already covered by has_image); contested teeth are still processed, and
    # named below because that disagreement is worth seeing in the log.
    # Flip TRUST_MODEL_ABSENCE to let the model's call win instead -- which is
    # also what reinstates the barrier between the two batches.
    if dry_run:
        global_results = [{} for _ in global_jobs]
        to_process = list(has_image)
        tooth_results = [{} for _ in to_process]
        absent = set()
        print(f"  [{case_id}] dental_elements ({len(to_process)} teeth with images)...",
              file=sys.stderr)

    elif TRUST_MODEL_ABSENCE:
        # The gate is live, so which teeth to ask about is not known until the
        # panoramic read is in: two batched phases, with a barrier between.
        global_results = run_calls_concurrently(
            global_jobs, max_concurrency, progress_prefix=f"[{case_id}] global ")
        absent = model_absent_teeth(merge_dicts(global_results))
        to_process = [fdi for fdi in has_image if fdi not in absent]
        print(f"  [{case_id}] dental_elements ({len(to_process)} teeth to call, "
              f"{len(has_image)} with images, {len(absent)} marked absent by model)...",
              file=sys.stderr)
        tooth_results = run_calls_concurrently(
            [tooth_job(fdi) for fdi in to_process], max_concurrency,
            progress_prefix=f"[{case_id}] teeth ")

    else:
        # Nothing in the tooth calls depends on the global ones, so the whole
        # case goes out as one batch -- which is the point: the global calls
        # are 28.5% of the requests and were being paid for serially.
        to_process = list(has_image)
        jobs = global_jobs + [tooth_job(fdi) for fdi in to_process]
        print(f"  [{case_id}] dental_elements ({len(to_process)} teeth with images)...",
              file=sys.stderr)
        print(f"  [{case_id}] {len(jobs)} call(s), up to {max_concurrency} in flight...",
              file=sys.stderr)
        results = run_calls_concurrently(jobs, max_concurrency,
                                         progress_prefix=f"[{case_id}] ")
        global_results = results[:len(global_jobs)]
        tooth_results = results[len(global_jobs):]
        absent = model_absent_teeth(merge_dicts(global_results))

    global_pred = merge_dicts(global_results)
    prediction["global"] = global_pred

    contested = sorted(absent.intersection(has_image))
    if contested:
        verb = "skipped" if TRUST_MODEL_ABSENCE else "called anyway"
        print(f"    [WARN] model called {contested} absent but they are "
              f"segmented and have images -- {verb}", file=sys.stderr)

    for fdi, result in zip(to_process, tooth_results):
        result["detected"] = "yes"
        prediction["teeth"][f"tooth_{fdi}"] = result

    # Mark model-declared-absent teeth -- but NEVER overwrite a tooth that was
    # actually called. With TRUST_MODEL_ABSENCE off, `to_process` deliberately
    # includes the contested teeth (segmented, imaged, yet called absent by the
    # panoramic read) so the stronger source gets its say; a blind assignment
    # here threw that answer away immediately afterwards, silently undoing the
    # policy above and the "still calling them" warning with it. On the v4
    # validate run that destroyed 3 completed tooth calls (F043 tooth 43,
    # A022 teeth 24 and 25).
    for fdi in absent:
        tooth_key = f"tooth_{fdi}"
        if tooth_key in prediction["teeth"]:
            continue
        prediction["teeth"][tooth_key] = {"detected": "no"}

    # Mark teeth with no image at all (never had any composite generated)
    for fdi in ALL_FDIS:
        tooth_key = f"tooth_{fdi}"
        if tooth_key not in prediction["teeth"]:
            prediction["teeth"][tooth_key] = {"detected": "no_image"}

    # ── Write output ─────────────────────────────────────────────────
    if not dry_run:
        out_path = os.path.join(out_dir, f"{case_id}_pred.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(prediction, indent=2))
        print(f"  [{case_id}] done -> {out_path}", file=sys.stderr)

    # ── Postprocessed summary (optional) ──────────────────────────────
    if not dry_run and summaries_out_dir is not None:
        write_summary_for_case(prediction, summaries_out_dir)

    return prediction


def resolve_path(path: Optional[str], base_dir: Optional[str]) -> Optional[str]:
    """
    Resolve an image path from qa_pairs.jsonl against base_dir.

    build_vqa_pairs.py stores paths RELATIVE to the project root (via
    --project-dir), so this file works unchanged whether read on the host
    (base_dir = $PROJECT_DIR) or inside a container (base_dir = /project,
    the mount point) -- just join with whatever base is in scope here.

    If path is already absolute (old-format qa_pairs.jsonl, or a fallback
    from build_vqa_pairs.py when a file wasn't under project_dir), it's
    left unchanged.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if not base_dir:
        return path
    return os.path.join(base_dir, path)


def resolve_record_paths(record: Dict, base_dir: Optional[str]) -> Dict:
    """
    Resolve every image path inside one record in place. Every call
    everywhere in the record ("global" calls,
    dental_elements per-tooth calls) now has the same {"images": {key:
    path}} shape, so this just walks all three sections uniformly.
    """
    if not base_dir:
        return record

    for group_name in GLOBAL_GROUPS:
        for call_data in record.get(group_name, {}).values():
            images = call_data.get("images", {})
            for k, v in list(images.items()):
                images[k] = resolve_path(v, base_dir)

    for tooth_data in record.get("dental_elements", {}).values():
        images = tooth_data.get("images", {})
        for k, v in list(images.items()):
            images[k] = resolve_path(v, base_dir)

    return record


def load_vqa_records(vqa_jsonl: str, base_dir: Optional[str] = None) -> List[Dict]:
    """Load vqa records (one per case) from qa_pairs.jsonl.

    If base_dir is given, every relative image path in every record is
    resolved against it (see resolve_path()).
    """
    records = []
    with open(vqa_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                record = resolve_record_paths(record, base_dir)
                records.append(record)
    return records


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="VLM inference over global/dental_elements calls.")

    ap.add_argument("--vqa-jsonl", required=True,
                    help="qa_pairs.jsonl from build_vqa_pairs.py")
    ap.add_argument("--out-dir", required=True,
                    help="Output predictions directory")
    ap.add_argument("--model", default="qwen3.5-vl",
                    help="Model name (served-model-name on the vLLM server)")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--case-ids", nargs="+", default=None,
                    help="Filter case IDs (optional)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max cases (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip the (expensive, image-based) inference for a case if "
                         "{case_id}_pred.json already exists. With "
                         "--summaries-out-dir, a case whose prediction exists but "
                         "whose {case_id}_summary.json does not still gets its "
                         "summary built from the existing prediction -- local "
                         "postprocessing, no inference at all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip vLLM calls (validate structure only)")
    ap.add_argument("--max-concurrency", type=int,
                    default=int(os.environ.get("MAX_CONCURRENCY",
                                               DEFAULT_MAX_CONCURRENCY)),
                    help="How many of a case's calls may be in flight at once. "
                         "The calls within a case are independent (see "
                         "run_calls_concurrently), and issuing them one at a "
                         "time left vLLM's continuous batching with nothing to "
                         "batch. 1 restores the old strictly-sequential path, "
                         "including its log format. Raising it past what the "
                         "server's KV cache holds does not fail -- vLLM queues "
                         "-- it just stops helping. Env: MAX_CONCURRENCY.")
    ap.add_argument("--no-debug", action="store_true",
                    help="Suppress raw-response debug logging")
    ap.add_argument("--base-dir", default=None,
                    help="Directory that image paths in qa_pairs.jsonl (which are "
                         "stored relative to the project root) should be resolved "
                         "against. Pass $PROJECT_DIR when running on the host, or "
                         "the container mount point (e.g. /project) when running "
                         "inside a container.")
    ap.add_argument("--summaries-out-dir", default=None,
                    help="If given, ALSO write each case's postprocessed "
                         "{summaries_out_dir}/{case_id}_summary.json -- the compact "
                         "intermediate that postprocess_pred.py builds and that "
                         "synthesize_report.py renders. Same content and filenames "
                         "as running postprocess_pred.py separately, so it can be "
                         "inspected, diffed or hand-corrected without re-running "
                         "inference.")

    args = ap.parse_args()

    if not Path(args.vqa_jsonl).exists():
        print(f"[FAIL] {args.vqa_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Loading qa_pairs.jsonl...", file=sys.stderr)
    if args.base_dir:
        print(f"[INFO] Resolving relative image paths against: {args.base_dir}",
              file=sys.stderr)
    records = load_vqa_records(args.vqa_jsonl, args.base_dir)

    if not records:
        print("[FAIL] qa_pairs.jsonl is empty", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(records)} case(s)", file=sys.stderr)

    if args.case_ids:
        records = [r for r in records if r["case_id"] in args.case_ids]
        print(f"[INFO] Filtered to {len(records)} case(s)", file=sys.stderr)

    if args.limit:
        records = records[:args.limit]
        print(f"[INFO] Limited to {len(records)}", file=sys.stderr)

    if not args.dry_run:
        import openai
        client = openai.OpenAI(base_url=args.vllm_url, api_key="none")
    else:
        client = None
        print("[INFO] DRY-RUN (no vLLM calls)", file=sys.stderr)

    if args.summaries_out_dir:
        os.makedirs(args.summaries_out_dir, exist_ok=True)
        print(f"[INFO] Postprocessed summaries will be written -> "
              f"{args.summaries_out_dir} (the exact JSON handed to the report LLM)",
              file=sys.stderr)
    print("\n", file=sys.stderr)
    processed = 0
    skipped = 0
    filled_in = 0

    for rec in records:
        case_id = rec["case_id"]
        out_path = os.path.join(args.out_dir, f"{case_id}_pred.json")

        if args.resume and os.path.exists(out_path):
            # The prediction is done, but its derived summary may not be (an
            # earlier run died before the summary stage, or ran without the
            # flag). Build it from the prediction on disk -- postprocessing is
            # local, so this costs no inference. Skipping outright here is what
            # silently ended a 40-prediction / 0-summary run.
            summary_path = (os.path.join(args.summaries_out_dir, f"{case_id}_summary.json")
                            if args.summaries_out_dir else None)
            need_summary = bool(summary_path) and not os.path.exists(summary_path)

            if need_summary and not args.dry_run:
                print(f"[fill] {case_id} (prediction exists, missing: summary)",
                      file=sys.stderr)
                try:
                    prediction = json.loads(Path(out_path).read_text())
                except Exception as e:
                    print(f"  [FAIL] {case_id}: unreadable prediction {out_path}: {e}",
                          file=sys.stderr)
                    FILL_FAILURES.append((case_id, f"unreadable prediction: {e}"))
                    continue
                prediction.setdefault("case_id", case_id)

                write_summary_for_case(prediction, args.summaries_out_dir)
                filled_in += 1
                continue

            print(f"[skip] {case_id}", file=sys.stderr)
            skipped += 1
            continue

        print(f"[infer] {case_id}", file=sys.stderr)
        try:
            process_case(client, args.model, rec, args.out_dir,
                         dry_run=args.dry_run, debug=not args.no_debug,
                         summaries_out_dir=args.summaries_out_dir,
                         max_concurrency=args.max_concurrency)
            processed += 1
        except Exception as e:
            print(f"  [FAIL] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[INFO] Done: {processed} processed, {filled_in} filled in from an "
          f"existing prediction, {skipped} skipped", file=sys.stderr)

    if FILL_FAILURES:
        print(f"[WARN] {len(FILL_FAILURES)} case(s) produced a prediction but NO "
              f"summary -- predictions/ and summaries/ will not match:",
              file=sys.stderr)
        for case_id, err in FILL_FAILURES:
            print(f"[WARN]   {case_id}: {err}", file=sys.stderr)
        print("[WARN] Rebuild the summaries and reports from the predictions with:",
              file=sys.stderr)
        print("[WARN]   scripts/postprocess_now.sh <run_dir>",
              file=sys.stderr)