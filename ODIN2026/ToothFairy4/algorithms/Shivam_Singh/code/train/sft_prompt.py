#!/usr/bin/env python3
"""
sft_prompt.py -- one SFT row -> the exact messages the pipeline would send.

docs/vision_sft_plan.md §3.4. This is the ONLY place that turns a row of
`sft_*.jsonl` into a chat conversation, and both sides of the experiment go
through it: check_prompt_parity.py compares its output against vLLM's own
tokenizer, and trainer.py's collator builds its training text from
it. Two copies of a prompt drift, and the drift is invisible -- the same
argument build_sft_targets.py makes for storing the call inputs rather than a
rendered prompt.

Nothing here formats a prompt itself. `run_vqa_inference.build_call_prompt` is
imported and called, so a schema edit that changes TOOTH_USER_TEMPLATE changes
training and serving together or not at all.

TWO THINGS IT DOES THAT THE PIPELINE DOES NOT, BOTH DELIBERATE
──────────────────────────────────────────────────────────────
1. **Image paths are resolved against the project root, not the working
   directory.** qa_pairs.jsonl stores them project-relative so one payload
   resolves on the host and at /project inside the container, and
   build_captioned_image_blocks() resolves them against `os.getcwd()`. That is
   correct for the pipeline, which runs from the project dir; it is a trap for
   anything else, and the plan's log records what the trap cost -- a teacher job with no
   working directory drafted 9,301 evidence strings having never loaded an
   image, because the resolver SILENTLY DROPS what it cannot find.

2. **A missing image raises.** `build_call_prompt` returns None and the
   pipeline logs a warning and skips the call, which is right at inference
   time: a case with one unrendered image should still produce a report. In
   training it is not right. A row whose image vanished would be trained as
   text-only against a target full of visual findings -- exactly the "assert
   what you cannot see" behaviour the experiment exists to remove. So
   `NoImage` propagates and the caller counts it.
"""

from __future__ import annotations

import sys
from pathlib import Path


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

ROOT = REPO_ROOT

import run_vqa_inference as RVI          # noqa: E402  -- prompt parity, §3.4

# What call_vllm_messages() sends as chat_template_kwargs on EVERY pipeline
# call. Named here because the chat template acts on it -- with thinking off it
# emits a literal "<think>\n\n</think>\n\n" after the generation prompt, and a
# collator that leaves it out trains the model on a prefix the server will
# never send it (R8).
ENABLE_THINKING = False

# HOW THE SERVER ASSEMBLES A MULTIMODAL TURN -- MEASURED, NOT ASSUMED.
#
# vLLM resolves a "chat template content format" per server by inspecting the
# template's Jinja AST, and guesses `string` for Qwen3.5's. In that mode it does
# NOT render the content parts in order: _get_full_multimodal_text_prompt()
# hoists every image placeholder to the front of the turn and joins the text
# parts after it with "\n".
#
# So the caption this pipeline carefully puts BEFORE each image arrives after
# it, plus two newline tokens that are in no message anywhere. Job 555208
# measured it -- 6,126 host tokens against 6,128 served, diverging at the first
# token of the user turn -- and flatten_string_format() below reproduces the
# server's sequence exactly.
#
# THIS IS SET TO WHAT THE SERVER DOES, NOT TO WHAT THE PIPELINE INTENDS, and
# that is a deliberate decision (2026-08-14). The alternative -- serving with
# --chat-template-content-format openai -- restores the intended interleave,
# but it changes the prompt every run in this repo has been sent, so it
# invalidates the §5 baseline and every arm scored against it. Training
# against the prompt the model will actually receive costs nothing and keeps
# the whole result history comparable; making the caption-first design real is
# a separate, measured experiment (the plan's log, 'The image is hoisted').
#
# When that experiment is run, flip this to "openai" and re-baseline. Nothing
# else in the training path needs to know.
CONTENT_FORMAT = "string"

# What the chat template emits for one image, whatever the content format.
PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class NoImage(FileNotFoundError):
    """A row's image is not on disk. Never silently a text-only sample."""


def resolve_call(call: dict, root: Path = ROOT) -> dict:
    """The call entry with absolute image paths, or raise NoImage.

    build_captioned_image_blocks() tests `Path(p).exists()` against the working
    directory. Making the paths absolute first means the answer does not depend
    on where the job happened to be launched from.
    """
    images = call.get("images") or {}
    resolved, missing = {}, []
    for key, rel in images.items():
        if not rel:
            missing.append(key)
            continue
        p = Path(rel)
        p = p if p.is_absolute() else (root / p)
        if not p.exists():
            missing.append(f"{key} -> {p}")
            continue
        resolved[key] = str(p)
    if missing or not resolved:
        raise NoImage(f"{len(missing)} image(s) not on disk: {missing}")
    return {**call, "images": resolved}


def row_label(row: dict) -> str:
    """`A013 tooth 33` / `A013 pano_others_mandible`, for errors and --show.

    Arch rows carry no `fdi`, so formatting one into an error message raises
    KeyError from inside the handler and reports the wrong failure.
    """
    who = f"tooth {row['fdi']}" if "fdi" in row else row.get("call_key", "?")
    return f"{row['case_id']} {who}"


def build_row_messages(row: dict, root: Path = ROOT) -> tuple[list, dict | None]:
    """(messages, json_schema) for one SFT row -- the pipeline's own shape.

    One system turn and one user turn, the user turn being [caption, image]
    blocks followed by the rendered question text.

    THE TWO CALL KINDS DO NOT SHARE A PROMPT, AND §3.4 IS ABOUT EXACTLY THIS.
    ────────────────────────────────────────────────────────────────────────
    `run_vqa_inference.py` serves `dental_elements` with TOOTH_SYSTEM /
    TOOTH_USER_TEMPLATE and `extra_fmt={fdi, HOW_TO_READ_A_TOOTH}`, and the
    `global` group with **CATEGORY_SYSTEM / CATEGORY_USER_TEMPLATE** and
    `HOW_TO_READ_THE_PANORAMIC`, applied only to the four calls in
    TOOTH_READING_CALLS. CATEGORY_USER_TEMPLATE has no `{fdi}` field at all.

    Training an arch row through the tooth template would produce a prompt the
    server never sends -- the same class of defect as §3.4's caption hoisting,
    which sat under every run in this repo until it was measured. So the kind is
    dispatched from the row, and the constants come from the serving module
    rather than being restated here.
    """
    call = resolve_call(row["call"], root)
    if row.get("kind") == "arch":
        call_key = row.get("call_key", "")
        system = RVI.CATEGORY_SYSTEM
        template = RVI.CATEGORY_USER_TEMPLATE
        extra = {"tooth_reading": ("\n" + RVI.HOW_TO_READ_THE_PANORAMIC
                                   if call_key in RVI.TOOTH_READING_CALLS
                                   else "")}
    else:
        system = RVI.TOOTH_SYSTEM
        template = RVI.TOOTH_USER_TEMPLATE
        extra = {"fdi": row["fdi"],
                 "tooth_reading": "\n" + RVI.HOW_TO_READ_A_TOOTH}

    user_blocks, json_schema = RVI.build_call_prompt(call, template,
                                                     extra_fmt=extra)
    if user_blocks is None:
        # resolve_call already proved the files exist, so this is a real defect
        # in the call entry (no images key at all), not a missing render.
        raise NoImage(f"{row_label(row)}: no image blocks")
    return ([{"role": "system", "content": system},
             {"role": "user", "content": user_blocks}], json_schema)


def image_paths(row: dict, root: Path = ROOT) -> list[str]:
    """The row's images, in the order build_captioned_image_blocks emits them.

    The processor pairs pixel values with `<|image_pad|>` runs positionally, so
    this order is the one thing the collator may not get wrong.
    """
    return list(resolve_call(row["call"], root)["images"].values())


def flatten_string_format(messages: list, placeholder: str = PLACEHOLDER) -> list:
    """The conversation as vLLM's `string` content format assembles it.

    Placeholders first, then "\\n", then the text parts joined by "\\n" --
    vllm/entrypoints/chat_utils.py, `_get_full_multimodal_text_prompt`. Every
    turn becomes a plain string, which is what the chat template then renders.

    Not a diagnostic: with CONTENT_FORMAT == "string" this is the training
    prompt, because it is the serving prompt.
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            texts = [b["text"] for b in content if b.get("type") == "text"]
            n_img = sum(1 for b in content if b.get("type") == "image_url")
            joined = "\n".join(texts)
            content = ("\n".join([placeholder] * n_img) + "\n" + joined
                       if n_img else joined)
        out.append({**m, "content": content})
    return out


def render_chat_text(processor, messages: list,
                     content_format: str = None) -> str:
    """The prompt text, exactly as the server would build it.

    One function for both sides of §3.4: check_prompt_parity.py compares its
    output against POST /tokenize, and the collator tokenizes it. Anything that
    renders a training prompt some other way is out of parity by construction.
    """
    fmt = content_format or CONTENT_FORMAT
    if fmt == "string":
        conv = flatten_string_format(messages)
    elif fmt == "openai":
        conv = strip_images(messages)
    else:
        raise ValueError(f"unknown content format {fmt!r}")
    return processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING)


def strip_images(messages: list) -> list:
    """The same conversation with each image block replaced by a marker.

    Chat templates emit `<|vision_start|><|image_pad|><|vision_end|>` for any
    block carrying an `image`/`image_url` key, regardless of what is in it, so
    the rendered TEXT is identical either way. Useful for printing a prompt, or
    for tokenizing one without carrying ~0.7 MB of base64 per image.
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [{"type": "image"} if b.get("type") == "image_url" else b
                       for b in content]
        out.append({**m, "content": content})
    return out
