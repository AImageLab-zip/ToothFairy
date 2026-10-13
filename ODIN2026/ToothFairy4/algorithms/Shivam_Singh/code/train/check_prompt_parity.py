#!/usr/bin/env python3
"""
check_prompt_parity.py -- docs/vision_sft_plan.md §3.4's acceptance test.

Render one tooth call through the TRAINING path and through the SERVING path
and require identical token ids. A token-id diff, not a text diff: the failures
this is for -- `enable_thinking`, `add_generation_prompt`, the caption/image
interleave order -- are all invisible in a rendered string that looks right.

WHY THIS RUNS BEFORE ANY TRAINING
─────────────────────────────────
It costs a server rather than a GPU-hour, and every hour spent after it is
spent on the assumption it verifies: that the prompt the collator builds is the
prompt the server sends. The plan's log entry 'The teacher pass that never saw an image' is the argument. A job with no working directory
resolved every image path to nothing, `build_captioned_image_blocks` silently
skipped what it could not find, and a teacher wrote 9,301 fluent strings having
never seen an image -- and nothing downstream could tell, because a prompt that
is missing something still renders.

THE THREE COMPARISONS
─────────────────────
  A. **template**   -- the chat text, unexpanded, host tokenizer vs vLLM
     /tokenize. This is the one §3.4 asks for: same special tokens, same turn
     structure, same thinking prefix, same block order.
  B. **expansion**  -- the host processor's `<|image_pad|>` run against the
     grid arithmetic (h/16/2 x w/16/2). What the collator will feed the model.
  C. **schema**     -- the call's json_schema is present, since STRUCTURED
     decoding is what the served answer is constrained by (R2).

vLLM's /tokenize may or may not expand multimodal placeholders into the full
`<|image_pad|>` run -- that belongs to the engine's input processor, not the
template. That is not a mismatch, and this script says which regime it saw
rather than assuming either: A compares against the unexpanded host ids, and if
the server did expand, A is compared against the expanded ones instead.

THE FAILURE IT FOUND FIRST (job 555208), AND WHY IT IS DIAGNOSED HERE
─────────────────────────────────────────────────────────────────────
vLLM resolves a "chat template content format" per server -- `string` or
`openai` -- by inspecting the template's Jinja AST, and guesses `string` for
Qwen3.5's. In that mode `_get_full_multimodal_text_prompt` does not render the
content parts in order: it HOISTS every image placeholder to the front of the
turn, then joins the text parts after it with "\n". So the pipeline's
caption-then-image interleave arrives at the model as image-then-caption, plus
two newline tokens that are not in the conversation anywhere.

That is a one-line report if the script names it and a long afternoon if it
prints a token window and stops, so `diagnose_images_first()` reconstructs the
hoisted arrangement on the host and checks whether it reproduces the server's
ids exactly. Confirming the cause is the same work as ruling it out.
"""

from __future__ import annotations

import argparse
import json
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

import sft_prompt as SP                   # noqa: E402

IMAGE_PAD = "<|image_pad|>"


def load_row(rows_path: Path, fdi: int | None, case_id: str | None) -> dict:
    """The first row matching --fdi/--case, or the first row at all."""
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if fdi is not None and row["fdi"] != fdi:
            continue
        if case_id and row["case_id"] != case_id:
            continue
        return row
    sys.exit(f"[FAIL] no row in {rows_path} matching fdi={fdi} case={case_id}")


def first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def report_divergence(tok, a: list[int], b: list[int], names=("host", "vllm")) -> None:
    """Print the neighbourhood of the first differing token, decoded.

    A raw index is not actionable; the ~10 tokens either side of it name the
    field, the turn or the special token that moved.
    """
    i = first_divergence(a, b)
    lo, hi = max(0, i - 12), i + 12
    print(f"  first divergence at index {i} "
          f"({names[0]} {len(a)} tokens, {names[1]} {len(b)} tokens)")
    for name, ids in ((names[0], a), (names[1], b)):
        window = ids[lo:hi]
        print(f"    {name:6} [{lo}:{hi}] ids   {window}")
        print(f"    {name:6} [{lo}:{hi}] text  "
              f"{tok.decode(window, skip_special_tokens=False)!r}")


def ids_under(proc, messages: list, fmt: str, n_vision: int) -> list[int]:
    """The host's token ids under one content format, expanded to match a
    server that expanded.

    Re-expanding the single placeholder is only sound for a single image, which
    every tooth call has; with two images of different sizes the per-image
    split is not recoverable from a total, so it is left alone rather than
    guessed.
    """
    text = SP.render_chat_text(proc, messages, content_format=fmt)
    ids = proc.tokenizer(text, add_special_tokens=False).input_ids
    pad = proc.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    if n_vision > ids.count(pad) == 1:
        i = ids.index(pad)
        ids = ids[:i] + [pad] * n_vision + ids[i + 1:]
    return ids


def vllm_tokenize(url: str, model: str, messages: list, timeout: int) -> tuple[list[int], str]:
    """POST /tokenize with the pipeline's own chat_template_kwargs.

    Returns (token_ids, note). `chat_template_kwargs` carries enable_thinking,
    exactly as call_vllm_messages() puts it in extra_body -- if the server
    ignores it, the thinking prefix is where A diverges, which is the point.
    """
    import requests

    body = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": SP.ENABLE_THINKING},
    }
    r = requests.post(f"{url.rstrip('/')}/tokenize", json=body, timeout=timeout)
    if r.status_code != 200:
        sys.exit(f"[FAIL] /tokenize returned {r.status_code}: {r.text[:500]}")
    data = r.json()
    if "tokens" not in data:
        sys.exit(f"[FAIL] /tokenize returned no `tokens` key: {list(data)}")
    return data["tokens"], f"count={data.get('count')}"


def served_model_name(url: str, timeout: int) -> str:
    import requests
    r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=timeout)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path,
                    default=ROOT / "outputs/training_results/vsft_pool_training/sft_targets.jsonl")
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "models/Qwen3.5-9B-AWQ")
    ap.add_argument("--fdi", type=int, default=46,
                    help="§3.4 names tooth 46 -- a lower molar, so the call "
                         "carries mandible_canal and is the LONGEST shape")
    ap.add_argument("--case", help="pin a specific case id")
    ap.add_argument("--vllm-url", default=None,
                    help="e.g. http://localhost:8000 . Omitted = host side only "
                         "(comparisons B and C), which needs no GPU.")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dump", type=Path, help="write the rendered prompt text here")
    ap.add_argument("--content-format", choices=("string", "openai"),
                    default=SP.CONTENT_FORMAT,
                    help="how the server assembles a multimodal turn. The "
                         "default is what vLLM measurably DOES here, not what "
                         "the pipeline intends -- see sft_prompt.CONTENT_FORMAT.")
    ap.add_argument("--save-ids", type=Path,
                    help="on a mismatch, write both id sequences here so the "
                         "diff can be worked on without a GPU")
    args = ap.parse_args()

    row = load_row(args.rows, args.fdi, args.case)
    print(f"[INFO] row: {row['case_id']} tooth {row['fdi']} "
          f"({row['supervised']} supervised field(s))")

    # ── build the messages, once, through the pipeline's own builders ──────
    messages, json_schema = SP.build_row_messages(row)
    paths = SP.image_paths(row)
    n_img = sum(1 for b in messages[1]["content"] if b.get("type") == "image_url")
    print(f"[INFO] {len(messages)} turn(s), {n_img} image(s): "
          f"{[Path(p).name for p in paths]}")
    if n_img != len(paths):
        print(f"[FAIL] {n_img} image block(s) built from {len(paths)} path(s)")
        return 1

    # ── C: structured decoding ────────────────────────────────────────────
    if json_schema:
        print(f"[PASS] C schema: json_schema present "
              f"({len(json_schema.get('properties') or {})} propert(ies))")
    else:
        print("[FAIL] C schema: the call carries no json_schema -- the served "
              "answer would be prompt-only (R2)")

    # ── host side: chat template, then the processor ──────────────────────
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    tok = proc.tokenizer

    text = SP.render_chat_text(proc, messages, content_format=args.content_format)
    print(f"[INFO] content format {args.content_format!r} -- "
          + ("images hoisted to the front of the turn, text joined with "
             "newlines (what the server does)" if args.content_format == "string"
             else "content parts rendered in order"))
    if args.dump:
        args.dump.write_text(text, encoding="utf-8")
        print(f"[INFO] rendered prompt -> {args.dump} ({len(text):,} chars)")

    thinking_prefix = "<think>\n\n</think>"
    print(f"[INFO] thinking prefix {'present' if thinking_prefix in text else 'ABSENT'} "
          f"in the rendered text (enable_thinking={SP.ENABLE_THINKING})")

    host_flat = tok(text, add_special_tokens=False).input_ids

    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    enc = proc(text=[text], images=images, return_tensors="pt")
    host_full = enc["input_ids"][0].tolist()

    pad_id = tok.convert_tokens_to_ids(IMAGE_PAD)
    n_pad_flat = host_flat.count(pad_id)
    n_pad_full = host_full.count(pad_id)

    # ── B: does the pad run match the grid arithmetic? ────────────────────
    merge = getattr(proc.image_processor, "merge_size", 2)
    patch = getattr(proc.image_processor, "patch_size", 16)
    grid = enc.get("image_grid_thw")
    expected = None
    if grid is not None:
        g = grid.tolist()
        expected = sum(t * h * w for t, h, w in g) // (merge * merge)
        print(f"[INFO] image_grid_thw {g} (patch {patch}, merge {merge}) "
              f"-> {expected:,} vision token(s) for {images[0].size[0]}x{images[0].size[1]} px")
    ok_b = expected is not None and n_pad_full == expected and n_pad_flat == len(paths)
    print(f"[{'PASS' if ok_b else 'FAIL'}] B expansion: {n_pad_flat} pad(s) unexpanded, "
          f"{n_pad_full:,} expanded"
          + ("" if expected is None else f", arithmetic says {expected:,}"))
    print(f"[INFO] full prompt {len(host_full):,} tokens "
          f"({len(host_full) - n_pad_full + len(paths):,} text + "
          f"{n_pad_full:,} vision)")

    if not args.vllm_url:
        print("\n[INFO] no --vllm-url: comparison A (the one §3.4 asks for) NOT run. "
              "This is the host half only.")
        return 0 if ok_b else 1

    # ── A: the same conversation through vLLM's own tokenizer ─────────────
    model = served_model_name(args.vllm_url, args.timeout)
    print(f"\n[INFO] server {args.vllm_url}, model {model!r}")
    vllm_ids, note = vllm_tokenize(args.vllm_url, model, messages, args.timeout)
    n_pad_vllm = vllm_ids.count(pad_id)
    expanded = n_pad_vllm > len(paths)
    print(f"[INFO] /tokenize returned {len(vllm_ids):,} tokens ({note}), "
          f"{n_pad_vllm:,} image pad(s) -- "
          f"{'EXPANDED' if expanded else 'placeholders unexpanded'}")

    host_ids = host_full if expanded else host_flat
    label = "host_full" if expanded else "host_flat"
    ok_a = host_ids == vllm_ids
    print(f"[{'PASS' if ok_a else 'FAIL'}] A template: {label} vs vllm "
          f"({len(host_ids):,} vs {len(vllm_ids):,} tokens)")
    if not ok_a:
        report_divergence(tok, host_ids, vllm_ids, (label, "vllm"))
        if args.save_ids:
            args.save_ids.write_text(json.dumps(
                {"host": host_ids, "vllm": vllm_ids, "expanded": expanded},
                indent=1), encoding="utf-8")
            print(f"  ids -> {args.save_ids}")

        other = "openai" if args.content_format == "string" else "string"
        if ids_under(proc, messages, other, n_pad_vllm) == vllm_ids:
            print(f"\n  CAUSE: the server is rendering in the {other!r} content "
                  f"format, not {args.content_format!r} -- that reconstruction "
                  f"reproduces its ids exactly.")
            print(f"  FIX:   pass --content-format {other} (train against what "
                  f"the server sends), or serve with "
                  f"--chat-template-content-format {args.content_format} "
                  f"(CHANGES the prompt every run in this repo has been sent, "
                  f"so it invalidates the baseline).")
        elif thinking_prefix in text:
            has_think = tok(thinking_prefix, add_special_tokens=False).input_ids
            if has_think and has_think[0] not in vllm_ids:
                print("\n  CAUSE: the thinking prefix is in the host render and "
                      "not the server's -- chat_template_kwargs was ignored.")
        return 1

    print("\n[PASS] §3.4: the training prompt and the served prompt are the "
          "same token sequence.")
    return 0 if ok_b else 1


if __name__ == "__main__":
    sys.exit(main())
