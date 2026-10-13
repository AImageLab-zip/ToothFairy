#!/usr/bin/env python3
"""
rebuild_captions.py

Regenerate the caption sidecar JSONs for ALREADY-GENERATED images, without
re-rendering a single PNG.

Why this exists
---------------
Captions are written at generation time by the four image-creation scripts
(create_panoramic / create_3d_renders / create_tooth_detail /
create_sinus_detail), which is the right default -- but it means that
editing caption WORDING otherwise forces a full re-run of image generation
(nibabel volume loads, marching cubes, 32 composites per case) just to
rewrite a few kilobytes of text.

Nothing in the caption text actually needs the volume or the mask: every
builder is a pure function of the case's facts.json.

So this script imports those same builders -- it does NOT restate any
caption text of its own, so there is still exactly ONE definition of every
caption -- and rewrites the four sidecars from the existing PNGs.

Only images that EXIST get a caption entry, same rule the generators use --
a caption must never point at a file that was skipped.

Usage:
    # rewrite captions for a whole run, in place
    python code/pipeline/preprocess/rebuild_captions.py \
        --images-dir outputs/aksssr_v4_validate/images \
        --facts-dir dataset/validate/facts

    # check what would change first
    python code/pipeline/preprocess/rebuild_captions.py ... --dry-run

    # one case only
    python code/pipeline/preprocess/rebuild_captions.py ... --cases A008

    # strip EVERY case finding out of the captions (leakage-free arms), and
    # the same command with --dry-run as the verifier that they're already gone
    python code/pipeline/preprocess/rebuild_captions.py \
        --images-dir outputs/report_generation_without_schema_validate/images \
        --static-captions

Then rebuild the QA pairs so the new captions reach the VLM (step 5 of
scripts/aksssr_pipeline.sh -- everything before it can be skipped):

    python code/pipeline/preprocess/build_vqa_pairs.py \
        --schema schema/schema.json \
        --images-dir outputs/aksssr_v4_validate/images \
        --project-dir $HOME/project_ToothFairy4 \
        --out outputs/aksssr_v4_validate/qa_pairs.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


# The single source of truth for filename -> schema image key. Reused rather
# than re-derived so this script and the QA builder can never disagree about
# which images a case has.

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

from build_vqa_pairs import discover_images

from create_panoramic import build_panoramic_caption
from create_3d_renders import (build_3d_view_addenda, drawn_structures,
                               mask_label_volumes, view_caption)
from create_tooth_detail import build_tooth_caption
from create_sinus_detail import build_sinus_caption

_TOOTH_KEY = re.compile(r"^tooth_(?P<fdi>\d{2})_composite$")

_3D_VIEWS = ("left", "frontal", "right")


def build_case_sidecars(image_keys, facts, volumes=None):
    """
    Build every caption sidecar for one case.

    `image_keys` is the set of schema image keys that actually exist on disk
    for this case (build_vqa_pairs.discover_images' inner dict). `facts` is
    the parsed facts.json ({} is allowed -- captions then carry the static
    description only).

    `volumes` is create_3d_renders.mask_label_volumes' {label: mm^3} for the
    case, and it is what the 3D color key and every "included in the
    acquisition" claim are measured from. THIS IS THE ONE THING A CAPTION
    NEEDS BEYOND THE FACTS FILE -- pass it whenever a mask is available.
    Without it the 3D captions fall back to naming all four structures and to
    the facts file's own present_label_ids/fov, which is what this script did
    before masks were read at all.

    Returns {sidecar_suffix: {image_key: caption}}, omitting any sidecar
    whose images are all missing (the generators don't write empty sidecars
    either).
    """
    sidecars = {}

    # ── panoramic ───────────────────────────────────────────────────────
    if "panoramic" in image_keys:
        sidecars["_panoramic_caption.json"] = {
            "panoramic": build_panoramic_caption(facts)
        }

    # ── 3D renders ──────────────────────────────────────────────────────
    addenda = build_3d_view_addenda(facts, volumes)
    drawn = drawn_structures(volumes)
    view_captions = {}
    for view in _3D_VIEWS:
        if f"3d_{view}" not in image_keys:
            continue
        caption = view_caption(view, drawn)
        if addenda.get(view):
            caption = f"{caption} {addenda[view]}"
        view_captions[f"3d_{view}"] = caption
    if view_captions:
        sidecars["_3d_captions.json"] = view_captions

    # ── tooth composites ────────────────────────────────────────────────
    # No layout recovery needed: the 3x3 grid is the same for every tooth,
    # so the caption depends on nothing but the FDI and the facts.
    tooth_captions = {}
    for key in image_keys:
        m = _TOOTH_KEY.match(key)
        if not m:
            continue
        tooth_captions[key] = build_tooth_caption(int(m["fdi"]), facts)
    if tooth_captions:
        # Sorted by FDI so the file diffs cleanly against the generated one.
        sidecars["_tooth_captions.json"] = dict(
            sorted(tooth_captions.items(), key=lambda kv: int(_TOOTH_KEY.match(kv[0])["fdi"]))
        )

    # ── sinus ───────────────────────────────────────────────────────────
    # No nasal cavity entry: create_sinus_detail.py no longer generates that
    # image, so a stale PNG on disk must not resurrect its caption.
    sinus_captions = {}
    for side in ("right", "left"):
        if f"sinus_{side}_detail" in image_keys:
            sinus_captions[f"sinus_{side}_detail"] = build_sinus_caption(side)
    if sinus_captions:
        sidecars["_sinus_captions.json"] = sinus_captions

    return sidecars


def load_mask_volumes(masks_dir, case_id):
    """
    {label: mm^3} for one case's mask, or None when there is no masks dir, no
    file for the case, or nibabel cannot read it. None means "no mask to go
    on" everywhere downstream, never "the mask was empty".
    """
    if not masks_dir:
        return None
    path = Path(masks_dir) / f"{case_id}.nii.gz"
    if not path.exists():
        return None
    try:
        import nibabel as nib
        import numpy as np  # noqa: F401  (mask_label_volumes needs numpy loaded)
        nii = nib.as_closest_canonical(nib.load(str(path)))
        return mask_label_volumes(nii.get_fdata().astype("int32"),
                                  nii.header.get_zooms())
    except Exception as exc:                                # pragma: no cover
        print(f"[{case_id}] [WARN] could not read mask {path}: {exc}")
        return None


def rebuild(images_dir, facts_dir, cases=None, dry_run=False,
            allow_missing_facts=False, static_captions=False, masks_dir=None):
    images_path = Path(images_dir)
    if not images_path.is_dir():
        sys.exit(f"[FAIL] images dir not found: {images_path}")

    if not masks_dir and not static_captions:
        print("[WARN] no --masks-dir: the 3D color key will name all four "
              "structures and the")
        print("[WARN] maxilla's coverage will come from the facts file rather "
              "than from the mask,")
        print("[WARN] which is what put 'maxilla (tan)' on mandible-only "
              "pictures. Pass it.")

    if static_captions:
        print("[INFO] ***** STATIC-CAPTION MODE *****")
        print("[INFO] No facts file is read for any case. Every sidecar is "
              "rewritten to the")
        print("[INFO] static image description alone -- every case finding "
              "(absent teeth, crowns,")
        print("[INFO] implants, bridges, canal proximity, FOV) is STRIPPED. "
              "This is for arms that")
        print("[INFO] must not be handed ground truth through their inputs. "
              "Never run it against")
        print("[INFO] an aksssr run's images dir.")

    images_by_case = discover_images(str(images_path))
    if cases:
        images_by_case = {k: v for k, v in images_by_case.items() if k in cases}
        missing = sorted(set(cases) - set(images_by_case))
        if missing:
            print(f"[WARN] no images found for requested case(s): {missing}")
    if not images_by_case:
        sys.exit(f"[FAIL] no generated images discovered under {images_path}")

    print(f"[INFO] Rebuilding captions for {len(images_by_case)} case(s) "
          f"in {images_path}" + (" (DRY RUN)" if dry_run else ""))

    n_written = n_unchanged = n_skipped = 0

    for case_id in sorted(images_by_case):
        image_keys = set(images_by_case[case_id])

        if static_captions:
            # Deliberately not even looking for the file: the whole point is
            # that no case finding can reach the caption, so there must be no
            # code path where a facts file that happens to exist gets read.
            facts = {}
        elif (facts_path := Path(facts_dir) / f"{case_id}.json").exists():
            facts = json.loads(facts_path.read_text())
        elif allow_missing_facts:
            print(f"[{case_id}] [WARN] no facts file ({facts_path}) -- writing "
                  "static-only captions")
            facts = {}
        else:
            # Silently writing static-only captions here would QUIETLY strip
            # every case finding out of the prompts, which looks like a
            # wording change and isn't -- so skip instead, unless asked.
            print(f"[{case_id}] [SKIP] no facts file: {facts_path} "
                  "(pass --allow-missing-facts to write static captions anyway)")
            n_skipped += 1
            continue

        # Even in --static-captions mode the mask is read: which structures
        # are DRAWN is a property of the picture, not a case finding, so
        # naming them leaks nothing -- the same reasoning that keeps the
        # wisdom-tooth color key in the static caption.
        sidecars = build_case_sidecars(image_keys, facts,
                                       load_mask_volumes(masks_dir, case_id))

        changed = []
        for suffix, captions in sidecars.items():
            out_path = images_path / f"{case_id}{suffix}"
            new_text = json.dumps(captions, indent=2)
            old_text = out_path.read_text() if out_path.exists() else None
            if old_text is not None and old_text.rstrip("\n") == new_text.rstrip("\n"):
                n_unchanged += 1
                continue
            changed.append((out_path, new_text, old_text is None))
            n_written += 1

        for out_path, new_text, is_new in changed:
            tag = "new" if is_new else "updated"
            if dry_run:
                print(f"[{case_id}] would write ({tag}): {out_path.name}")
            else:
                out_path.write_text(new_text)
                print(f"[{case_id}] {tag}: {out_path.name}")

        if not changed:
            print(f"[{case_id}] up to date ({len(sidecars)} sidecar(s))")

    print("\n[INFO] ========== Captions Rebuilt ==========")
    print(f"[INFO] Sidecars {'to write' if dry_run else 'written'}: {n_written}")
    print(f"[INFO] Sidecars already up to date: {n_unchanged}")
    if n_skipped:
        print(f"[INFO] Cases skipped (no facts file): {n_skipped}")
    if n_written and not dry_run:
        print("[INFO] Re-run build_vqa_pairs.py so the new captions reach "
              "qa_pairs.jsonl (see this file's docstring).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                    help="Directory holding the already-generated PNGs and "
                         "their caption sidecars (e.g. outputs/<run>/images)")
    ap.add_argument("--facts-dir", default=None,
                    help="Per-case facts JSONs ({case_id}.json), i.e. "
                         "dataset/<split>/facts. Required unless "
                         "--static-captions is given.")
    ap.add_argument("--masks-dir", default=None,
                    help="Per-case segmentation masks ({case_id}.nii.gz), i.e. "
                         "dataset/<split>/masks. The 3D color key names only "
                         "the structures actually in the mask, and the "
                         "maxilla's FOV line is measured from it rather than "
                         "taken from the facts file. Strongly recommended: "
                         "without it a mandible-only case is still captioned "
                         "'maxilla (tan)'.")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="Only rebuild these case IDs (default: every case "
                         "with images in --images-dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report which sidecars would change; write nothing")
    ap.add_argument("--allow-missing-facts", action="store_true",
                    help="Write static-only captions for cases with no facts "
                         "file instead of skipping them")
    ap.add_argument("--static-captions", action="store_true",
                    help="Rewrite EVERY sidecar to the static image description "
                         "alone, reading no facts file for any case -- every "
                         "case finding is stripped. Unlike --allow-missing-facts "
                         "this is unconditional, not a fallback for a missing "
                         "file. For leakage-free arms; never run it against an "
                         "aksssr run's images dir. Combine with --dry-run to use "
                         "it as a VERIFIER: on a dir generated facts-free it "
                         "must report every sidecar already up to date.")

    args = ap.parse_args()
    if not args.facts_dir and not args.static_captions:
        ap.error("--facts-dir is required unless --static-captions is given")
    rebuild(args.images_dir, args.facts_dir, cases=args.cases,
            dry_run=args.dry_run, allow_missing_facts=args.allow_missing_facts,
            static_captions=args.static_captions, masks_dir=args.masks_dir)
