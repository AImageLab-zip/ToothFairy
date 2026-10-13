"""ToothFairy4 submission entrypoint.

Reads a CBCT from the `cbct-image` socket, segments it with the Trinity model,
derives a structured findings record, and renders the report deterministically.

Failure policy: try hard, then fail loudly. Inference is attempted three ways
(GPU full window, GPU reduced window, CPU) and each optional finding is isolated
so one bad slot cannot take down the report. But if all three attempts fail we
RAISE rather than emit the corpus-prior report.

That fallback used to be here, and it is precisely what hid a total failure: a
CUDA OOM killed every case on Grand Challenge, the prior report went out for
every patient, and the leaderboard score looked like a working algorithm. Two
submissions were spent that way. A constant report is worth almost nothing for
ranking but is indistinguishable from success, which is the worst combination
available; a crash is worse for one case and much better for the project.
"""

import gc
import glob
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
APP_PATH = Path("/opt/app")

sys.path.insert(0, str(APP_PATH))
sys.path.insert(0, str(APP_PATH / "vendor"))

from vendor.mask_features import derive_findings  # noqa: E402
from vendor.render import render  # noqa: E402
from vendor.schema import Findings  # noqa: E402
from vendor.trinity_infer import (  # noqa: E402
    load_model, predict_volume, resample_to_target, to_canonical_array,
    to_training_frame, from_training_frame,
)

WEIGHTS = MODEL_PATH / "trinity_weights_fp16.pth"
NORM_STATS = MODEL_PATH / "norm_stats.json"


def sitk_to_affine(image: SimpleITK.Image) -> np.ndarray:
    """Build a nibabel-style affine for the numpy array SimpleITK hands back.

    Two conversions are needed, and getting either wrong silently mirrors the
    anatomy:
      1. GetArrayFromImage returns (k, j, i) -- reversed relative to the image's
         own index order -- so the affine columns are reversed to match.
      2. ITK works in LPS physical space, nibabel affines are RAS, so the first
         two physical axes flip sign.
    """
    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), dtype=float)
    affine = np.eye(4)
    for out_axis, idx_axis in enumerate((2, 1, 0)):  # array axes k, j, i
        affine[:3, out_axis] = direction[:, idx_axis] * spacing[idx_axis]
    affine[:3, 3] = np.array(image.GetOrigin(), dtype=float)
    affine[:2, :] *= -1.0  # LPS -> RAS
    return affine


def load_image_file(location: Path) -> SimpleITK.Image:
    files = sorted(glob.glob(str(location / "*.mha")))
    if not files:
        raise RuntimeError(f"No .mha CBCT image found in {location}")
    return SimpleITK.ReadImage(files[0])


def build_report(image: SimpleITK.Image, force_cpu: bool = False,
                 patch_size=None) -> str:
    array = SimpleITK.GetArrayFromImage(image).astype(np.float32)
    affine = sitk_to_affine(image)

    data, spacing = to_canonical_array(array, affine)
    data, spacing = resample_to_target(data, spacing)

    dev = "cpu" if force_cpu else device()
    model = load_model(WEIGHTS, device=dev)
    with open(NORM_STATS) as fh:
        norm_stats = json.load(fh)

    mask = predict_volume(model, to_training_frame(data), norm_stats, device=dev,
                          patch_size=patch_size)
    mask = from_training_frame(mask)

    # `data` is the resampled volume in the same canonical frame as `mask`, which
    # is what the intensity-derived slots (endodontic filling material) read.
    findings = derive_findings(mask, spacing=float(np.mean(spacing)), image=data)
    return render(findings, use_priors=True)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run() -> int:
    # Three attempts, cheapest first. The GPU path died on EVERY Grand Challenge
    # case with a CUDA OOM that never reproduced on the 32 GB development card,
    # and the prior-only fallback hid it perfectly: the reports looked plausible,
    # so two submissions silently scored the same 64 words for every patient.
    # Never again on one attempt.
    #
    #   1. GPU, full (120, 200, 200) window -- what we want.
    #   2. GPU, smaller window. The workspace scales with the patch volume, so
    #      (96, 160, 160) needs 11.87 GiB instead of 23.17 GiB. Costs tooth Dice
    #      0.916 -> 0.901, measured. Still seconds, unlike the CPU.
    #   3. CPU. ~680 s for the largest case against a 900 s budget, so it is a
    #      last resort, but it does produce a real report.
    attempts = [
        ("GPU, full window", dict()),
        ("GPU, reduced window", dict(patch_size=(96, 160, 160))),
        ("CPU", dict(force_cpu=True)),
    ]

    # On Turing the full window provably cannot fit, so do not waste an attempt
    # on it. The decoder concatenates a 64-channel upsample with a 32-channel
    # skip, and cuDNN on sm_75 has no better algorithm than im2col for the
    # resulting 96-channel convolution at full patch resolution:
    #     27 taps x 96 ch x 120x200x200 x 2 bytes = 23.17 GiB
    # which is more than any Turing card has. Measured on a T4: "Tried to
    # allocate 23.18 GiB". Ampere and later pick a cheaper algorithm and run the
    # full window in under 5 GiB, so this only skips the tier where it is futile.
    # Disabling cuDNN does not help -- PyTorch's own conv3d needs the same 23.18
    # GiB, measured directly.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        name = torch.cuda.get_device_name()
        print(f"{name} is pre-Ampere; the full window needs 23.2 GiB there, "
              f"so starting at the reduced window.", flush=True)
        attempts = attempts[1:]

    image = load_image_file(INPUT_PATH / "images" / "cbct")
    print(f"CBCT size={image.GetSize()} spacing={image.GetSpacing()}", flush=True)

    report = ""
    for name, kwargs in attempts:
        try:
            print(f"Attempting: {name}", flush=True)
            report = build_report(image, **kwargs)
            if report.strip():
                print(f"Succeeded on: {name}", flush=True)
                break
        except Exception:
            traceback.print_exc()
        finally:
            # Hand the failed attempt's memory back before the next one. Without
            # this the next tier inherits the previous model and its activations:
            # a T4 log showed 2.73 GiB in use at the first failure and 4.08 GiB
            # at the second, leaving the reduced window ~10.6 GiB free against
            # the 11.87 GiB it needs. It succeeded by a hair. Each tier should
            # start on a clean card.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not report or not report.strip():
        # DELIBERATELY FATAL. There used to be a prior-only fallback here, and it
        # is what made this pipeline's total failure invisible: two submissions
        # emitted the same 64 words for every patient and still produced a
        # plausible-looking leaderboard score, so nothing looked wrong for days.
        #
        # A constant report is worth close to nothing for ranking but looks
        # exactly like a working algorithm, which is the worst of both. Crashing
        # is worse per-case and far better per-project: the evaluator raises on
        # any job whose status is not "Succeeded", so this costs the whole
        # submission -- and that is the point. A visible failure gets fixed.
        #
        # This is only safe because three real attempts run first. Set
        # PRIOR_FALLBACK=1 to restore the old behaviour without touching code.
        if os.environ.get("PRIOR_FALLBACK") == "1":
            print("All attempts failed; PRIOR_FALLBACK=1, emitting prior report.", flush=True)
            report = render(Findings(), use_priors=True)
        else:
            raise RuntimeError(
                "All inference attempts failed (GPU full window, GPU reduced window, CPU). "
                "Refusing to emit the prior-only report: it scores like a working "
                "algorithm and hides the failure. See the tracebacks above."
            )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / "diagnostic-imaging-report.json", "w", encoding="utf-8") as fh:
        json.dump({"report": report}, fh, indent=4)
    print(f"--- report ({len(report.split())} words) ---\n{report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
