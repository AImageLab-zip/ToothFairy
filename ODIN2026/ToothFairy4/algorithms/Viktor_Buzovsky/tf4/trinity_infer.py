"""Memory-safe inference wrapper around the Trinity-Segment ToothFairyUNet.

The upstream `utils/inference.py` allocates the full probability volume
(`num_classes x D x H x W` float32 = 312 bytes/voxel), which is 7.2 GB for a
median P-set volume and ~81 GB for a large A-set volume -- far over the 32 GB
DRAM cap on Grand Challenge. This wrapper keeps only a running best-probability
and best-label pair (3 bytes/voxel) and fuses overlapping windows by max
probability instead of averaging.

The model definition is imported from third_party/trinity/, a verbatim copy of
the relevant Trinity-Segment modules.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
TRINITY_SRC = Path(os.environ.get("TRINITY_SRC", _REPO / "third_party" / "trinity"))
MODEL_DIR = Path(os.environ.get("TF4_MODEL_DIR", _REPO / "algorithm" / "model"))
DEFAULT_CKPT = MODEL_DIR / "trinity_weights_fp16.pth"

# Trinity's modules use flat imports (`from model import ...`, `from utils...`),
# so its own directory has to be importable as a top-level package root. Inside
# the submission container the same modules are vendored next to this file, so
# this path simply will not exist and the local copies win.
if TRINITY_SRC.exists() and str(TRINITY_SRC) not in sys.path:
    sys.path.insert(0, str(TRINITY_SRC))

from model import ToothFairyUNet  # noqa: E402
from utils.labels_remap import LABEL_MAPPING  # noqa: E402

NUM_CLASSES = len(set(LABEL_MAPPING.values()))  # 78
PATCH = (120, 200, 200)
TARGET_SPACING = 0.3  # the model was trained on 0.3 mm isotropic TF3 data


def _inverse_lut() -> np.ndarray:
    """trained id (0..77) -> original ToothFairy3 label id (0..148)."""
    lut = np.zeros(max(LABEL_MAPPING.values()) + 1, dtype=np.uint8)
    for orig, trained in LABEL_MAPPING.items():
        lut[trained] = orig
    return lut


def load_model(ckpt_path: Path | str = DEFAULT_CKPT, device: str = "cuda") -> torch.nn.Module:
    model = ToothFairyUNet(input_channels=1, num_classes=NUM_CLASSES)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def load_norm_stats(path: Path | str = MODEL_DIR / "norm_stats.json") -> dict:
    with open(path) as f:
        return json.load(f)


def _patch_starts(extent: int, tile: int, stride: int) -> list[int]:
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    norm_stats: dict,
    device: str = "cuda",
    overlap: float = 0.5,
    amp: bool = True,
    progress: bool = False,
    patch_size: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Sliding-window prediction returning original TF3 label ids as uint8.

    Peak host memory is 3 bytes/voxel (float16 best-prob + uint8 best-label)
    plus the input volume, instead of 312 bytes/voxel.

    `patch_size` exists only as an out-of-memory escape hatch. cuDNN's workspace
    scales with the patch volume, so a smaller window is the cheapest way to fit
    a card that picked an im2col convolution algorithm. The network is fully
    convolutional, so this changes the context each voxel sees but not the
    weights; keep any override divisible by 8, which is what the default
    (120, 200, 200) is divisible by.
    """
    patch = tuple(patch_size or PATCH)
    img = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    img = torch.clamp(img, norm_stats["clip_min"], norm_stats["clip_max"])
    img = (img - norm_stats["mean"]) / norm_stats["std"]

    orig_shape = img.shape
    pad = [max(0, patch[i] - orig_shape[i]) for i in range(3)]
    if any(pad):
        # F.pad takes the last dimension first.
        img = F.pad(img, (0, pad[2], 0, pad[1], 0, pad[0]), mode="constant", value=0)
    shape = img.shape

    best_prob = np.zeros(shape, dtype=np.float16)
    best_lab = np.zeros(shape, dtype=np.uint8)

    stride = [max(1, int(t * (1 - overlap))) for t in patch]
    zs = _patch_starts(shape[0], patch[0], stride[0])
    ys = _patch_starts(shape[1], patch[1], stride[1])
    xs = _patch_starts(shape[2], patch[2], stride[2])
    total = len(zs) * len(ys) * len(xs)

    def forward(window):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            out = model(window)
        return out[0] if isinstance(out, (list, tuple)) else out

    done = 0
    for z in zs:
        for y in ys:
            for x in xs:
                window = img[z:z + patch[0], y:y + patch[1], x:x + patch[2]]
                window = window.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                try:
                    out = forward(window)
                except torch.OutOfMemoryError:
                    # Not a real shortage -- fragmentation. cuDNN picks a conv
                    # algorithm by how much memory is FREE, so on a 32 GB card it
                    # takes a fast one with a ~7.7 GB workspace and the allocator
                    # keeps that block cached for the next patch. On a 14.7 GB T4
                    # the cache wins the space and the next workspace request
                    # cannot find a contiguous block: "7.54 GiB is reserved by
                    # PyTorch but unallocated". Returning the cache to the driver
                    # makes the retry succeed. This failed EVERY case on Grand
                    # Challenge while never once reproducing on the 5090.
                    torch.cuda.empty_cache()
                    out = forward(window)
                prob, lab = torch.softmax(out.float(), dim=1).squeeze(0).max(dim=0)
                prob = prob.half().cpu().numpy()
                lab = lab.to(torch.uint8).cpu().numpy()

                sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(x, x + patch[2]))
                better = prob > best_prob[sl]
                best_prob[sl] = np.where(better, prob, best_prob[sl])
                best_lab[sl] = np.where(better, lab, best_lab[sl])

                done += 1
                if progress:
                    print(f"  patch {done}/{total}", flush=True)

    del best_prob
    best_lab = best_lab[: orig_shape[0], : orig_shape[1], : orig_shape[2]]
    return _inverse_lut()[best_lab]


def resample_to_target(
    data: np.ndarray, spacing: tuple[float, float, float], target: float = TARGET_SPACING
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Trilinear resample to `target` mm isotropic. Set A ships 0.15-0.2 mm data,
    which the model has never seen at native resolution."""
    factors = [s / target for s in spacing]
    if all(abs(f - 1.0) < 0.02 for f in factors):
        return data, spacing
    new_shape = [max(1, int(round(data.shape[i] * factors[i]))) for i in range(3)]
    t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    return t[0, 0].numpy(), (target, target, target)


# Canonical frame for everything downstream: axis 0 -> patient LEFT,
# axis 1 -> ANTERIOR, axis 2 -> SUPERIOR. This is what TF4 volumes declare, and
# what the hidden test set will declare, so it is the frame we trust.
CANONICAL_AXCODES = ("L", "A", "S")

# The model was trained on the raw ToothFairy3 `imagesTr` arrays. Those arrays do
# NOT agree with the TF4 arrays under their own affines: TF3 declares LPS and TF4
# declares LAS, yet `np.flip(tf4_array, axis=(1, 2))` reproduces the TF3 array
# EXACTLY (mean |diff| = 0.0 on P001/P002/F001). The two datasets' affines are
# mutually inconsistent, so the training convention has to be reached by an
# explicit array flip after header-based reorientation -- not by trusting headers.
# Getting this wrong silently costs ~0.90 tooth Dice (mirrored FDI numbering).
TRAIN_FRAME_FLIP = (1, 2)


def to_canonical_array(
    data: np.ndarray, affine: np.ndarray, zooms: tuple[float, float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Header-based reorientation into the canonical LAS frame.

    Takes a raw array plus its affine so the same path serves NIfTI (training and
    local evaluation) and SimpleITK `.mha` (what Grand Challenge actually ships).
    """
    if zooms is None:
        zooms = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    start = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(start, target)
    out = nib.orientations.apply_orientation(np.asarray(data, dtype=np.float32), transform)
    spacing = tuple(float(np.asarray(zooms)[int(transform[i, 0])]) for i in range(3))
    return np.ascontiguousarray(out), spacing


def to_canonical(nii) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Header-based reorientation into the canonical LAS frame, from a NIfTI."""
    return to_canonical_array(
        np.asarray(nii.dataobj, dtype=np.float32),
        nii.affine,
        tuple(float(z) for z in nii.header.get_zooms()[:3]),
    )


def to_training_frame(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(data, axis=TRAIN_FRAME_FLIP))


def from_training_frame(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(data, axis=TRAIN_FRAME_FLIP))


def segment_case(
    nifti_path: Path | str,
    model: torch.nn.Module,
    norm_stats: dict,
    device: str = "cuda",
    resample: bool = True,
    reorient: bool = True,
    progress: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Returns a mask in the canonical LAS frame at ~0.3 mm, with TF3 label ids.

    Downstream geometry assumes that frame: axis 0 -> patient LEFT,
    axis 1 -> anterior, axis 2 -> superior.
    """
    nii = nib.load(str(nifti_path))
    if reorient:
        data, spacing = to_canonical(nii)
    else:
        data = np.ascontiguousarray(np.asarray(nii.dataobj, dtype=np.float32))
        spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    if resample:
        data, spacing = resample_to_target(data, spacing)
    mask = predict_volume(model, to_training_frame(data), norm_stats, device=device, progress=progress)
    return from_training_frame(mask), spacing
