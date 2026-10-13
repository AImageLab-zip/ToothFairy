"""Geometric feature extractor: teeth_landmarks.json (CLIK landmarks, from
lmdet.api) -> a fixed-length named feature vector per subject.

Right/left does NOT need to be resolved geometrically -- FDI quadrant numbers
are authoritative (Q1/Q4 = patient's right, Q2/Q3 = patient's left), so
"right molar relation" always means tooth 16 vs 46, regardless of scan pose.

What DOES need resolving is a pose-free anatomical frame, since these scans
are not consistently oriented (established earlier: reorient='auto' is a
best-effort heuristic, not a guarantee). The frame is built from the
landmark cloud itself:
    up  = mean(upper centroids) - mean(lower centroids), normalized
    ant = mean(incisor centroids) - mean(molar centroids), orthogonalized
          against `up`
    lat = up x ant
This is the same construction validated earlier in-session: it separated
open-bite from deep-bite cases at AUC 0.766 using a single derived scalar.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad

UPPER_ANT = ("11", "21")          # central incisors
LOWER_ANT = ("31", "41")
UPPER_MOLARS = ("16", "26")
LOWER_MOLARS = ("36", "46")
UPPER_CANINES = ("13", "23")
LOWER_CANINES = ("33", "43")
UPPER_PREMOLARS_1 = ("14", "24")
LOWER_PREMOLARS_1 = ("34", "44")

RIGHT_UPPER_MOLAR, RIGHT_LOWER_MOLAR = "16", "46"
LEFT_UPPER_MOLAR, LEFT_LOWER_MOLAR = "26", "36"
RIGHT_UPPER_CANINE, RIGHT_LOWER_CANINE = "13", "43"
LEFT_UPPER_CANINE, LEFT_LOWER_CANINE = "23", "33"

ANT6_UPPER = ["13", "12", "11", "21", "22", "23"]
ANT6_LOWER = ["43", "42", "41", "31", "32", "33"]

ALL_UPPER = [str(q * 10 + p) for q in (1, 2) for p in range(1, 9)]
ALL_LOWER = [str(q * 10 + p) for q in (3, 4) for p in range(1, 9)]


def _pt(tooth: dict, *names: str) -> np.ndarray | None:
    for n in names:
        if n in tooth:
            return np.asarray(tooth[n], dtype=float)
    return None


def _functional_center(tooth_fdi: str, get) -> np.ndarray | None:
    """Landmark that defines the tooth's point on the line of occlusion,
    chosen by tooth type (FDI last digit) -- same convention as the Andrews
    6-Keys analysis: fovea/groove for posteriors, cusp tip for canines,
    incisal center for incisors. Falls back to Centroid."""
    pos = int(tooth_fdi[1])
    if pos in (6, 7, 8):
        pt = get(tooth_fdi, "Fovea center", "Groove center", "Buccal groove peak")
    elif pos in (4, 5):
        pt = get(tooth_fdi, "Groove center", "Fovea center")
    elif pos == 3:
        pt = get(tooth_fdi, "Cusp tip")
    else:
        pt = get(tooth_fdi, "Incisal center")
    return pt if pt is not None else get(tooth_fdi, "Centroid")


def build_frame(land: dict) -> np.ndarray | None:
    """Returns 3x3 rotation matrix [lat; ant; up] (rows), or None if the
    case lacks enough teeth to define it (too few landmarks to trust)."""
    cu = [land[t]["Centroid"] for t in ALL_UPPER if t in land and "Centroid" in land[t]]
    cl = [land[t]["Centroid"] for t in ALL_LOWER if t in land and "Centroid" in land[t]]
    if len(cu) < 6 or len(cl) < 6:
        return None
    up = np.mean(cu, axis=0) - np.mean(cl, axis=0)
    n = np.linalg.norm(up)
    if n < 1e-6:
        return None
    up /= n

    inc = [land[t]["Centroid"] for t in UPPER_ANT + LOWER_ANT if t in land and "Centroid" in land[t]]
    mol = [land[t]["Centroid"] for t in UPPER_MOLARS + LOWER_MOLARS if t in land and "Centroid" in land[t]]
    if len(inc) < 2 or len(mol) < 2:
        return None
    ant = np.mean(inc, axis=0) - np.mean(mol, axis=0)
    ant = ant - ant.dot(up) * up
    n = np.linalg.norm(ant)
    if n < 1e-6:
        return None
    ant /= n
    lat = np.cross(up, ant)
    return np.stack([lat, ant, up])


def _proj(land: dict, R: np.ndarray) -> dict:
    return {k: {n: R @ np.asarray(v, dtype=float) for n, v in t.items()} for k, t in land.items()}


def compute_features(land_raw: dict) -> dict[str, float] | None:
    """land_raw: the teeth_landmarks.json dict, {fdi_str: {landmark_name: [x,y,z]}}.
    Returns None if too sparse to build a reliable frame (few present teeth)."""
    R = build_frame(land_raw)
    if R is None:
        return None
    p = _proj(land_raw, R)
    f: dict[str, float] = {}

    def get(tooth, *names):
        t = p.get(tooth)
        return _pt(t, *names) if t else None

    # --- overjet / overbite: central incisor edge-to-edge relationship ---
    up_labial = [get(t, "Labial low", "Incisal center") for t in UPPER_ANT]
    lo_labial = [get(t, "Labial low", "Incisal center") for t in LOWER_ANT]
    up_edge = [get(t, "Incisal center") for t in UPPER_ANT]
    lo_edge = [get(t, "Incisal center") for t in LOWER_ANT]
    if all(x is not None for x in up_labial + lo_labial):
        # NOTE: empirically (not just by axis convention) lower-minus-upper is
        # the sign that makes "increased" cases score higher -- verified
        # against the 200 gold labels, see work/train_severity_regressors.py.
        # The naive up-minus-lower was backwards: markedly_increased cases
        # averaged -1.3mm and "zero" cases averaged +10mm under that sign.
        f["overjet_mm"] = float(np.mean([b[1] for b in lo_labial]) - np.mean([a[1] for a in up_labial]))
    if all(x is not None for x in up_edge + lo_edge):
        f["overbite_mm"] = float(np.mean([a[2] for a in up_edge]) - np.mean([b[2] for b in lo_edge]))

    # --- molar anteroposterior relation, per FDI side (Angle proxy).
    # Tried the "clinically correct" asymmetric Angle definition (upper MB
    # cusp vs lower fovea/groove) -- measured WORSE for mr_class/ml_class
    # (70.9%->68.9%, 83.2%->79.0%) than this symmetric cusp-vs-cusp version,
    # almost certainly because Fovea center is a noisier landmark for the
    # detector than a cusp tip. Verified empirically; keep this version.
    for name, ut, lt in (
        ("molar_ap_right_mm", RIGHT_UPPER_MOLAR, RIGHT_LOWER_MOLAR),
        ("molar_ap_left_mm", LEFT_UPPER_MOLAR, LEFT_LOWER_MOLAR),
    ):
        u = get(ut, "Mesiobuccal cusp", "Cusp tip", "Centroid")
        l = get(lt, "Mesiobuccal cusp", "Cusp tip", "Centroid")
        if u is not None and l is not None:
            f[name] = float(u[1] - l[1])

    # --- canine anteroposterior relation, per FDI side (cusp-tip proxy) ---
    for name, ut, lt in (
        ("canine_ap_right_mm", RIGHT_UPPER_CANINE, RIGHT_LOWER_CANINE),
        ("canine_ap_left_mm", LEFT_UPPER_CANINE, LEFT_LOWER_CANINE),
    ):
        u = get(ut, "Cusp tip", "Centroid")
        l = get(lt, "Cusp tip", "Centroid")
        if u is not None and l is not None:
            f[name] = float(u[1] - l[1])

    # --- transverse (crossbite proxy): upper vs lower antagonist lateral offset
    for name, ut, lt in (
        ("transverse_right_mm", RIGHT_UPPER_MOLAR, RIGHT_LOWER_MOLAR),
        ("transverse_left_mm", LEFT_UPPER_MOLAR, LEFT_LOWER_MOLAR),
        ("transverse_premolar_right_mm", "14", "44"),
        ("transverse_premolar_left_mm", "24", "34"),
    ):
        u = get(ut, "Centroid")
        l = get(lt, "Centroid")
        if u is not None and l is not None:
            f[name] = float(u[0] - l[0])

    # --- midline deviation: contact point between the two central incisors ---
    um = get("11", "Mesial contact") if get("11", "Mesial contact") is not None else None
    um2 = get("21", "Mesial contact")
    lm = get("31", "Mesial contact")
    lm2 = get("41", "Mesial contact")
    if um is not None and um2 is not None and lm is not None and lm2 is not None:
        upper_mid = (um + um2) / 2
        lower_mid = (lm + lm2) / 2
        f["midline_dev_mm"] = float(upper_mid[0] - lower_mid[0])

    # --- arch width ---
    for name, a, b in (
        ("intermolar_width_upper_mm", "16", "26"), ("intermolar_width_lower_mm", "36", "46"),
        ("interpremolar_width_upper_mm", "14", "24"), ("interpremolar_width_lower_mm", "34", "44"),
    ):
        pa, pb = get(a, "Centroid"), get(b, "Centroid")
        if pa is not None and pb is not None:
            f[name] = float(np.linalg.norm(pa[[0, 1]] - pb[[0, 1]]))

    # --- Little's Irregularity Index per arch (anterior-only crowding proxy) ---
    for name, teeth in (("little_index_upper_mm", ANT6_UPPER), ("little_index_lower_mm", ANT6_LOWER)):
        idx = _little_index(p, teeth)
        if idx is not None:
            f[name] = idx

    # --- whole-arch crowding: required (sum of tooth widths) minus available
    # (arc length of the fitted line of occlusion) -- catches crowding from
    # rotated/overlapped posterior teeth that Little's Index (anteriors only)
    # misses entirely.
    for name, teeth in (("arch_crowding_upper_mm", ALL_UPPER), ("arch_crowding_lower_mm", ALL_LOWER)):
        cr = _arch_crowding(p, teeth, get)
        if cr is not None:
            f[name] = cr

    # --- Curve of Spee / Wilson depth per arch ---
    for arch, teeth in (("upper", ["13", "14", "15", "16", "17", "23", "24", "25", "26", "27"]),
                        ("lower", ["43", "44", "45", "46", "47", "33", "34", "35", "36", "37"])):
        heights = []
        for t in teeth:
            pt = get(t, "Cusp tip", "Buccal cusp", "Mesiobuccal cusp")
            if pt is not None:
                heights.append(pt[2])
        if len(heights) >= 3:
            f[f"curve_of_spee_{arch}_mm"] = float(max(heights) - min(heights))

        wilson = []
        for t in teeth:
            b = get(t, "Buccal cusp", "Mesiobuccal cusp")
            l = get(t, "Lingual cusp", "Mesiolingual cusp")
            if b is not None and l is not None:
                wilson.append(abs(b[2] - l[2]))
        if wilson:
            f[f"curve_of_wilson_{arch}_mm"] = float(np.mean(wilson))

    # --- presence / dentition ---
    f["n_teeth_upper"] = float(sum(1 for t in ALL_UPPER if t in land_raw))
    f["n_teeth_lower"] = float(sum(1 for t in ALL_LOWER if t in land_raw))

    return f


def _little_index(p: dict, teeth: list[str]) -> float | None:
    """Sum of gaps between where adjacent anterior teeth's contact points
    should meet. Ordered along the lateral axis (monotonic for anteriors,
    unlike a full-arch angular sort which zigzags -- the bug from the
    original TSALD attempt earlier this session)."""
    pts = []
    for t in teeth:
        tt = p.get(t)
        if not tt or "Mesial contact" not in tt or "Distal contact" not in tt:
            return None
        pts.append((tt["Mesial contact"][0], t, "M", tt["Mesial contact"]))
        pts.append((tt["Distal contact"][0], t, "D", tt["Distal contact"]))
    pts.sort(key=lambda r: r[0])
    total = 0.0
    for i in range(len(pts) - 1):
        total += float(np.linalg.norm(np.asarray(pts[i][3]) - np.asarray(pts[i + 1][3])))
    # subtract the "should touch" contribution: consecutive same-tooth M/D
    # pairs are the tooth's own width, not irregularity -- only inter-tooth
    # gaps count, so estimate irregularity as the excess over sum of true
    # adjacent-tooth contact gaps (skip every other segment: intra-tooth)
    gaps = []
    for i in range(1, len(pts) - 1, 2):
        gaps.append(float(np.linalg.norm(np.asarray(pts[i][3]) - np.asarray(pts[i + 1][3]))))
    return float(sum(gaps)) if gaps else None


def _fit_arch_curve(centers: list[tuple[float, float]]):
    """4th-order poly ant = f(lat) through functional centers (Andrews-style
    line of occlusion), falling back to a parabola with fewer points."""
    if len(centers) < 3:
        return None
    lat = [c[0] for c in centers]
    ant = [c[1] for c in centers]
    deg = 4 if len(centers) >= 5 else 2
    try:
        return np.poly1d(np.polyfit(lat, ant, deg))
    except np.linalg.LinAlgError:
        return None


def _arch_crowding(p: dict, teeth: list[str], get) -> float | None:
    """Andrews-style arch-length discrepancy: required space (sum of
    mesiodistal tooth widths) minus available space (arc length of the
    fitted line of occlusion between the outermost present teeth).
    Positive = crowding, negative = spacing. Whole-arch, unlike
    _little_index (anteriors only) -- catches posterior rotation/overlap."""
    total_width = 0.0
    n_widths = 0
    centers = []
    for t in teeth:
        tt = p.get(t)
        if tt and "Mesial contact" in tt and "Distal contact" in tt:
            total_width += float(np.linalg.norm(
                np.asarray(tt["Mesial contact"]) - np.asarray(tt["Distal contact"])))
            n_widths += 1
        fc = _functional_center(t, get)
        if fc is not None:
            centers.append((float(fc[0]), float(fc[1])))
    if n_widths < 3 or len(centers) < 3:
        return None

    curve = _fit_arch_curve(centers)
    if curve is None:
        return None
    deriv = curve.deriv()
    lat_min = min(c[0] for c in centers)
    lat_max = max(c[0] for c in centers)
    available, _ = quad(lambda y: np.sqrt(1 + deriv(y) ** 2), lat_min, lat_max)
    return float(total_width - available)
