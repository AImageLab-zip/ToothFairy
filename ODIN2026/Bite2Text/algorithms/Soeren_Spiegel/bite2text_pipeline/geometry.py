"""Shared case geometry for Bite2Text: mesh reading + the per-case anatomical frame.

One SHARED rigid frame per case (never per-jaw — the bite registration is the
signal). Axes are anatomically pinned:
  occ    : occlusal normal, points lower -> upper (anatomically UP)
  a_dir  : anterior (narrow end of the crown horseshoe)
  l_left : patient-LEFT = occ x a_dir (right-handed, deterministic laterality)
z0 is the occlusal-plane offset along occ.
"""
from dataclasses import dataclass

import numpy as np

STL_DT = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def _face_normals(verts):
    """(3T,3) per-triangle vertex triplets -> (3T,3) flat-shaded unit normals."""
    tri = verts.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nn > 1e-9, n / np.maximum(nn, 1e-9), 0.0)
    return np.repeat(n, 3, axis=0)


def read_stl(data: bytes):
    """Binary STL -> (verts (N,3), normals (N,3)); verts are per-triangle triplets
    sharing the triangle normal (flat shading)."""
    ntri = np.frombuffer(data, "<u4", 1, 80)[0]
    rec = np.frombuffer(data, STL_DT, ntri, 84)
    verts = rec["v"].reshape(-1, 3).astype(np.float64)
    normals = np.repeat(rec["n"].astype(np.float64), 3, axis=0)
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(nn > 1e-9, normals / np.maximum(nn, 1e-9), 0.0)
    return verts, normals


def read_stl_ascii(data: bytes):
    """ASCII STL -> same contract as read_stl. Normals are recomputed from the winding
    rather than read from the `facet normal` line: exporters routinely write zeros there,
    and the geometric normal is what read_stl's binary path effectively carries anyway."""
    verts = []
    for line in data.decode("utf-8", "ignore").splitlines():
        f = line.split()
        if len(f) == 4 and f[0] == "vertex":
            verts.append((float(f[1]), float(f[2]), float(f[3])))
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if len(v) < 3 or len(v) % 3:
        raise ValueError(f"ASCII STL: {len(v)} vertices is not a whole number of triangles")
    return v, _face_normals(v)


def read_obj(data: bytes):
    """Wavefront OBJ -> same contract as read_stl (per-triangle vertex triplets + flat
    normals), so render_views() cannot tell the two formats apart.

    Only `v` and `f` are read. Face entries may be `i`, `i/t`, `i//n` or `i/t/n`; indices
    are 1-based and may be negative (relative to the end of the vertex list so far).
    Polygons are fan-triangulated. `vn` is ignored on purpose -- see read_stl_ascii.
    """
    vs = []
    faces = []
    for line in data.decode("utf-8", "ignore").splitlines():
        if not line or line[0] not in "vf":
            continue
        f = line.split()
        if not f:
            continue
        if f[0] == "v" and len(f) >= 4:
            vs.append((float(f[1]), float(f[2]), float(f[3])))
        elif f[0] == "f" and len(f) >= 4:
            idx = []
            for tok in f[1:]:
                i = int(tok.split("/", 1)[0])
                idx.append(i - 1 if i > 0 else len(vs) + i)   # negative = relative
            for k in range(1, len(idx) - 1):                  # fan-triangulate
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not vs or not faces:
        raise ValueError(f"OBJ: parsed {len(vs)} vertices / {len(faces)} faces")
    v_all = np.asarray(vs, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if tri.min() < 0 or tri.max() >= len(v_all):
        raise ValueError("OBJ: face index out of range")
    verts = v_all[tri.reshape(-1)]
    return verts, _face_normals(verts)


def read_mesh(data: bytes, name: str = ""):
    """Format-sniffing front door -> (verts, normals). Grand Challenge's Bite2Text template
    globs BOTH `*.obj` and `*.stl` for the IOS sockets (their inference.py
    load_single_file_compat), and the training corpus ships binary STL, so the container has
    to read whichever turns up rather than assume one.

    Sniffing is by CONTENT, not extension: the binary-STL header length check
    (84 + 50*ntri == filesize) is exact, and a binary STL may still begin with b"solid",
    which is why the usual "starts with solid -> ASCII" test is not used. `name` only breaks
    the tie for OBJ, which has no magic number.
    """
    if name.lower().endswith(".obj"):
        return read_obj(data)
    if len(data) >= 84:
        ntri = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if 84 + 50 * ntri == len(data):
            return read_stl(data)
    head = data[:8192]
    if b"facet" in head or head.lstrip()[:5] == b"solid":
        return read_stl_ascii(data)
    # OBJ has no magic number, so look for an element keyword on the first line that is
    # neither blank nor a comment -- exporters routinely lead with a comment banner, which
    # is why "does it start with 'v '" is not enough.
    for line in head.decode("utf-8", "ignore").splitlines()[:200]:
        tok = line.split(None, 1)
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] in ("v", "vn", "vt", "f", "g", "o", "s", "mtllib", "usemtl"):
            return read_obj(data)
        break
    try:                        # last resort: trust the binary layout
        return read_stl(data)
    except Exception as e:
        raise ValueError(f"unrecognised mesh format ({len(data)} bytes, name={name!r})") from e


@dataclass
class CaseFrame:
    occ: np.ndarray      # up (lower -> upper)
    a_dir: np.ndarray    # anterior
    l_left: np.ndarray   # patient left
    z0: float            # occlusal plane offset along occ
    a_c: float           # crown-horseshoe center, anterior coord
    l_c: float           # crown-horseshoe center, left coord

    def coords(self, v):
        """(N,3) verts -> (a, l, z) anatomical coordinates."""
        return v @ self.a_dir, v @ self.l_left, v @ self.occ


def case_frame(vu: np.ndarray, vl: np.ndarray, iters: int = 3) -> CaseFrame:
    occ = vu.mean(0) - vl.mean(0)
    occ /= np.linalg.norm(occ)
    crowns = None
    for _ in range(iters):  # crown-plane refinement (palate/flange tilt removal)
        zu, zl = vu @ occ, vl @ occ
        z0 = (np.percentile(zu, 1) + np.percentile(zl, 99)) / 2.0
        cu = vu[(zu - z0 > -2) & (zu - z0 < 8)]
        cl = vl[(z0 - zl > -2) & (z0 - zl < 8)]
        crowns = np.vstack([cu, cl])
        _, _, Vt = np.linalg.svd(crowns - crowns.mean(0), full_matrices=False)
        n = Vt[2]
        occ = n if n @ occ > 0 else -n
    zu, zl = vu @ occ, vl @ occ
    z0 = float((np.percentile(zu, 1) + np.percentile(zl, 99)) / 2.0)

    cov = np.cov((crowns - crowns.mean(0)).T)
    w, V = np.linalg.eigh(cov)
    e2 = V[:, np.argsort(w)[1]]
    e2 = e2 - (e2 @ occ) * occ
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, occ)
    ap, lr = crowns @ e2, crowns @ e1
    lo, hi = np.percentile(ap, [15, 85])
    ant = -1.0 if np.ptp(lr[ap <= lo]) < np.ptp(lr[ap >= hi]) else 1.0
    a_dir = e2 * ant
    l_left = np.cross(occ, a_dir)

    ca, cl_ = crowns @ a_dir, crowns @ l_left
    return CaseFrame(occ=occ, a_dir=a_dir, l_left=l_left, z0=z0,
                     a_c=float((ca.min() + ca.max()) / 2),
                     l_c=float((cl_.min() + cl_.max()) / 2))
