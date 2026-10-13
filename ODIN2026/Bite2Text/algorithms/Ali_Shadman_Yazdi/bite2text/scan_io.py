"""Loading intraoral scans, trusting file content over file name.

Grand Challenge names the scan sockets ``*.obj`` while the bytes are binary STL,
and trimesh's OBJ reader silently returns an empty mesh for STL input. So the
format is sniffed from the content first and only then handed to trimesh.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

import trimesh

MESH_TYPES = ("stl", "obj", "ply", "off")


def sniff_mesh_type(path) -> Optional[str]:
    """Mesh format from the file's bytes; ``None`` if unrecognised."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(1024)
        # Binary STL has no magic number, but its size is exactly
        # 80-byte header + uint32 triangle count + 50 bytes per triangle.
        if size >= 84:
            fh.seek(80)
            count = struct.unpack("<I", fh.read(4))[0]
            if size == 84 + 50 * count:
                return "stl"
    if head[:3] == b"ply":
        return "ply"
    if head[:3] == b"OFF":
        return "off"
    stripped = head.lstrip().lower()
    if stripped.startswith(b"solid"):
        return "stl"
    if stripped.startswith(b"v ") or b"\nv " in head:
        return "obj"
    return None


def probe_mesh_type(path) -> Optional[str]:
    """Brute force: the first parser that yields a non-empty mesh."""
    for file_type in MESH_TYPES:
        try:
            with open(path, "rb") as fh:
                mesh = trimesh.load(fh, file_type=file_type, force="mesh", process=False)
        except Exception:
            continue
        if getattr(mesh, "vertices", None) is not None and len(mesh.vertices) >= 4:
            return file_type
    return None


def load_scan(path) -> trimesh.Trimesh:
    """Load a single-jaw scan (.stl / .obj / .ply / .off) as a processed Trimesh."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scan not found: {path}")

    file_type = sniff_mesh_type(path) or probe_mesh_type(path)
    if file_type is None:
        raise ValueError(f"{path} ({path.stat().st_size} bytes) is not a mesh any of "
                         f"{list(MESH_TYPES)} can read")

    with open(path, "rb") as fh:
        mesh = trimesh.load(fh, file_type=file_type, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.vertices) < 4 or len(mesh.faces) == 0:
        raise ValueError(f"{path}: empty or corrupt scan ({len(mesh.vertices)} vertices)")
    return mesh
