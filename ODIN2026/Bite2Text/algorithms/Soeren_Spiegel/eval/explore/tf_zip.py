"""Streaming access to the ToothFairy4 dataset zip.

The dataset ships as a single ~44 GB zip:
    <CASE>/cbct/volume.nii.gz
    <CASE>/reports_it/<rid>.txt
    <CASE>/reports_en/<rid>.txt

We never extract it. This module reads report text directly from the zip and
parses NIfTI-1 headers by decompressing only the first few KB of each gzip
stream (the 348-byte header lives at the very start), so a full metadata pass
over all volumes costs KBs per case instead of hundreds of MB.
"""
from __future__ import annotations

import gzip
import os
import re
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field

ZIP_PATH = os.environ.get("TF_ZIP_PATH", "/path/to/toothfairy4_v03.zip")

# NIfTI-1 datatype code -> (numpy dtype string, bytes per voxel)
NIFTI_DTYPE = {
    2: ("uint8", 1), 4: ("int16", 2), 8: ("int32", 4), 16: ("float32", 4),
    64: ("float64", 8), 256: ("int8", 1), 512: ("uint16", 2), 768: ("uint32", 4),
}


@dataclass
class Case:
    case_id: str
    family: str                       # leading letter group of the case id (P/A/F/S)
    volume_member: str | None = None
    reports_en: list[str] = field(default_factory=list)   # zip member paths
    reports_it: list[str] = field(default_factory=list)


def index_cases(zip_path: str = ZIP_PATH) -> dict[str, Case]:
    """One pass over the central directory -> {case_id: Case}."""
    cases: dict[str, Case] = {}
    fam_re = re.compile(r"^([A-Za-z]+)\d+$")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) < 2 or not parts[1]:
                continue
            cid = parts[0]
            if cid not in cases:
                m = fam_re.match(cid)
                cases[cid] = Case(case_id=cid, family=m.group(1) if m else "?")
            c = cases[cid]
            if name.endswith("cbct/volume.nii.gz"):
                c.volume_member = name
            elif "/reports_en/" in name and name.endswith(".txt"):
                c.reports_en.append(name)
            elif "/reports_it/" in name and name.endswith(".txt"):
                c.reports_it.append(name)
    return cases


def read_text(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member) as fh:
        return fh.read().decode("utf-8", errors="replace")


def read_nifti_header(zf: zipfile.ZipFile, member: str) -> dict:
    """Parse shape/spacing/dtype from a .nii.gz by reading only the header.

    Decompresses just enough of the gzip stream to cover the 348-byte NIfTI-1
    header. Returns dict with dim, pixdim, dtype, sizeof_hdr, magic.
    """
    with zf.open(member) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            hdr = gz.read(352)
    if len(hdr) < 348:
        raise ValueError(f"short header ({len(hdr)} bytes) for {member}")
    sizeof_hdr = struct.unpack_from("<i", hdr, 0)[0]
    little = sizeof_hdr == 348
    end = "<" if little else ">"
    dim = struct.unpack_from(end + "8h", hdr, 40)
    datatype = struct.unpack_from(end + "h", hdr, 70)[0]
    bitpix = struct.unpack_from(end + "h", hdr, 72)[0]
    pixdim = struct.unpack_from(end + "8f", hdr, 76)
    vox_offset = struct.unpack_from(end + "f", hdr, 108)[0]
    magic = hdr[344:348].rstrip(b"\x00").decode("latin1")
    ndim = dim[0]
    shape = tuple(dim[1:1 + ndim]) if 1 <= ndim <= 7 else tuple(dim[1:4])
    spacing = tuple(round(pixdim[i + 1], 4) for i in range(min(ndim, 3)))
    dtype_name, _ = NIFTI_DTYPE.get(datatype, (f"code{datatype}", bitpix // 8))
    return {
        "shape": shape,
        "spacing": spacing,
        "ndim": ndim,
        "dtype": dtype_name,
        "bitpix": bitpix,
        "sizeof_hdr": sizeof_hdr,
        "vox_offset": vox_offset,
        "magic": magic,
        "byteorder": "little" if little else "big",
    }


def report_rid(member: str) -> str:
    """Report id from member path, e.g. .../reports_en/cbct_34_3872.txt -> cbct_34_3872."""
    return member.rsplit("/", 1)[-1][:-4]


if __name__ == "__main__":
    cases = index_cases()
    by_fam = defaultdict(int)
    for c in cases.values():
        by_fam[c.family] += 1
    print(f"cases: {len(cases)}")
    print("by family:", dict(sorted(by_fam.items())))
    with zipfile.ZipFile(ZIP_PATH) as zf:
        sample = next(iter(cases.values()))
        print("sample case:", sample.case_id, sample.family)
        print("  volume:", sample.volume_member)
        print("  header:", read_nifti_header(zf, sample.volume_member))
