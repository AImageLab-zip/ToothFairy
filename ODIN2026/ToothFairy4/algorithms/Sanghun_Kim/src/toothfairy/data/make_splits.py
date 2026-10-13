#!/usr/bin/env python
"""Build the internal train/val split (per patient PID, prefix-stratified, deterministic).

Reads seed / val_ratio / strategy from a single source (split_config.yaml), scans the
`data/<PID>` folders and splits the valid PIDs into train/val, stratified by the PID prefix
(A/F/P/S). The same seed gives the same split (reproducible). No inline `random.split` is used —
the split is a deterministic function of the seed.

Official split: train 630 (public) / test 50 (hidden, not available locally). This script only
separates an internal val set out of the 627 local cases; it does not create a test split.

Regenerate:
    python -m toothfairy.data.make_splits --config data/splits/split_config.yaml

Outputs:
    data/splits/{train.txt, val.txt, split_manifest.json}
    <log_dir>/data_inventory.json
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Paths are resolved against the repository root (see toothfairy.paths).
from ..paths import REPO as ROOT  # noqa: E402
KST = timezone(timedelta(hours=9))
DEFAULT_CONFIG = ROOT / "data" / "splits" / "split_config.yaml"


# --------------------------------------------------------------------------- #
# Reproducibility: fix the seed at the entry point
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Config loading (pyyaml if available, otherwise a minimal parser for flat configs)
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _minimal_yaml(text)


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v.strip().strip('"').strip("'")


def _minimal_yaml(text: str) -> dict:
    """Minimal parser for flat / one-level configs, used when pyyaml is unavailable.

    Supports only `key: value` and a single level of two-space indented nesting, i.e. exactly the
    structure of split_config.yaml.
    """
    root: dict = {}
    stack = [(-1, root)]  # (indent, container)
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(val)
    return root


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def git_commit(root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root)
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(root)
            ).strip()
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def prefix_of(pid: str) -> str:
    """Leading alphabetic prefix of a PID (A003 -> A, S0008 -> S)."""
    m = re.match(r"[A-Za-z]+", pid)
    return m.group(0) if m else "?"


def list_sha256(train: list, val: list) -> str:
    """Canonical sha256 of the sorted train/val lists, used to check split identity."""
    canonical = "\n".join(sorted(train)) + "\n--\n" + "\n".join(sorted(val))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Data inventory
# --------------------------------------------------------------------------- #
def scan_inventory(data_root: Path) -> dict:
    """Scan `data/<PID>` -> valid/invalid PIDs, report counts, prefix distribution."""
    pids = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and p.name != "splits"
    )

    valid, invalid, multi_report, records = [], [], [], []
    for pid in pids:
        d = data_root / pid
        has_cbct = (d / "cbct" / "volume.nii.gz").is_file()
        en_dir, it_dir = d / "reports_en", d / "reports_it"
        n_en = len(list(en_dir.glob("*.txt"))) if en_dir.is_dir() else 0
        n_it = len(list(it_dir.glob("*.txt"))) if it_dir.is_dir() else 0

        reasons = []
        if not has_cbct:
            reasons.append("missing_cbct")
        if n_en == 0:
            reasons.append("no_reports_en")

        records.append({
            "pid": pid, "prefix": prefix_of(pid),
            "has_cbct": has_cbct, "n_reports_en": n_en, "n_reports_it": n_it,
        })
        if reasons:
            invalid.append({"pid": pid, "reasons": reasons})
        else:
            valid.append(pid)
            if n_en > 1:
                multi_report.append({"pid": pid, "n_reports_en": n_en})

    def prefix_dist(items):
        dist: dict = {}
        for pid in items:
            dist[prefix_of(pid)] = dist.get(prefix_of(pid), 0) + 1
        return dict(sorted(dist.items()))

    try:
        data_root_str = str(data_root.relative_to(ROOT))
    except ValueError:
        data_root_str = str(data_root)

    return {
        "data_root": data_root_str,
        "n_folders": len(pids),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_ratio": round(len(invalid) / len(pids), 4) if pids else 0.0,
        "valid_pids": valid,
        "invalid_pids": invalid,
        "prefix_distribution": prefix_dist(valid),
        "n_multi_report_en": len(multi_report),
        "multi_report_en": multi_report,
        "records": records,
    }


# --------------------------------------------------------------------------- #
# Stratified split (deterministic)
# --------------------------------------------------------------------------- #
def _group_seed(seed: int, prefix: str) -> int:
    """Deterministic integer seed derived from seed+prefix (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256(f"{seed}:{prefix}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 32)


def stratified_split(valid: list, val_ratio: float, seed: int):
    """Sort per prefix, shuffle deterministically, and move round(n*ratio) cases to val."""
    groups: dict = {}
    for pid in sorted(valid):
        groups.setdefault(prefix_of(pid), []).append(pid)

    train, val = [], []
    for prefix in sorted(groups):
        members = sorted(groups[prefix])
        rng = random.Random(_group_seed(seed, prefix))
        shuffled = members[:]
        rng.shuffle(shuffled)
        n_val = int(len(members) * val_ratio + 0.5)  # round half up
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return sorted(train), sorted(val)


def prefix_counts(pids: list) -> dict:
    dist: dict = {}
    for pid in pids:
        dist[prefix_of(pid)] = dist.get(prefix_of(pid), 0) + 1
    return dict(sorted(dist.items()))


# --------------------------------------------------------------------------- #
# Writing / verification
# --------------------------------------------------------------------------- #
def write_list(path: Path, pids: list) -> None:
    path.write_text("\n".join(sorted(pids)) + "\n", encoding="utf-8")


def read_list(path: Path) -> list:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def verify(splits_dir: Path, valid: list, val_ratio: float, seed: int) -> None:
    """Re-read the written files and assert that re-running with the same seed is
    deterministic."""
    train = read_list(splits_dir / "train.txt")
    val = read_list(splits_dir / "val.txt")
    s_train, s_val, s_valid = set(train), set(val), set(valid)

    assert s_train.isdisjoint(s_val), "train and val overlap"
    assert s_train | s_val == s_valid, "train + val does not cover every valid PID"
    assert len(train) == len(s_train), "duplicate entries in train.txt"
    assert len(val) == len(s_val), "duplicate entries in val.txt"

    re_train, re_val = stratified_split(valid, val_ratio, seed)
    assert re_train == sorted(train) and re_val == sorted(val), "re-run gave a different split"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build the deterministic train/val split")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to split_config.yaml")
    ap.add_argument("--data-root", default=None, help="patient folder root (overrides config)")
    ap.add_argument("--splits-dir", default=None, help="split output dir (overrides config)")
    ap.add_argument("--log-dir", default=None, help="inventory output dir (overrides config)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg["experiment"]["seed"])
    val_ratio = float(cfg["split"]["val_ratio"])
    strategy = cfg["split"]["strategy"]
    source = cfg["split"]["source"]
    data_cfg = cfg.get("data", {})

    data_root = Path(args.data_root or (ROOT / data_cfg.get("root", "data")))
    splits_dir = Path(args.splits_dir or (ROOT / data_cfg.get("splits_dir", "data/splits")))
    log_dir = Path(args.log_dir or (ROOT / data_cfg.get("log_dir", "experiments/logs/splits")))
    splits_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # 1) Inventory
    inv = scan_inventory(data_root)
    (log_dir / "data_inventory.json").write_text(
        json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[inventory] folders={inv['n_folders']} valid={inv['n_valid']} "
          f"invalid={inv['n_invalid']} (ratio={inv['invalid_ratio']:.2%}) "
          f"multi_report_en={inv['n_multi_report_en']}")
    if inv["invalid_ratio"] > 0.05:
        print(f"[WARN] invalid PID ratio {inv['invalid_ratio']:.2%} > 5% — check preprocessing")

    valid = inv["valid_pids"]
    if not valid:
        raise SystemExit("no valid PIDs found — check the data path and folder structure")

    # 2) Split
    train, val = stratified_split(valid, val_ratio, seed)
    write_list(splits_dir / "train.txt", train)
    write_list(splits_dir / "val.txt", val)

    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "strategy": strategy,
        "n_total": len(valid),
        "counts": {"train": len(train), "val": len(val)},
        "actual_val_ratio": round(len(val) / len(valid), 4),
        "prefix_distribution": {
            "train": prefix_counts(train),
            "val": prefix_counts(val),
        },
        "source": source,
        "git_commit": git_commit(ROOT),
        "created_at": datetime.now(KST).isoformat(),
        "list_sha256": list_sha256(train, val),
    }
    (splits_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) Verify
    verify(splits_dir, valid, val_ratio, seed)
    print(f"[split] train={len(train)} val={len(val)} "
          f"(val_ratio={manifest['actual_val_ratio']:.4f}) "
          f"sha256={manifest['list_sha256'][:16]}")
    print(f"[split] prefix train={manifest['prefix_distribution']['train']} "
          f"val={manifest['prefix_distribution']['val']}")
    print("[verify] disjoint + union + deterministic re-run OK")


if __name__ == "__main__":
    main()
