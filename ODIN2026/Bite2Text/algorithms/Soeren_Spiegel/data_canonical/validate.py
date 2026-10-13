"""Validate + normalise canonical report records, and measure structure-space agreement.

Reads data_canonical/raw_batches/*.json (each: {case_id: record}), checks the schema, fills defaults, and reports:
  - JSON parse + FDI-key validity + controlled-vocabulary violations
  - coverage: every split case has a record; not_assessable / FOV never dropped
  - inter-annotator agreement in structure space (FDI-set Jaccard) on multi-report cases

Usage: python data_canonical/validate.py [--emit]   (--emit writes cases/ + splits/canonical/)
"""
from __future__ import annotations
import json, sys, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = os.path.join(ROOT, "data_canonical", "raw_batches")
CASES = os.path.join(ROOT, "data_canonical", "cases")
SPLITS_OUT = os.path.join(ROOT, "splits", "canonical")

FOV_STATUS = {"included","partial","excluded","not_assessable","not_mentioned"}
TOOTH_STATUS = {"present","absent","impacted","root_remnant","not_assessable","not_mentioned"}
RESTOR = {"filling","prosthetic_crown","post_and_core","bridge_abutment","prosthetic_pontic","implant","implant_crown"}
ENDO = {None,"adequate","inadequate","present"}
LESION = {None,"periapical_granuloma","periapical_cyst","periapical_lesion","root_fracture","caries","peri_implant_resorption","root_remnant","other"}
CANAL = {"regular","lingual","apico_lingual","buccal","not_assessable","not_mentioned"}
FOV_REGIONS = ["mandible","maxilla","maxillary_sinuses","mandibular_condyles","coronoid_processes"]

# Valid FDI: permanent 11-48, deciduous 51-85. Quadrant 1-8, position 1-8.
def valid_fdi(k: str) -> bool:
    return len(k)==2 and k.isdigit() and k[0] in "12345678" and k[1] in "12345678"


def load_reports_meta():
    d = os.environ.get("TF_REPORTS_DIR", "/path/to/reports_en")
    return json.load(open(os.path.join(d, "_manifest.json")))


def norm_tooth(t):
    return {
        "status": t.get("status","not_mentioned"),
        "restorations": t.get("restorations",[]),
        "endo": t.get("endo"),
        "lesion": t.get("lesion"),
        "impaction": t.get("impaction"),
        "note": t.get("note",""),
        "src": t.get("src","both"),
    }


def norm_record(cid, fam, r, manifest):
    rec = {
        "case_id": cid, "family": fam,
        "n_source_reports": r.get("n_source_reports"),
        "source_rids": r.get("source_rids", []),
        "fov": {}, "teeth": {}, "prostheses": r.get("prostheses",[]),
        "ian_canal": r.get("ian_canal", {"left":{"course":"not_mentioned"},"right":{"course":"not_mentioned"}}),
        "bone_and_perio": r.get("bone_and_perio",{}),
        "sinuses": r.get("sinuses",{"status":"not_mentioned"}),
        "other_findings": r.get("other_findings",[]),
        "merge_notes": r.get("merge_notes",[]),
    }
    for reg in FOV_REGIONS:
        f = r.get("fov",{}).get(reg, {"status":"not_mentioned"})
        rec["fov"][reg] = {"status": f.get("status","not_mentioned"), "src": f.get("src","both"), "note": f.get("note","")}
    for k,t in r.get("teeth",{}).items():
        rec["teeth"][k] = norm_tooth(t)
    return rec


def validate():
    manifest = load_reports_meta()
    splits = {s: json.load(open(os.path.join(ROOT,"splits",f"{s}.json"))) for s in ("train","val","test")}
    records = {}
    errors, warns = [], []
    for path in sorted(glob.glob(os.path.join(BATCH,"*.json"))):
        data = json.load(open(path))
        for cid, r in data.items():
            fam = ''.join(c for c in cid if c.isalpha())
            if cid in records:
                errors.append(f"duplicate record {cid}")
            # schema checks
            for k,t in r.get("teeth",{}).items():
                if not valid_fdi(k): errors.append(f"{cid}: bad FDI key '{k}'")
                if t.get("status") not in TOOTH_STATUS: errors.append(f"{cid}.{k}: bad status {t.get('status')}")
                for rr in t.get("restorations",[]):
                    if rr not in RESTOR: errors.append(f"{cid}.{k}: bad restoration {rr}")
                if t.get("endo") not in ENDO: errors.append(f"{cid}.{k}: bad endo {t.get('endo')}")
                if t.get("lesion") not in LESION: errors.append(f"{cid}.{k}: bad lesion {t.get('lesion')}")
            for reg,f in r.get("fov",{}).items():
                if reg not in FOV_REGIONS: errors.append(f"{cid}: bad fov region {reg}")
                if f.get("status") not in FOV_STATUS: errors.append(f"{cid}.{reg}: bad fov status {f.get('status')}")
            for side in ("left","right"):
                c = r.get("ian_canal",{}).get(side,{})
                if c and c.get("course") not in CANAL: errors.append(f"{cid}.canal.{side}: bad course {c.get('course')}")
            # source count matches manifest
            if cid in manifest and r.get("n_source_reports") not in (None, manifest[cid]["n_en"]):
                warns.append(f"{cid}: n_source_reports {r.get('n_source_reports')} != manifest {manifest[cid]['n_en']}")
            records[cid] = norm_record(cid, fam, r, manifest)
    # coverage
    for s,cs in splits.items():
        missing = [c for c in cs if c not in records]
        if missing: warns.append(f"{s}: {len(missing)} cases without a record: {missing[:8]}{'...' if len(missing)>8 else ''}")
    return records, splits, errors, warns


def agreement(records):
    """FDI-set Jaccard per multi-report case: |teeth src==both| / |all teeth|."""
    per_case = {}
    fov_agree = collections.Counter(); fov_tot = collections.Counter()
    for cid,rec in records.items():
        if rec["n_source_reports"] and rec["n_source_reports"]>=2:
            teeth = rec["teeth"]
            if teeth:
                both = sum(1 for t in teeth.values() if t["src"]=="both")
                per_case[cid] = both/len(teeth)
            for reg,f in rec["fov"].items():
                if f["status"]!="not_mentioned":
                    fov_tot[reg]+=1
                    if f["src"]=="both": fov_agree[reg]+=1
    return per_case, fov_agree, fov_tot


if __name__ == "__main__":
    records, splits, errors, warns = validate()
    print(f"records: {len(records)}")
    print(f"ERRORS: {len(errors)}")
    for e in errors[:40]: print("  ✗", e)
    print(f"WARNINGS: {len(warns)}")
    for w in warns[:40]: print("  !", w)

    per_case, fov_agree, fov_tot = agreement(records)
    if per_case:
        import statistics
        vals = list(per_case.values())
        print(f"\nStructure-space FDI-set Jaccard (multi-report cases, n={len(vals)}):")
        print(f"  mean={statistics.mean(vals):.3f}  median={statistics.median(vals):.3f}")
        print(f"  (prose baseline from diagnosis: FDI 0.50, sentence 0.00)")
        print("FOV agreement (src==both / stated):")
        for reg in FOV_REGIONS:
            if fov_tot[reg]: print(f"  {reg}: {fov_agree[reg]}/{fov_tot[reg]} = {fov_agree[reg]/fov_tot[reg]:.2f}")

    # not_assessable / FOV never dropped audit
    na = sum(1 for r in records.values() for f in r["fov"].values() if f["status"]=="not_assessable")
    na += sum(1 for r in records.values() for t in r["teeth"].values() if t["status"]=="not_assessable")
    print(f"\nnot_assessable values preserved: {na}")

    if "--emit" in sys.argv:
        os.makedirs(CASES, exist_ok=True); os.makedirs(SPLITS_OUT, exist_ok=True)
        for cid,rec in records.items():
            json.dump(rec, open(os.path.join(CASES,f"{cid}.json"),"w"), indent=1, ensure_ascii=False)
        from render_prose import render
        for s,cs in splits.items():
            with open(os.path.join(SPLITS_OUT,f"{s}.jsonl"),"w") as fh:
                for c in cs:
                    if c in records:
                        rec = records[c]
                        fh.write(json.dumps({"case_id":c,"family":rec["family"],"record":rec,"prose":render(rec)}, ensure_ascii=False)+"\n")
        print(f"\nemitted cases/ and splits/canonical/*.jsonl")
    if errors: sys.exit(1)
