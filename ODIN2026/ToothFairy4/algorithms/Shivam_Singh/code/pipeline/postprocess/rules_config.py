#!/usr/bin/env python3
"""
code/pipeline/postprocess/rules_config.py

THE ARM IS THE CONFIG FILE. The code is not.

Every knob that docs/postprocess.md varies between arms -- the eleven source
rules, the cross-source vote gates, the maxilla-FOV policy, and the one
renderer switch -- used to be a module-level constant edited in place, or a CLI
flag added to three scripts. Both make an arm something you reconstruct from a
git diff and a job-script header. Here they are one YAML file, and a run's arm
is a path you can name, diff and cite.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT
────────────────────────────────────────────────────
It is a BINDING TABLE plus a loader. `BINDINGS` below is the single place in
the repo that says "config key X sets postprocess_pred.Y"; `apply()` writes
those module globals, and the ~4000 lines of postprocess_pred.py,
source_rules.py and synthesize_report.py read them exactly as they always did.
No builder signature changed and no logic moved.

It is NOT a general settings system. Only VARIABLE things live here. The FDI
ranges, the enum vocabulary, ARCH_VALUE_ALIASES, ARCH_FINDING_PRIORITY,
GATE_EXEMPT and every builder function stay in code, because they are what the
schema and the anatomy say, not what an experiment chooses.

WHY GLOBALS AND NOT A CONFIG OBJECT THREADED THROUGH
────────────────────────────────────────────────────
postprocess_pred.py has no test suite; the only verification available is a
survey diff. Threading a `cfg` parameter through the nine builders and their
forty-odd helpers is a large, silent-failure-shaped change for no behavioural
gain. Setting the globals the code already reads is a change of ~0 lines in
the logic and reproduces the previous defaults exactly when no config is
given -- which is the property that makes every stored measurement still
comparable.

FAIL LOUD ON AN UNKNOWN KEY
───────────────────────────
A typo in a config file must not be a silently-ignored arm. `load()` rejects
any key not in BINDINGS, and any value of the wrong type, with the path to the
offending key. The failure mode this replaces -- `RULES["fillings"] = False`
misspelled as `RULES["filling"]`, running clean, and scoring like the default
-- cost a day once.

FORMAT
──────
YAML (PyYAML is in env/requirements.txt, the cbct_base env that runs the whole
postprocess path) so each key can carry its one-line why and a pointer to its
section of docs/postprocess.md. `.json` is accepted too, for an environment
without PyYAML.

USAGE
─────
    python3 code/pipeline/postprocess/postprocess_pred.py \\
        --config configs/postprocess/default.yaml ...

    STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \\
        scripts/run_infer.sh validate

    # in code
    import rules_config
    rules_config.load_and_apply("configs/postprocess/default.yaml")

Omitting --config applies DEFAULTS below, which are byte-identical to the
constants the modules carried before this file existed -- i.e. the arm-6
settings that scored 0.4658. `configs/postprocess/default.yaml` restates them
explicitly, and `verify_defaults()` is the check that the two never drift.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo bootstrap -- see code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()


#: Where the shipped arm configs live.
CONFIG_DIR = REPO_ROOT / "configs" / "postprocess"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


# ── the binding table ───────────────────────────────────────────────────────
#
# One entry per knob: dotted config path -> where it is read from at runtime.
#
#   target   "<module>.<GLOBAL>", or "<module>.<GLOBAL>[<key>]" for one member
#            of a dict-valued constant. A LIST of targets means the same value
#            feeds two modules -- the only such case is the canal-location
#            prior, which postprocess/source_rules and synthesize_report each
#            carried their own copy of. The config is now the one place it is
#            stated, which is a duplicate constant removed rather than moved.
#   kind     bool | str | prefix_set. Checked on load; a wrong type is an error
#            with the config path in the message, not a coercion.
#   doc      the section of docs/postprocess.md that measured it.

class _B:
    __slots__ = ("targets", "kind", "doc")

    def __init__(self, target, kind: str = "bool", doc: str = ""):
        self.targets = [target] if isinstance(target, str) else list(target)
        self.kind = kind
        self.doc = doc


BINDINGS: Dict[str, _B] = {
    # ── the eleven source rules ────────────────────────────────────────────
    # Each re-sources ONE finding from the segmentation mask or from a single
    # image instead of voting between every read that can assert it. All are
    # no-ops without --facts-dir. docs/postprocess.md §1, "The eleven rules".
    "source_rules.absent_teeth":    _B("source_rules.RULES[absent_teeth]",   doc="THE RULE -- absent teeth"),
    "source_rules.impaction":       _B("source_rules.RULES[impaction]",      doc="THE RULE -- impaction"),
    "source_rules.endodontic":      _B("source_rules.RULES[endodontic]",     doc="THE RULE -- endodontic treatment"),
    "source_rules.fillings":        _B("source_rules.RULES[fillings]",       doc="THE RULE -- fillings"),
    "source_rules.crown":           _B("source_rules.RULES[crown]",          doc="THE RULE -- crown"),
    "source_rules.implants":        _B("source_rules.RULES[implants]",       doc="THE RULE -- implants"),
    "source_rules.canal_adjacent":  _B("source_rules.RULES[canal_adjacent]", doc="THE RULE -- canal-adjacent teeth"),
    "source_rules.canal_location":  _B("source_rules.RULES[canal_location]", doc="THE RULE -- canal location"),
    "source_rules.atrophy":         _B("source_rules.RULES[atrophy]",        doc="THE RULE -- alveolar bone atrophy"),
    "source_rules.bridges":         _B("source_rules.RULES[bridges]",        doc="THE RULE -- fixed bridges"),
    "source_rules.condyle_fov":     _B("source_rules.RULES[condyle_fov]",    doc="THE RULE -- condyle scope"),

    # ── priors ─────────────────────────────────────────────────────────────
    # Not a guess: 61 of the 70 validate sides a reference places
    # buccolingually are lingual, 640/154 in training, and neither read has
    # ever identified a buccal canal.
    "priors.canal_location": _B(["source_rules.DEFAULT_CANAL_LOCATION",
                                 "synthesize_report.DEFAULT_CANAL_LOCATION"],
                                kind="str", doc="THE RULE -- canal location"),

    # ── cross-source vote gates ────────────────────────────────────────────
    # The pre-2026-08-07 arm gated a finding on two sources agreeing; the
    # current one reports the UNION and only annotates the disagreement.
    # `require_agreement` is the master switch -- everything below it is
    # subordinate and only consulted while it is on.
    "cross_source.require_agreement":     _B("postprocess_pred.REQUIRE_CROSS_SOURCE_AGREEMENT"),
    "cross_source.impaction":             _B("postprocess_pred.CROSS_VALIDATE_IMPACTION"),
    "cross_source.endodontic":            _B("postprocess_pred.CROSS_VALIDATE_ENDODONTIC"),
    "cross_source.restorations.fillings": _B("postprocess_pred.CROSS_VALIDATE_RESTORATIONS[fillings]"),
    "cross_source.restorations.post_and_core": _B("postprocess_pred.CROSS_VALIDATE_RESTORATIONS[post_and_core]"),
    "cross_source.restorations.crown":    _B("postprocess_pred.CROSS_VALIDATE_RESTORATIONS[crown]"),
    "cross_source.canal_adjacency":       _B("postprocess_pred.CROSS_VALIDATE_CANAL_ADJACENCY"),
    "cross_source.prefer_composite_canal_location": _B("postprocess_pred.PREFER_COMPOSITE_CANAL_LOCATION"),
    "cross_source.caries_arch_agreement": _B("postprocess_pred.REQUIRE_CARIES_ARCH_AGREEMENT"),
    "cross_source.bone_quality_tooth_confirmation": _B("postprocess_pred.REQUIRE_BONE_QUALITY_TOOTH_CONFIRMATION"),

    # ── standalone gates ───────────────────────────────────────────────────
    "gates.demote_uncertain_to_normal":    _B("postprocess_pred.DEMOTE_UNCERTAIN_TO_NORMAL"),
    "gates.drop_impaction_on_absent_teeth": _B("postprocess_pred.DROP_IMPACTION_ON_ABSENT_TEETH"),
    "gates.drop_implants_on_present_teeth": _B("postprocess_pred.DROP_IMPLANTS_ON_PRESENT_TEETH"),

    # ── the maxilla FOV policy ─────────────────────────────────────────────
    # docs/postprocess.md, THE RULE -- maxilla FOV scope (2026-08-17).
    "maxilla_fov.drop_excluded":       _B("postprocess_pred.DROP_EXCLUDED_MAXILLA",
                                          doc="THE RULE -- maxilla FOV scope"),
    "maxilla_fov.drop_excluded_teeth": _B("postprocess_pred.DROP_EXCLUDED_MAXILLA_TEETH",
                                          doc="THE RULE -- maxilla FOV scope"),
    "maxilla_fov.force_excluded_prefixes": _B("postprocess_pred.FORCE_MAXILLA_EXCLUDED_PREFIXES",
                                              kind="prefix_set",
                                              doc="THE RULE -- maxilla FOV scope"),

    # ── extraction-side fallbacks ──────────────────────────────────────────
    # Required by any schema that omits the four 3D wisdom facts --
    # schema_dedup.json does. Not a no-op on the aksssr arm.
    "extraction.wisdom_eruption_fallback": _B("postprocess_pred.WISDOM_ERUPTION_FALLBACK"),

    # ── what reaches the report text ───────────────────────────────────────
    # crown and post_and_core are silenced in code (0.18 and 0.00 precision,
    # and no second source filters either); fillings is a switch because it is
    # only safe once THE RULE -- fillings has re-sourced the group.
    "report.render_fillings": _B("synthesize_report.RENDER_FILLINGS",
                                 doc="THE RULE -- fillings; §1 What reaches the report"),
}


#: The arm-6 settings, i.e. exactly what the modules carried as constants
#: before this file existed. Anything absent from a config file keeps these.
DEFAULTS: Dict[str, Any] = {
    "source_rules.absent_teeth":   True,
    "source_rules.impaction":      True,
    "source_rules.endodontic":     True,
    "source_rules.fillings":       True,
    "source_rules.crown":          True,
    "source_rules.implants":       True,
    "source_rules.canal_adjacent": True,
    "source_rules.canal_location": True,
    "source_rules.atrophy":        True,
    "source_rules.bridges":        True,
    "source_rules.condyle_fov":    True,

    "priors.canal_location": "lingual",

    "cross_source.require_agreement":     False,
    "cross_source.impaction":             False,
    "cross_source.endodontic":            False,
    "cross_source.restorations.fillings": False,
    "cross_source.restorations.post_and_core": True,
    "cross_source.restorations.crown":    True,
    "cross_source.canal_adjacency":       True,
    "cross_source.prefer_composite_canal_location": True,
    "cross_source.caries_arch_agreement": True,
    "cross_source.bone_quality_tooth_confirmation": True,

    "gates.demote_uncertain_to_normal":     False,
    "gates.drop_impaction_on_absent_teeth": False,
    "gates.drop_implants_on_present_teeth": True,

    "maxilla_fov.drop_excluded":           True,
    "maxilla_fov.drop_excluded_teeth":     False,
    "maxilla_fov.force_excluded_prefixes": [],

    "extraction.wisdom_eruption_fallback": False,

    "report.render_fillings": True,
}


# ── loading ─────────────────────────────────────────────────────────────────

class ConfigError(ValueError):
    """A config file that names a key or a value the pipeline cannot honour."""


def _flatten(tree: Dict, prefix: str = "") -> Dict[str, Any]:
    """Nested YAML -> the dotted keys of BINDINGS.

    A dict stops being descended into once its own dotted path is a binding --
    that is how `maxilla_fov.force_excluded_prefixes: [P, S]` stays a value
    rather than being read as three more keys.
    """
    flat: Dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and path not in BINDINGS:
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _check(path: str, value: Any) -> Any:
    """Type-check one value against its binding. Raises, never coerces."""
    kind = BINDINGS[path].kind
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if kind == "prefix_set":
        # 'PS', ['P', 'S'] and [] all mean the same thing. The empty list is
        # the default and the one that must survive a round trip.
        if isinstance(value, str):
            value = list(value)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{path}: expected a list of case-ID prefix "
                              f"letters, got {value!r}")
        return {c.upper() for c in value if c.strip()}
    raise ConfigError(f"{path}: unknown binding kind {kind!r}")   # unreachable


def read(path: "str | Path") -> Dict[str, Any]:
    """Parse one config file into flat, validated, dotted keys. No defaults."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"no such config file: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml                                    # noqa: PLC0415
        except ImportError as exc:                          # pragma: no cover
            raise ConfigError(
                f"{p} is YAML but PyYAML is not installed in this environment. "
                f"Install it (it is in env/requirements.txt) or write the "
                f"config as .json -- the loader accepts both.") from exc
        tree = yaml.safe_load(text) or {}
    else:
        tree = json.loads(text)
    if not isinstance(tree, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(tree).__name__}")

    # `arm:` and `notes:` are free-text provenance a config may carry for the
    # reader's benefit; they bind to nothing and are recorded, not applied.
    meta = {k: tree.pop(k) for k in ("arm", "notes", "based_on") if k in tree}

    flat = _flatten(tree)
    unknown = sorted(k for k in flat if k not in BINDINGS)
    if unknown:
        raise ConfigError(
            f"{p}: unknown key(s): {', '.join(unknown)}\n"
            f"  Every settable key is listed in BINDINGS in "
            f"code/pipeline/postprocess/rules_config.py; `--print-config` prints them "
            f"with their current values.")
    out = {k: _check(k, v) for k, v in flat.items()}
    if meta:
        out["_meta"] = meta
    return out


def resolve(path: "Optional[str | Path]" = None,
            overrides: "Optional[Dict[str, Any]]" = None) -> Dict[str, Any]:
    """DEFAULTS, overlaid by the config file, overlaid by explicit overrides.

    The three-layer order is what keeps the legacy CLI flags working: they are
    passed as `overrides` and so still win over a config file, which is what
    someone typing `--demote-uncertain` on top of an arm config means.
    """
    settings = copy.deepcopy(DEFAULTS)
    # The one default whose runtime type is not its YAML type: the module
    # constant is a set, and DEFAULTS states it as [] so default.yaml can too.
    settings["maxilla_fov.force_excluded_prefixes"] = set()
    meta: Dict[str, Any] = {}
    if path is not None:
        loaded = read(path)
        meta = loaded.pop("_meta", {})
        settings.update(loaded)
    for key, value in (overrides or {}).items():
        if key not in BINDINGS:
            raise ConfigError(f"override names an unknown key: {key}")
        settings[key] = _check(key, value) if not isinstance(value, set) else value
    settings["_meta"] = meta
    return settings


# ── applying ────────────────────────────────────────────────────────────────

#: What the last apply() installed, for provenance. Read by
#: postprocess_pred.postprocess_prediction and stamped into every summary.
ACTIVE: Dict[str, Any] = {"config": None, "settings": {}}


def _module(name: str):
    """The LIVE module object called `name`, __main__ included.

    THIS IS NOT `importlib.import_module(name)`, and the difference is the
    whole correctness of this file. A script run as
    `python3 .../synthesize_report.py` is in sys.modules under `__main__`;
    import_module("synthesize_report") then imports a SECOND, independent copy
    and every global written to it is written to an object nothing reads. The
    symptom is a config that loads cleanly, reports the right values from
    --print-config, and changes nothing about the output.

    So: check __main__ first, by the file it was run from.
    """
    import importlib
    main = _sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if main_file and Path(main_file).stem == name:
        return main
    return importlib.import_module(name)


def _set(target: str, value: Any) -> None:
    """Write one module global, or one member of a dict-valued one."""
    mod_name, _, attr = target.partition(".")
    module = _module(mod_name)
    if attr.endswith("]"):
        attr, _, key = attr.partition("[")
        getattr(module, attr)[key.rstrip("]")] = value
    else:
        setattr(module, attr, value)


def apply(settings: Dict[str, Any], source: "Optional[str | Path]" = None) -> None:
    """Install `settings` into the modules that read them.

    Every module named in BINDINGS is resolved here rather than imported at
    file scope: synthesize_report.py is a separate process from
    postprocess_pred.py in the normal pipeline, and neither should have to
    import the other because a config file happens to mention both. A module
    this process does not have loaded is imported on demand; one it is RUNNING
    is found under __main__ -- see _module.
    """
    for key, value in settings.items():
        if key == "_meta":
            continue
        for target in BINDINGS[key].targets:
            _set(target, value)
    ACTIVE["config"] = str(source) if source is not None else None
    ACTIVE["settings"] = {k: sorted(v) if isinstance(v, set) else v
                          for k, v in settings.items() if k != "_meta"}
    ACTIVE["meta"] = settings.get("_meta", {})


def load_and_apply(path: "Optional[str | Path]" = None,
                   overrides: "Optional[Dict[str, Any]]" = None) -> Dict[str, Any]:
    """read + resolve + apply, the one call every entry point makes."""
    settings = resolve(path, overrides)
    apply(settings, path)
    return settings


def provenance() -> Dict[str, Any]:
    """What went into the summary JSON so a stored run names its own arm.

    Only the NON-DEFAULT settings are listed, plus the config path. A summary
    that carries `{"config": null, "differs_from_default": {}}` was produced by
    the arm-6 defaults, and that is worth being able to read off the file
    rather than off a job log that has since rotated.
    """
    settings = ACTIVE.get("settings") or {}
    differs = {}
    for key, value in settings.items():
        default = DEFAULTS.get(key)
        if isinstance(default, list):
            default = sorted(default)
        if value != default:
            differs[key] = value
    out = {"config": ACTIVE.get("config"), "differs_from_default": differs}
    if ACTIVE.get("meta"):
        out.update(ACTIVE["meta"])
    return out


# ── introspection, and the drift check ──────────────────────────────────────

def describe(settings: "Optional[Dict[str, Any]]" = None) -> str:
    """Every settable key, its value, and the section that measured it."""
    settings = settings if settings is not None else resolve()
    width = max(len(k) for k in BINDINGS)
    lines: List[str] = []
    section = None
    for key in BINDINGS:
        head = key.split(".")[0]
        if head != section:
            section = head
            lines.append("")
        value = settings.get(key, DEFAULTS.get(key))
        if isinstance(value, set):
            value = sorted(value)
        mark = " " if value == DEFAULTS.get(key) or (
            isinstance(value, list) and sorted(value) == sorted(DEFAULTS.get(key) or [])) else "*"
        doc = f"   # docs/postprocess.md {BINDINGS[key].doc}" if BINDINGS[key].doc else ""
        lines.append(f"{mark} {key:<{width}} = {json.dumps(value)}{doc}")
    lines.append("")
    lines.append("(* = differs from the arm-6 default)")
    return "\n".join(lines).lstrip("\n")


def verify_defaults() -> List[str]:
    """DEFAULTS against the constants the modules actually carry.

    The one thing this design can get wrong is DEFAULTS drifting from the
    module constant it claims to restate -- then a run WITHOUT --config and a
    run WITH default.yaml quietly differ. Called by --print-config, and cheap
    enough to call anywhere.
    """
    problems: List[str] = []
    for key, default in DEFAULTS.items():
        for target in BINDINGS[key].targets:
            mod_name, _, attr = target.partition(".")
            try:
                module = _module(mod_name)
            except ImportError:
                continue        # a module this process does not use
            if attr.endswith("]"):
                attr, _, dkey = attr.partition("[")
                have = getattr(module, attr).get(dkey.rstrip("]"))
            else:
                have = getattr(module, attr, None)
            want = set() if (BINDINGS[key].kind == "prefix_set" and not default) else default
            if have != want:
                problems.append(f"{key}: DEFAULTS says {default!r}, "
                                f"{target} carries {have!r}")
    return problems


# ── CLI: `python3 code/pipeline/postprocess/rules_config.py [config.yaml]` ──

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Print the resolved postprocess configuration -- every "
                    "settable key, its value, and where it was measured.")
    ap.add_argument("config", nargs="?", default=None,
                    help="A config file to resolve over the defaults. "
                         "Omitted, the arm-6 defaults are printed.")
    ap.add_argument("--check", action="store_true",
                    help="Also verify DEFAULTS against the module constants "
                         "and exit non-zero if they have drifted.")
    args = ap.parse_args()

    try:
        settings = resolve(args.config)
    except ConfigError as exc:
        raise SystemExit(f"[FAIL] {exc}")
    if settings.get("_meta"):
        for k, v in settings["_meta"].items():
            print(f"{k}: {v}")
        print()
    print(describe(settings))

    if args.check:
        import postprocess_pred   # noqa: F401  -- imported for its constants
        import source_rules       # noqa: F401
        import synthesize_report  # noqa: F401
        problems = verify_defaults()
        if problems:
            print("\n[FAIL] DEFAULTS have drifted from the module constants:")
            for p in problems:
                print(f"  {p}")
            raise SystemExit(1)
        print("\n[PASS] DEFAULTS match the module constants.")


if __name__ == "__main__":
    main()
