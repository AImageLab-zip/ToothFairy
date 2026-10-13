#!/usr/bin/env python3
"""Merge independent Stage-2 branch predictions into case-level evidence reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evigre.dataloading.dataset_toothfairy_stage2 import FDI_TEETH, GLOBAL_LABELS, RELATION_LABELS, TOOTH_LABELS
from evigre.dataloading.dataset_toothfairy_stage2_branch import (
    TEN_REGION_NAMES,
    ToothFairyStage2GlobalDataset,
    ToothFairyStage2RelationDataset,
    ToothFairyStage2ToothInstanceDataset,
)
from evigre.inference.model_loading import load_model_from_checkpoint


DEFAULT_ROOT = Path("results/stage2")
DEFAULT_PREPROCESSED = Path("data/stage2_preprocessed/Dataset401_EviDentCBCT_Stage2Evidence")


TOOTH_FACT_MAP = {
    'missing': ('missing_tooth', 'missing tooth'),
    'implant': ('implant', 'implant present'),
    'crown': ('crown', 'prosthetic crown present'),        # v2
    'restoration': ('restoration', 'restoration present'),  # v2
    'endodontic_treatment': ('endodontic_treatment', 'endodontic treatment present'),
    'prosthesis_related': ('prosthesis', 'prosthesis-related finding present'),
    'impacted_or_disodontiasis': ('impaction', 'impaction or disodontiasis present'),
    'caries': ('caries', 'caries present'),
    'root_remnant': ('root_remnant', 'root remnant present'),
    'root_fracture': ('root_fracture', 'root fracture present'),
    'lesion_related': ('lesion', 'lesion-related finding present'),
}


# (kind, target, text, fact_type). fact_type is the render-facing 'type' for kind=='fact'
# entries (None for coverage/warning). v2 splits atrophy per-arch/severe and lesion by density.
GLOBAL_TEXT = {
    'maxilla_partially_included': ('coverage', 'maxilla', 'maxilla partially included', None),
    'maxilla_not_included': ('coverage', 'maxilla', 'maxilla not included', None),
    'maxillary_sinuses_not_included': ('coverage', 'maxillary_sinuses', 'maxillary sinuses not included', None),
    'mandibular_condyles_not_included': ('coverage', 'mandibular_condyles', 'mandibular condyles not included', None),
    'periodontitis_present': ('fact', 'periodontal_bone', 'periodontitis present', 'periodontitis'),
    'atrophy_maxilla': ('fact', 'maxilla', 'maxillary alveolar atrophy present', 'atrophy'),
    'atrophy_mandible': ('fact', 'mandible', 'mandibular alveolar atrophy present', 'atrophy'),
    'atrophy_severe': ('fact', 'alveolar_bone', 'severe alveolar atrophy present', 'atrophy_severe'),
    'osteolytic_lesion_present': ('fact', 'bone', 'osteolytic lesion present', 'osteolytic_lesion'),
    'osteosclerotic_lesion_present': ('fact', 'bone', 'osteosclerotic (condensing) lesion present', 'osteosclerotic_lesion'),
    'sinus_pathology_present': ('fact', 'maxillary_sinus', 'maxillary sinus pathology present', 'sinus_pathology'),
    'poor_quality_or_artifact': ('warning', 'image_quality', 'poor quality or artifact', None),
}


RELATION_TEXT = {
    '38_close_to_left_IAC': ('tooth:38', '38 close to left inferior alveolar canal'),
    '48_close_to_right_IAC': ('tooth:48', '48 close to right inferior alveolar canal'),
    'any_tooth_close_to_IAC': ('mandibular_canal', 'tooth close to inferior alveolar canal'),
    'lesion_close_to_IAC': ('mandibular_canal', 'lesion close to inferior alveolar canal'),
    'left_IAC_predominantly_lingual': ('mandibular_canal:left', 'left inferior alveolar canal predominantly lingual'),
    'left_IAC_predominantly_buccal': ('mandibular_canal:left', 'left inferior alveolar canal predominantly buccal'),
    'right_IAC_predominantly_lingual': ('mandibular_canal:right', 'right inferior alveolar canal predominantly lingual'),
    'right_IAC_predominantly_buccal': ('mandibular_canal:right', 'right inferior alveolar canal predominantly buccal'),
}


def _default_model_folder(root: Path, branch: str, plan: str) -> Path:
    trainer = 'nnUNetTrainer_ToothFairyStage2Branch'
    return root / f'{trainer}__{plan}__3d_fullres' / 'fold_0'


def _load_thresholds(path: Path | None, branch: str, fallback: float = 0.5, tooth_threshold_view: str = 'tooth') -> Dict[str, float]:
    names = {'global': GLOBAL_LABELS, 'tooth': TOOTH_LABELS, 'relation': RELATION_LABELS}[branch]
    if path is None or not path.exists():
        return {name: float(fallback) for name in names}
    obj = json.loads(path.read_text())
    if branch == 'tooth':
        candidates = [tooth_threshold_view, 'tooth', 'tooth_trusted', 'tooth_strict']
    else:
        candidates = [branch]
    values = None
    for key in candidates:
        if isinstance(obj, Mapping) and isinstance(obj.get(key), Mapping):
            values = obj[key]
            break
    if values is None and all(name in obj for name in names):
        values = obj
    if values is None:
        return {name: float(fallback) for name in names}
    return {name: float(values.get(name, fallback)) for name in names}


def _threshold_path(folder: Path, checkpoint_name: str) -> Path:
    stem = checkpoint_name.removesuffix('.pth')
    return folder / f'threshold_tuning_{stem}' / f'per_label_thresholds_{stem}.json'


def load_tooth_region_mask_config(path: Path | None) -> Dict[str, Any] | None:
    if path is None or str(path).strip() == '':
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text())
    if str(obj.get('action', 'force_negative')) != 'force_negative':
        raise ValueError(f"Unsupported tooth region mask action in {path}: {obj.get('action')!r}")
    return obj


def build_tooth_region_force_negative_lookup(config: Dict[str, Any] | None) -> tuple[set[tuple[int, str]], Dict[str, Any]]:
    if config is None:
        return set(), {}
    region_to_id = {name: i for i, name in enumerate(TEN_REGION_NAMES)}
    blocked: set[tuple[int, str]] = set()
    details = []
    for rule in config.get('rules', []):
        label = str(rule.get('label', ''))
        if label not in TOOTH_LABELS:
            raise ValueError(f'Unknown tooth label in region mask config: {label!r}')
        regions = [str(r) for r in rule.get('regions', [])]
        missing = [r for r in regions if r not in region_to_id]
        if missing:
            raise ValueError(f'Unknown tooth regions in region mask config for {label}: {missing}')
        for region in regions:
            blocked.add((region_to_id[region], label))
        details.append({'label': label, 'regions': regions})
    return blocked, {
        'tooth_region_mask_config': config.get('name', ''),
        'tooth_region_mask_action': 'force_negative_probability_to_zero',
        'blocked_pairs': details,
    }


def _device(name: str) -> torch.device:
    return torch.device(name if torch.cuda.is_available() or name == 'cpu' else 'cpu')


# Fusion modes that use precomputed Stage-1 visual tokens (everything except pure raw/3D).
_NO_VISUAL_TOKEN_MODES = {'scratch_only', 'raw_only', 'crop_only', '3d_only'}


def _use_visual_tokens(kwargs: Mapping[str, Any]) -> bool:
    return str(kwargs.get('visual_fusion_mode', 'scratch_only')).lower() not in _NO_VISUAL_TOKEN_MODES


def _to_device_optional(batch: Mapping[str, Any], key: str, device: torch.device):
    value = batch.get(key, None)
    return value.to(device) if torch.is_tensor(value) else None


@torch.no_grad()
def predict_global(csv_path: Path, checkpoint: Path, thresholds: Mapping[str, float], batch_size: int, num_workers: int, device_name: str) -> Dict[str, Dict[str, Any]]:
    device = _device(device_name)
    model, kwargs = load_model_from_checkpoint(checkpoint, device)
    ds = ToothFairyStage2GlobalDataset(
        str(csv_path),
        global_size=kwargs.get('global_size', [96, 192, 192]),
        include_mask_channel=kwargs.get('include_mask_channel', True),
        mask_mode=kwargs.get('mask_mode', 'semantic'),
        use_visual_tokens=_use_visual_tokens(kwargs),
        visual_token_levels=kwargs.get('visual_token_levels', [2, 3, 4]),
        visual_token_stats=kwargs.get('visual_token_stats', ['mean', 'max']),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    out: Dict[str, Dict[str, Any]] = {}
    model.eval()
    for batch in dl:
        logits = model(batch['global_image'].to(device), _to_device_optional(batch, 'global_visual_tokens', device))['global_logits']
        probs = torch.sigmoid(logits).cpu().numpy()
        for i, report_id in enumerate(batch['report_id']):
            labels = []
            for j, name in enumerate(GLOBAL_LABELS):
                prob = float(probs[i, j])
                labels.append({'label': name, 'probability': prob, 'threshold': float(thresholds[name]), 'predicted': bool(prob >= thresholds[name])})
            out[str(report_id)] = {'labels': labels, 'probabilities': probs[i].tolist()}
    return out


@torch.no_grad()
def predict_relation(csv_path: Path, checkpoint: Path, thresholds: Mapping[str, float], batch_size: int, num_workers: int, device_name: str) -> Dict[str, Dict[str, Any]]:
    device = _device(device_name)
    model, kwargs = load_model_from_checkpoint(checkpoint, device)
    ds = ToothFairyStage2RelationDataset(
        str(csv_path),
        relation_crop_size=kwargs.get('relation_crop_size', [80, 128, 128]),
        include_mask_channel=kwargs.get('include_mask_channel', True),
        mask_mode=kwargs.get('mask_mode', 'semantic'),
        use_visual_tokens=_use_visual_tokens(kwargs),
        visual_token_levels=kwargs.get('visual_token_levels', [2, 3, 4]),
        visual_token_stats=kwargs.get('visual_token_stats', ['mean', 'max']),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    out: Dict[str, Dict[str, Any]] = {}
    model.eval()
    for batch in dl:
        logits = model(batch['relation_crops'].to(device), _to_device_optional(batch, 'relation_visual_tokens', device))['relation_logits']
        probs = torch.sigmoid(logits).cpu().numpy()
        for i, report_id in enumerate(batch['report_id']):
            labels = []
            for j, name in enumerate(RELATION_LABELS):
                prob = float(probs[i, j])
                labels.append({'label': name, 'probability': prob, 'threshold': float(thresholds[name]), 'predicted': bool(prob >= thresholds[name])})
            out[str(report_id)] = {'labels': labels, 'probabilities': probs[i].tolist()}
    return out


@torch.no_grad()
def predict_tooth(csv_path: Path, checkpoint: Path, thresholds: Mapping[str, float], batch_size: int, num_workers: int, device_name: str, tooth_region_mask_config: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
    device = _device(device_name)
    model, kwargs = load_model_from_checkpoint(checkpoint, device)
    ds = ToothFairyStage2ToothInstanceDataset(
        str(csv_path),
        tooth_crop_size=kwargs.get('tooth_crop_size', [64, 96, 96]),
        include_mask_channel=kwargs.get('include_mask_channel', True),
        mask_mode=kwargs.get('mask_mode', 'semantic'),
        tooth_region_scheme=kwargs.get('tooth_region_scheme', 'ten_region'),
        use_visual_tokens=_use_visual_tokens(kwargs),
        visual_token_levels=kwargs.get('visual_token_levels', [2, 3, 4]),
        visual_token_stats=kwargs.get('visual_token_stats', ['mean', 'max']),
        include_prosthetic_channels=bool(kwargs.get('tooth_prosthetic_channels', False)),
        metal_channel=bool(kwargs.get('tooth_metal_channel', False)),
    )
    ds.active_indices = list(range(len(ds.instances)))
    ds.active_masks = [inst['base_mask'].copy() for inst in ds.instances]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    blocked_region_labels, postprocess_meta = build_tooth_region_force_negative_lookup(tooth_region_mask_config)
    out: Dict[str, Dict[str, Any]] = {}
    model.eval()
    for batch in dl:
        logits = model(batch['tooth_crop'].to(device), batch['tooth_region'].to(device), _to_device_optional(batch, 'tooth_visual_token', device))['tooth_logits']
        probs = torch.sigmoid(logits).cpu().numpy()
        for i, report_id in enumerate(batch['report_id']):
            rid = str(report_id)
            tooth_id = str(batch['tooth_id'][i])
            region_id = int(batch['tooth_region'][i]) if 'tooth_region' in batch else -1
            case = out.setdefault(rid, {'teeth': {}, 'tooth_order': FDI_TEETH})
            if postprocess_meta:
                case.setdefault('tooth_region_postprocess', postprocess_meta)
            labels = []
            row_probs = probs[i].copy()
            for j, name in enumerate(TOOTH_LABELS):
                forced_negative = (region_id, name) in blocked_region_labels
                prob = 0.0 if forced_negative else float(row_probs[j])
                labels.append({
                    'label': name,
                    'probability': prob,
                    'threshold': float(thresholds[name]),
                    'predicted': False if forced_negative else bool(prob >= thresholds[name]),
                    'forced_negative_by_region_mask': bool(forced_negative),
                })
                row_probs[j] = prob
            case['teeth'][tooth_id] = {
                'labels': labels,
                'probabilities': row_probs.tolist(),
                'region_id': region_id,
                'region': TEN_REGION_NAMES[region_id] if 0 <= region_id < len(TEN_REGION_NAMES) else '',
            }
    return out



def side_for_tooth(tooth_id: str) -> str:
    return 'right' if str(tooth_id).startswith(('1', '4')) else 'left' if str(tooth_id).startswith(('2', '3')) else 'unspecified'


def region_for_tooth(tooth_id: str) -> str:
    return 'maxilla' if str(tooth_id).startswith(('1', '2')) else 'mandible' if str(tooth_id).startswith(('3', '4')) else 'unspecified'


def make_fact(typ: str, target: str, finding: str, teeth: Sequence[str] | None, region: str, side: str, probability: float | None = None, source: str = 'stage2_prediction') -> Dict[str, Any]:
    item: Dict[str, Any] = {
        'type': typ,
        'target': target,
        'finding': finding,
        'teeth': list(teeth or []),
        'region': region,
        'side': side,
        'certainty': 'definite',
        'source': source,
    }
    if probability is not None:
        item['probability'] = float(probability)
    return item


def build_case_json(row: pd.Series, label_obj: Dict[str, Any], global_pred: Dict[str, Any], tooth_pred: Dict[str, Any], relation_pred: Dict[str, Any], model_sources: Dict[str, str] | None = None, threshold_policy: str | None = None) -> Dict[str, Any]:
    report_id = str(row['report_id'])
    case_id = str(row['case_id'])
    coverage: List[Dict[str, Any]] = []
    facts: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for item in global_pred.get('labels', []):
        if not item['predicted']:
            continue
        label = item['label']
        prob = float(item['probability'])
        kind, target, text, fact_type = GLOBAL_TEXT[label]
        if kind == 'coverage':
            status = 'partially_included' if 'partially' in text else 'not_included'
            coverage.append({'structure': target, 'status': status, 'side': 'bilateral' if 'sinuses' in target or 'condyles' in target else 'unspecified', 'probability': prob, 'source': 'stage2_global'})
        elif kind == 'warning':
            warnings.append({'message': text, 'probability': prob, 'source': 'stage2_global'})
        else:
            facts.append(make_fact(fact_type, target, text, [], target, 'unspecified', prob, 'stage2_global'))

    # Teeth absent from the Stage-1 mask have no crop, so their missing status is emitted directly.
    stage1_missing = sorted(set(str(t) for t in label_obj.get('missing_teeth_from_mask', []) if str(t) in FDI_TEETH), key=FDI_TEETH.index)
    source_missing = []
    for tooth_id in stage1_missing:
        facts.append(make_fact('missing_tooth', f'tooth:{tooth_id}', 'missing tooth', [tooth_id], region_for_tooth(tooth_id), side_for_tooth(tooth_id), None, 'stage1_mask_missing'))

    predicted_missing: List[str] = []
    for tooth_id in FDI_TEETH:
        if tooth_id in stage1_missing:
            continue  # no crop -> no prediction; missing already emitted above
        tooth_item = tooth_pred.get('teeth', {}).get(tooth_id)
        if not tooth_item:
            continue
        for label_item in tooth_item.get('labels', []):
            if not label_item['predicted']:
                continue
            if label_item['label'] == 'missing':
                predicted_missing.append(tooth_id)
            typ, finding = TOOTH_FACT_MAP[label_item['label']]
            facts.append(make_fact(typ, f'tooth:{tooth_id}', finding, [tooth_id], region_for_tooth(tooth_id), side_for_tooth(tooth_id), float(label_item['probability']), 'stage2_tooth'))
    merged_missing = sorted(set(stage1_missing) | set(predicted_missing), key=FDI_TEETH.index)

    for item in relation_pred.get('labels', []):
        if not item['predicted']:
            continue
        target, text = RELATION_TEXT[item['label']]
        teeth = []
        if target.startswith('tooth:'):
            teeth = [target.split(':', 1)[1]]
        side = 'left' if 'left' in item['label'] else 'right' if 'right' in item['label'] else 'unspecified'
        facts.append(make_fact('canal_relationship', target, text, teeth, 'mandibular_canal', side, float(item['probability']), 'stage2_relation'))

    return {
        'case_id': case_id,
        'report_id': report_id,
        'source_report_ids': str(row.get('source_report_ids', '')).split('|') if str(row.get('source_report_ids', '')).strip() else [report_id],
        'model_sources': model_sources or {
            'global': 'EVIDENT_CBCT_stage2_global',
            'tooth': 'EVIDENT_CBCT_stage2_tooth',
            'relation': 'EVIDENT_CBCT_stage2_relation',
        },
        'threshold_policy': threshold_policy or 'per-label tuned thresholds from validation threshold_tuning_checkpoint_best when available',
        'stage1_missing_teeth_from_mask': stage1_missing,
        'merged_missing_teeth': merged_missing,
        'coverage': coverage,
        'facts': facts,
        'normal_negative': [],
        'warnings': warnings,
        'predictions': {
            'global': global_pred,
            'tooth': tooth_pred,
            'relation': relation_pred,
        },
    }


def summarize_teeth(facts: Iterable[Dict[str, Any]], typ: str) -> List[str]:
    out = []
    for fact in facts:
        if fact.get('type') == typ:
            out.extend(str(t) for t in fact.get('teeth', []))
    return sorted(set(out), key=lambda x: FDI_TEETH.index(x) if x in FDI_TEETH else 999)


def report_text(case_obj: Dict[str, Any]) -> str:
    facts = case_obj['facts']
    coverage = case_obj['coverage']
    warnings = case_obj['warnings']
    lines = [f"Case ID: {case_obj['case_id']}", ""]
    if coverage:
        lines.append("Coverage:")
        for item in coverage:
            lines.append(f"- {item['structure']}: {item['status']} (p={item.get('probability', 0):.3f})")
        lines.append("")
    if case_obj['merged_missing_teeth']:
        lines.append("Missing teeth:")
        lines.append("- " + ", ".join(case_obj['merged_missing_teeth']))
        lines.append("")

    tooth_types = ['implant', 'restoration', 'endodontic_treatment', 'prosthesis', 'impaction', 'caries', 'root_remnant', 'root_fracture', 'lesion']
    tooth_lines = []
    for typ in tooth_types:
        teeth = summarize_teeth(facts, typ)
        if teeth:
            name = typ.replace('_', ' ')
            tooth_lines.append(f"- {name}: " + ", ".join(teeth))
    if tooth_lines:
        lines.append("Tooth-level findings:")
        lines.extend(tooth_lines)
        lines.append("")

    global_facts = [f for f in facts if f.get('source') == 'stage2_global']
    if global_facts:
        lines.append("Global findings:")
        for fact in global_facts:
            lines.append(f"- {fact['finding']} (p={fact.get('probability', 0):.3f})")
        lines.append("")

    relation_facts = [f for f in facts if f.get('source') == 'stage2_relation']
    if relation_facts:
        lines.append("Mandibular canal / relation findings:")
        for fact in relation_facts:
            lines.append(f"- {fact['finding']} (p={fact.get('probability', 0):.3f})")
        lines.append("")

    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning['message']} (p={warning.get('probability', 0):.3f})")
        lines.append("")
    if len(lines) <= 2:
        lines.append("No positive Stage-2 evidence predicted.")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Merge independent Stage2 branch validation predictions into case-level JSON and text reports')
    parser.add_argument('--csv-path', type=Path, default=DEFAULT_PREPROCESSED / 'validation_case_merged.csv')
    parser.add_argument('--results-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_ROOT / 'merged_stage2_val_results')
    parser.add_argument('--checkpoint-name', default='checkpoint_best.pth')
    # Optional per-branch checkpoint overrides (default: --checkpoint-name for all three).
    parser.add_argument('--tooth-checkpoint-name', default=None)
    parser.add_argument('--global-checkpoint-name', default=None)
    parser.add_argument('--relation-checkpoint-name', default=None)
    parser.add_argument('--global-plan', default='EVIDENT_CBCT_stage2_global')
    parser.add_argument('--tooth-plan', default='EVIDENT_CBCT_stage2_tooth')
    parser.add_argument('--relation-plan', default='EVIDENT_CBCT_stage2_relation')
    parser.add_argument('--global-folder', type=Path, default=None)
    parser.add_argument('--tooth-folder', type=Path, default=None)
    parser.add_argument('--relation-folder', type=Path, default=None)
    parser.add_argument('--global-thresholds', type=Path, default=None)
    parser.add_argument('--tooth-thresholds', type=Path, default=None)
    parser.add_argument('--relation-thresholds', type=Path, default=None)
    parser.add_argument('--tooth-threshold-view', default='tooth', choices=['tooth', 'tooth_trusted', 'tooth_strict'])
    parser.add_argument('--tooth-region-mask-config', type=Path, default=None, help='Optional JSON config forcing selected tooth-region/label predictions to negative.')
    parser.add_argument('--batch-size-global', type=int, default=1)
    parser.add_argument('--batch-size-tooth', type=int, default=16)
    parser.add_argument('--batch-size-relation', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)

    global_folder = args.global_folder or _default_model_folder(args.results_root, 'global', args.global_plan)
    tooth_folder = args.tooth_folder or _default_model_folder(args.results_root, 'tooth', args.tooth_plan)
    relation_folder = args.relation_folder or _default_model_folder(args.results_root, 'relation', args.relation_plan)
    global_ckpt_name = args.global_checkpoint_name or args.checkpoint_name
    tooth_ckpt_name = args.tooth_checkpoint_name or args.checkpoint_name
    relation_ckpt_name = args.relation_checkpoint_name or args.checkpoint_name
    global_ckpt = global_folder / global_ckpt_name
    tooth_ckpt = tooth_folder / tooth_ckpt_name
    relation_ckpt = relation_folder / relation_ckpt_name
    for path in [args.csv_path, global_ckpt, tooth_ckpt, relation_ckpt]:
        if not path.exists():
            raise FileNotFoundError(path)

    global_threshold_path = args.global_thresholds or _threshold_path(global_folder, global_ckpt_name)
    tooth_threshold_path = args.tooth_thresholds or _threshold_path(tooth_folder, tooth_ckpt_name)
    relation_threshold_path = args.relation_thresholds or _threshold_path(relation_folder, relation_ckpt_name)
    global_thr = _load_thresholds(global_threshold_path, 'global')
    tooth_thr = _load_thresholds(tooth_threshold_path, 'tooth', tooth_threshold_view=args.tooth_threshold_view)
    relation_thr = _load_thresholds(relation_threshold_path, 'relation')
    tooth_region_mask_config = load_tooth_region_mask_config(args.tooth_region_mask_config)

    global_pred = predict_global(args.csv_path, global_ckpt, global_thr, args.batch_size_global, args.num_workers, args.device)
    tooth_pred = predict_tooth(args.csv_path, tooth_ckpt, tooth_thr, args.batch_size_tooth, args.num_workers, args.device, tooth_region_mask_config=tooth_region_mask_config)
    relation_pred = predict_relation(args.csv_path, relation_ckpt, relation_thr, args.batch_size_relation, args.num_workers, args.device)

    df = pd.read_csv(args.csv_path)
    json_dir = args.out_dir / 'merged_json'
    text_dir = args.out_dir / 'reports_txt'
    json_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'csv_path': str(args.csv_path),
        'num_cases': int(len(df)),
        'checkpoint_name': args.checkpoint_name,
        'global_checkpoint': str(global_ckpt),
        'tooth_checkpoint': str(tooth_ckpt),
        'relation_checkpoint': str(relation_ckpt),
        'tooth_threshold_view': args.tooth_threshold_view,
        'global_thresholds': str(global_threshold_path),
        'tooth_thresholds': str(tooth_threshold_path),
        'relation_thresholds': str(relation_threshold_path),
        'tooth_region_mask_config': str(args.tooth_region_mask_config) if args.tooth_region_mask_config else '',
        'model_sources': {
            'global': global_folder.parent.name,
            'tooth': tooth_folder.parent.name,
            'relation': relation_folder.parent.name,
        },
        'cases': [],
    }
    for _, row in df.iterrows():
        rid = str(row['report_id'])
        label_obj = json.loads(Path(row['stage2_label_json']).read_text())
        model_sources = {
            'global': global_folder.parent.name,
            'tooth': tooth_folder.parent.name,
            'relation': relation_folder.parent.name,
        }
        threshold_policy = (
            f'global={global_threshold_path}; tooth={tooth_threshold_path} view={args.tooth_threshold_view}; '
            f"relation={relation_threshold_path}; tooth_region_mask={args.tooth_region_mask_config or 'none'}"
        )
        case_obj = build_case_json(
            row, label_obj, global_pred.get(rid, {}), tooth_pred.get(rid, {}), relation_pred.get(rid, {}),
            model_sources=model_sources, threshold_policy=threshold_policy,
        )
        (json_dir / f'{rid}.json').write_text(json.dumps(case_obj, indent=2) + '\n')
        (text_dir / f'{rid}.txt').write_text(report_text(case_obj))
        summary['cases'].append({
            'report_id': rid,
            'case_id': str(row['case_id']),
            'num_facts': len(case_obj['facts']),
            'num_coverage': len(case_obj['coverage']),
            'num_warnings': len(case_obj['warnings']),
            'merged_missing_teeth': case_obj['merged_missing_teeth'],
        })
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    out = {'out_dir': str(args.out_dir), 'num_cases': len(df), 'json_dir': str(json_dir), 'text_dir': str(text_dir)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
