#!/usr/bin/env python3
"""Raw-only single-case predictor for the frozen Transverse deployment models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def predict(
    upper_stl: Path,
    lower_stl: Path,
    upper_fdi: np.ndarray,
    lower_fdi: np.ndarray,
    case_id: str,
    *,
    asset_root: Path | str,
) -> dict:
    assets = Path(asset_root).expanduser().resolve()
    config = json.loads((assets / 'config.json').read_text())
    schema = json.loads((assets / 'feature_schema.json').read_text())
    fusion = load_module('transverse_fusion_runtime', CODE_ROOT / '04_evaluate_crossbite_fusion_v3.py')
    state = load_module('transverse_state_runtime', CODE_ROOT / '11_evaluate_transverse_severity_type.py')
    location = load_module('transverse_location_runtime', CODE_ROOT / '05_evaluate_crossbite_location.py')

    with tempfile.TemporaryDirectory(prefix='transverse_live_') as temp:
        temp = Path(temp)
        raw_root, label_root, output = temp / 'raw', temp / 'labels', temp / 'output'
        ios = raw_root / case_id / 'ios'; ios.mkdir(parents=True); label_root.mkdir()
        shutil.copy2(upper_stl, ios / 'ios_upper.stl'); shutil.copy2(lower_stl, ios / 'ios_lower.stl')
        (label_root / f'{case_id}_ios_upper.json').write_text(json.dumps(upper_fdi.tolist()))
        (label_root / f'{case_id}_ios_lower.json').write_text(json.dumps(lower_fdi.tolist()))
        posterior = CODE_ROOT / '01_extract_crossbite_geometry_v3.py'
        anterior = CODE_ROOT / '03_extract_anterior_crossbite_geometry_v2.py'
        subprocess.run([sys.executable, str(posterior), '--raw-root', str(raw_root), '--label-root', str(label_root), '--output-dir', str(output), '--patient-id', case_id], check=True)
        subprocess.run([sys.executable, str(anterior), '--posterior-script', str(posterior), '--raw-root', str(raw_root), '--label-root', str(label_root), '--output-dir', str(output), '--patient-id', case_id], check=True)
        post = fusion.prefix_dataframe(pd.read_csv(output / 'crossbite_geometry_features_v3.csv'), 'post__')
        ant = fusion.prefix_dataframe(pd.read_csv(output / 'anterior_crossbite_geometry_features_v2.csv'), 'ant__')
        merged = post.merge(ant, on='patient_id', validate='one_to_one')
        # Per-case CSVs omit all-null columns.  The training cohort retained
        # them, so restore those columns as NaN before frozen feature building.
        for tooth in (13, 12, 11, 21, 22, 23):
            for statistic in ('q20_margin_mm', 'median_margin_mm', 'negative_ratio'):
                column = f'ant__anterior_radial_tooth{tooth}_incisal_{statistic}'
                if column not in merged:
                    merged[column] = np.nan
        # Training selects columns on the cohort (>=5 non-null, >=2 values).
        # At single-case inference, reindex to that frozen schema; never apply
        # the cohort-dependent safe_numeric filter to one row.
        ant_raw_df, _, _ = fusion.build_anterior_raw_features(merged)
        X = pd.concat([
            merged.reindex(columns=schema['posterior_columns']).reset_index(drop=True),
            ant_raw_df.reindex(columns=schema['anterior_columns']).reset_index(drop=True),
        ], axis=1).reindex(columns=schema['concat_columns'])
        if list(X.columns) != schema['concat_columns']:
            raise RuntimeError('Frozen Transverse concat feature order mismatch')
        models = assets / 'models'
        load_started = time.perf_counter()
        severity_model = joblib.load(models / 'severity_state_logistic_97.joblib')
        type_model = joblib.load(models / 'type_state_logistic_97.joblib')
        present_model = joblib.load(models / 'crossbite_present_concat_raw_reference.joblib')
        side_model = joblib.load(models / 'crossbite_location_side_shared_raw4.joblib')
        print(f"B2T_MODEL_LOAD transverse count=4 seconds={time.perf_counter() - load_started:.6f}", flush=True)
        severity_state = severity_model.predict(X)[0]
        type_state = type_model.predict(X)[0]
        present_prob = float(present_model.predict_proba(X)[0, 1])
        present = present_prob >= float(config['crossbite_present']['threshold'])
        # build_side_rows only consumes geometry columns and this label is target-only.
        merged['gt_crossbite_location'] = 'None'
        _, _, ant_arrays = location.build_anterior_45(merged)
        side_X, _, _ = location.build_side_rows(merged, ant_arrays, np.asarray([0]))
        side_X = side_X.reindex(columns=schema['side_shared_columns'])
        side_probs = side_model.predict_proba(side_X)[:, 1]
        raw_location = location.decode_raw4_side(
            np.asarray([side_probs[0]]), np.asarray([side_probs[1]]), float(config['crossbite_location']['bilateral_scale'])
        )[0]
        return {
            'severity': state.state_to_raw('severity', severity_state),
            'type': state.state_to_raw('type', type_state),
            'crossbite_present': bool(present),
            'crossbite_type': 'Crossbite' if present else 'None',
            'crossbite_location': str(raw_location),
            'crossbite_teeth': [],
            '_debug': {'feature_columns': list(X.columns), 'present_probability': present_prob},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper-stl', required=True, type=Path)
    parser.add_argument('--lower-stl', required=True, type=Path)
    parser.add_argument('--upper-fdi-json', required=True, type=Path)
    parser.add_argument('--lower-fdi-json', required=True, type=Path)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.upper_stl, args.lower_stl, np.asarray(json.loads(args.upper_fdi_json.read_text())), np.asarray(json.loads(args.lower_fdi_json.read_text())), args.case_id)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
