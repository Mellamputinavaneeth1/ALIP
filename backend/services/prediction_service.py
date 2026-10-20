"""
ALIP — Prediction Service
Loads the trained crop ranking model and scores all candidate crops
for a given district × season feature vector.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.data.district_centroids import ALL_CROPS, get_allowed_crops
from ml.data.build_training_table import FEATURE_COLS

V21_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'regional_v2.1')
LEGACY_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'legacy')
BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_model      = None
_calibrator = None


def _load_model():
    global _model, _calibrator
    if _model is None:
        model_paths = [
            os.path.join(V21_DIR, 'crop_model_regional.pkl'),
            os.path.join(BASE_MODELS_DIR, 'crop_model_regional.pkl'),
            os.path.join(LEGACY_DIR, 'crop_model_regional.pkl'),
        ]
        for path in model_paths:
            if os.path.exists(path):
                _model = joblib.load(path)
                break
        if _model is None:
            raise FileNotFoundError(
                f"Regional crop model not found in {V21_DIR} or fallback locations. "
                "Run ml/pipeline/run_full_pipeline.py first."
            )

    if _calibrator is None:
        cal_paths = [
            os.path.join(V21_DIR, 'calibrator.pkl'),
            os.path.join(BASE_MODELS_DIR, 'calibrator.pkl'),
            os.path.join(LEGACY_DIR, 'calibrator.pkl'),
        ]
        for path in cal_paths:
            if os.path.exists(path):
                _calibrator = joblib.load(path)
                break

    return _model, _calibrator


def score_candidate_crops(features: dict, district_id: str, season: str, all_crops: bool = True) -> list:
    """
    Scores all candidate field crops for this district x season.
    Returns list of {crop, score, raw_probability} sorted descending.
    """
    model, calibrator = _load_model()

    allowed = ALL_CROPS if all_crops else (get_allowed_crops(district_id, season) or ALL_CROPS)

    results = []
    for crop in allowed:
        vec = _build_vector(features, crop)
        vec_df = pd.DataFrame([vec], columns=FEATURE_COLS, dtype=float)
        raw_prob = float(model.predict_proba(vec_df)[0][1])   # P(is_major)

        # Apply calibration if available
        if calibrator is not None:
            cal_pred = calibrator.predict([raw_prob])[0]
            cal_prob = float(np.clip(cal_pred, 0.0, 1.0))
        else:
            cal_prob = raw_prob

        results.append({
            "crop":          crop,
            "score":         round(cal_prob, 4),
            "raw_probability": round(raw_prob, 4),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def _build_vector(features: dict, crop: str) -> list:
    """Assembles feature vector in the exact training order with float dtype."""
    vec = []
    for col in FEATURE_COLS:
        if col == "temp_mean": vec.append(float(features.get("temp_mean") if features.get("temp_mean") is not None else 26.0))
        elif col == "rainfall_total": vec.append(float(features.get("rainfall_total") if features.get("rainfall_total") is not None else 100.0))
        elif col == "rainfall_variability": vec.append(float(features.get("rainfall_variability") if features.get("rainfall_variability") is not None else 0.4))
        elif col == "preceding_monsoon_rainfall": vec.append(float(features.get("preceding_monsoon_rainfall") if features.get("preceding_monsoon_rainfall") is not None else 0.0))
        elif col == "soil_ph": vec.append(float(features.get("soil_ph") if features.get("soil_ph") is not None else 6.8))
        elif col == "soil_oc": vec.append(float(features.get("soil_oc") if features.get("soil_oc") is not None else 4.5))
        elif col == "soil_clay": vec.append(float(features.get("soil_clay") if features.get("soil_clay") is not None else 30.0))
        elif col == "soil_cec": vec.append(float(features.get("soil_cec") if features.get("soil_cec") is not None else 20.0))
        elif col == "lagged_irrigated_share": vec.append(float(features.get("lagged_irrigated_share") if features.get("lagged_irrigated_share") is not None else 0.4))
        elif col == "season_kharif": vec.append(float(features.get("season_kharif", 1) or 0))
        elif col.startswith("crop_"):
            vec.append(1.0 if col == f"crop_{crop}" else 0.0)
        else:
            vec.append(0.0)
    return vec
