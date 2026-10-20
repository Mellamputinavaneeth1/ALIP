"""
ALIP — Explanation Service
Translates model SHAP values into farmer-readable reason strings.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Human-readable templates per feature direction ─────────────────────────────
SHAP_TEMPLATES = {
    "temp_mean": {
        "positive": "Seasonal temperature is favorable for this crop",
        "negative": "Temperature may be outside the optimal range for this crop",
    },
    "rainfall_total": {
        "positive": "Seasonal rainfall supports this crop's water needs",
        "negative": "Rainfall may be insufficient for reliable cultivation",
    },
    "rainfall_variability": {
        "positive": "Rainfall pattern is consistent and predictable",
        "negative": "Erratic rainfall increases cultivation risk for this crop",
    },
    "preceding_monsoon_rainfall": {
        "positive": "Strong monsoon provides good soil moisture for the Rabi season",
        "negative": "Low preceding monsoon may reduce soil moisture reserves",
    },
    "soil_ph": {
        "positive": "Soil pH is compatible with this crop's requirements",
        "negative": "Soil pH may not be ideal for this crop",
    },
    "soil_oc": {
        "positive": "Soil organic matter supports good crop nutrition",
        "negative": "Low organic matter may limit yield potential",
    },
    "soil_clay": {
        "positive": "Soil texture suits this crop's root system",
        "negative": "Soil texture may not be ideal for this crop",
    },
    "soil_cec": {
        "positive": "Good soil nutrient-retention capacity for this crop",
        "negative": "Limited soil nutrient retention may reduce fertilizer efficiency",
    },
    "lagged_irrigated_share": {
        "positive": "Irrigation availability in this region supports cultivation",
        "negative": "Limited irrigation may restrict suitable crop options",
    },
    "season_kharif": {
        "positive": "This crop is well-suited to the growing season",
        "negative": "This crop is not typically grown in this season",
    },
}

from ml.data.build_training_table import FEATURE_COLS

_shap_explainer = None
_model_ref      = None


def _get_explainer(model):
    global _shap_explainer, _model_ref
    if _shap_explainer is None or model is not _model_ref:
        import shap
        _shap_explainer = shap.TreeExplainer(model)
        _model_ref = model
    return _shap_explainer


def generate_reasons(model, feature_vector: list, top_n: int = 3) -> list:
    """
    Generates top_n farmer-readable reason strings using SHAP values.
    feature_vector: list in FEATURE_COLS order.
    Returns: list of reason strings.
    """
    try:
        import shap
        explainer = _get_explainer(model)
        vec_array = np.array([feature_vector])
        shap_vals = explainer.shap_values(vec_array)

        # shap_values returns list[class_0, class_1] for binary classifier
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]   # SHAP for positive class (is_major=1)
        else:
            vals = shap_vals[0]

        # Map feature names (exclude crop_* from explanations)
        named = [
            (FEATURE_COLS[i], float(vals[i]))
            for i in range(len(vals))
            if not FEATURE_COLS[i].startswith("crop_") and i < len(vals)
        ]

        # Sort by absolute SHAP value, take top_n
        named.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = named[:top_n]

        reasons = []
        for feat_name, shap_val in top_features:
            templates = SHAP_TEMPLATES.get(feat_name)
            if templates:
                direction = "positive" if shap_val > 0 else "negative"
                reasons.append(templates[direction])
            else:
                direction = "favorable" if shap_val > 0 else "unfavorable"
                reasons.append(f"{feat_name.replace('_', ' ').title()} is {direction}")

        return reasons

    except Exception:
        # Fallback: rule-based reasons if SHAP fails
        return _rule_based_reasons(feature_vector)


def _rule_based_reasons(feature_vector: list) -> list:
    """Fallback when SHAP is unavailable."""
    reasons = []
    feat = dict(zip(FEATURE_COLS, feature_vector))

    rain = feat.get("rainfall_total", 100)
    if rain >= 80:
        reasons.append(SHAP_TEMPLATES["rainfall_total"]["positive"])
    else:
        reasons.append(SHAP_TEMPLATES["rainfall_total"]["negative"])

    ph = feat.get("soil_ph", 6.8)
    if 5.5 <= ph <= 8.0:
        reasons.append(SHAP_TEMPLATES["soil_ph"]["positive"])
    else:
        reasons.append(SHAP_TEMPLATES["soil_ph"]["negative"])

    irr = feat.get("lagged_irrigated_share", 0.4)
    if irr >= 0.4:
        reasons.append(SHAP_TEMPLATES["lagged_irrigated_share"]["positive"])
    else:
        reasons.append(SHAP_TEMPLATES["lagged_irrigated_share"]["negative"])

    return reasons[:3]
