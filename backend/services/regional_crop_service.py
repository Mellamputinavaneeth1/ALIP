"""
ALIP — Regional Crop Service
Orchestrates the full regional recommendation pipeline:
  district resolver → feature lookup → scoring → guardrails → ranking → explanation
"""

import os
import sys
import logging
import json as json_module

logger = logging.getLogger("alip.regional")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

V21_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'regional_v2.1')
LEGACY_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'legacy')
BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_regional_model = None
_hybrid_config = None
_prior_table = None


def _load_regional_model():
    global _regional_model
    if _regional_model is None:
        import joblib
        model_paths = [
            os.path.join(V21_DIR, 'crop_model_regional.pkl'),
            os.path.join(BASE_MODELS_DIR, 'crop_model_regional.pkl'),
            os.path.join(LEGACY_DIR, 'crop_model_regional.pkl'),
        ]
        for path in model_paths:
            if os.path.exists(path):
                _regional_model = joblib.load(path)
                break
    return _regional_model


def _load_hybrid_artifacts():
    global _hybrid_config, _prior_table
    import pandas as pd
    import json
    if _hybrid_config is None:
        cfg_paths = [
            os.path.join(V21_DIR, 'hybrid_config.json'),
            os.path.join(BASE_MODELS_DIR, 'hybrid_config.json'),
        ]
        for path in cfg_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    _hybrid_config = json.load(f)
                break
        if _hybrid_config is None:
            _hybrid_config = {"alpha": 0.5, "model_version": "v2.1-regional-hybrid"}

    if _prior_table is None:
        prior_paths = [
            os.path.join(V21_DIR, 'district_season_prior.parquet'),
            os.path.join(BASE_MODELS_DIR, 'district_season_prior.parquet'),
        ]
        for path in prior_paths:
            if os.path.exists(path):
                _prior_table = pd.read_parquet(path)
                break
    return _hybrid_config, _prior_table


async def recommend_regional(
    location: str = None,
    latitude: float = None,
    longitude: float = None,
    season: str = "kharif",
    measured_ph: float = None,
) -> dict:
    """
    Full regional crop recommendation pipeline.
    Returns structured recommendation response.
    """
    trace = {"steps": []}

    # ── Step 1: Resolve district ──────────────────────────────────────────────
    from services.district_service import resolve_district
    district_info = resolve_district(location=location, lat=latitude, lon=longitude)
    trace["steps"].append({
        "step": "district_resolution",
        "found": district_info.get("found", False),
        "method": district_info.get("method", "name_lookup")
    })

    if not district_info["found"]:
        return {
            "success": False,
            "error": f"District not found for location '{location}'. "
                     "Use GET /crop/districts to see supported locations.",
            "abstain": True,
        }

    district_id   = district_info["district_id"]
    district_name = district_info["district_name"]
    state         = district_info["state"]

    # ── Step 2: Load precomputed features ─────────────────────────────────────
    try:
        from services.lookup_service import get_district_features
        features = get_district_features(
            district_id, season,
            measured_ph=measured_ph,
            district_name=district_name
        )

        if features.get("error"):
            return {
                "success": False,
                "error": f"Feature data not available for {district_name}. "
                         "Run the full pipeline to generate lookup data.",
                "abstain": True,
            }
    except FileNotFoundError:
        # Lookup table not generated yet — use real-time fallback
        features = _build_realtime_features(district_info, season, measured_ph)

    trace["steps"].append({
        "step": "feature_lookup",
        "district_id": district_id
    })

    # ── Step 3: Score candidate crops (Hybrid Model) ──────────────────────────
    model = _load_regional_model()
    if model is None:
        return {
            "success": False,
            "error": "Regional model not trained yet. Run ml/pipeline/run_full_pipeline.py.",
            "abstain": True,
        }

    from services.prediction_service import score_candidate_crops
    ml_candidates = score_candidate_crops(features, district_id, season)
    trace["steps"].append({
        "step": "ml_scoring",
        "candidates": len(ml_candidates)
    })

    hybrid_cfg, prior_df = _load_hybrid_artifacts()
    alpha = float(hybrid_cfg.get("alpha", 0.5)) if hybrid_cfg else 0.5

    ranked = []
    for item in ml_candidates:
        crop = item["crop"]
        ml_score = item["score"]
        prior_prob = 0.0

        if prior_df is not None:
            m = prior_df[
                (prior_df["district_id"] == district_id) &
                (prior_df["season"] == season.lower()) &
                (prior_df["crop"] == crop)
            ]
            if not m.empty:
                prior_prob = float(m.iloc[0]["prior_prob"])
            else:
                m_crop = prior_df[prior_df["crop"] == crop]
                prior_prob = float(m_crop["prior_prob"].mean()) if not m_crop.empty else 0.0

        hybrid_score = round(alpha * ml_score + (1.0 - alpha) * prior_prob, 4)
        ranked.append({
            "crop": crop,
            "score": hybrid_score,
            "raw_probability": item.get("raw_probability", ml_score),
            "ml_probability": ml_score,
            "prior_probability": round(prior_prob, 4),
        })

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    trace["steps"].append({
        "step": "hybrid_ranking",
        "alpha": alpha,
        "top_score": ranked[0]["score"] if ranked else 0
    })

    # ── Step 4: Guardrails + Ranking ──────────────────────────────────────────
    from services.guardrail_service import rank_and_top3
    top3, flags, abstain, abstain_reason = rank_and_top3(ranked, features, season)
    trace["steps"].append({
        "step": "guardrails",
        "flags_count": len(flags),
        "abstain": abstain
    })

    # ── Step 5: SHAP Explanations ─────────────────────────────────────────────
    from services.prediction_service import _build_vector
    from services.explanation_service import generate_reasons

    recommendations = []
    for rank, item in enumerate(top3, start=1):
        crop    = item["crop"]
        vec     = _build_vector(features, crop)

        try:
            reasons = generate_reasons(model, vec, top_n=3)
        except Exception:
            reasons = [
                "Based on historical regional crop patterns",
                "Soil and climate conditions are compatible",
                "Irrigation availability supports this crop",
            ]

        recommendations.append({
            "rank":                    rank,
            "crop":                    crop,
            "ml_probability":          item.get("ml_probability", 0.0),
            "prior_score":             item.get("prior_probability", 0.0),
            "hybrid_rank_score":       item["score"],
            "confidence_band":         item.get("confidence_band", "medium"),
            "guardrail_status":        item.get("guardrail_status", "pass" if not item.get("guardrail_warnings") else "flagged"),
            "guardrail_warnings":      item.get("guardrail_warnings", []),
            "reasons":                 reasons,
        })

    # ── Step 6: Build response ─────────────────────────────────────────────────
    model_version = hybrid_cfg.get("model_version", "v2.1-regional-hybrid") if hybrid_cfg else "v2.1-regional-hybrid"
    response_dict = {
        "success":          True,
        "district":         district_name,
        "district_id":      district_id,
        "state":            state,
        "season":           season,
        "model_version":    model_version,
        "provenance": {
            "crop_data": {
                "source": "Kaggle Crop Production in India (DES)",
                "version": "2023-kaggle-upload",
            },
            "climate": {
                "source": features.get("climate_source", "open_meteo"),
                "resolution": features.get("climate_resolution", "district_centroid"),
                "window": "10-year trailing normals [Y-10, Y-1]",
            },
            "soil": {
                "source": features.get("soil_source", "nbss_state_fallback"),
                "resolution": features.get("soil_resolution", "state_fallback"),
                "fallback": features.get("soil_is_fallback", True),
            },
            "irrigation": {
                "source": features.get("irrigation_source", "des_state_level"),
                "resolution": features.get("irrigation_resolution", "state_level"),
            },
            "boundary": {
                "version": "kaggle_des_district_names",
            },
            "model_version": model_version,
            "hybrid_alpha": alpha,
            "dataset_version": "v2.1",
            "feature_schema_version": "25-dim",
        },
        "data_quality":     features.get("data_quality_score", "good"),
        "ph_source":        features.get("ph_source", "soilgrids_regional"),
        "recommendations":  recommendations,
        "all_candidates":   ranked,
        "flags":            flags,
        "abstain":          abstain,
        "abstain_reason":   abstain_reason if abstain else None,
        "context": {
            "temp_mean":             features.get("temp_mean"),
            "rainfall_total":        features.get("rainfall_total"),
            "rainfall_variability":  features.get("rainfall_variability"),
            "soil_ph":               features.get("soil_ph"),
            "soil_oc":               features.get("soil_oc"),
            "soil_clay":             features.get("soil_clay"),
            "soil_cec":              features.get("soil_cec"),
            "irrigated_share":       features.get("lagged_irrigated_share"),
        },
    }

    logger.debug("Regional recommendation trace: %s", json_module.dumps(trace, default=str))
    if os.environ.get("ALIP_DEBUG_TRACE"):
        response_dict["_trace"] = trace

    return response_dict


def _build_realtime_features(district_info: dict, season: str, measured_ph: float = None) -> dict:
    """
    Fallback when lookup table is not yet generated.
    Uses soil_data.py + climate_data.py (existing ALIP data layer).
    """
    from data.soil_data import get_soil_data, get_rainfall
    from data.climate_data import get_climate_normals

    location = district_info["district_name"]
    soil     = get_soil_data(location)
    climate  = get_climate_normals(location, season)
    rainfall = get_rainfall(season)

    soil_ph = measured_ph if (measured_ph and 3.0 <= measured_ph <= 9.5) else soil.get("ph", 6.8)

    return {
        "temp_mean":                climate.get("avg_temperature", 26.0),
        "rainfall_total":           climate.get("seasonal_rainfall", rainfall),
        "rainfall_variability":     0.4,
        "preceding_monsoon_rainfall": climate.get("seasonal_rainfall", 80) * 2.0 if season == "rabi" else 0.0,
        "soil_ph":                  soil_ph,
        "soil_oc":                  4.5,
        "soil_clay":                30.0,
        "soil_cec":                 20.0,
        "lagged_irrigated_share":   0.40,
        "season_kharif":            int(season == "kharif"),
        "data_quality_score":       "fallback",
        "ph_source":                "measured" if measured_ph else "soildata_module",
        "coverage": {
            "has_climate": True, "has_soil": True,
            "has_irrigation": False, "complete": False,
        },
    }
