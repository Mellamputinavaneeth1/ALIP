"""
ALIP — Lookup Service
Loads the precomputed district × season feature table.
Optionally overrides soil pH with farmer-measured value.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LOOKUP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'lookup', 'district_season_features.parquet')

_lookup_cache: pd.DataFrame | None = None


def _load_lookup() -> pd.DataFrame:
    global _lookup_cache
    if _lookup_cache is None:
        if not os.path.exists(LOOKUP_PATH):
            raise FileNotFoundError(
                f"Lookup table not found at {LOOKUP_PATH}. "
                "Run ml/pipeline/run_full_pipeline.py to generate it."
            )
        _lookup_cache = pd.read_parquet(LOOKUP_PATH)
    return _lookup_cache


def get_district_features(district_id: str, season: str, measured_ph: float = None,
                           district_name: str = None) -> dict:
    """
    Returns feature vector for a district + season from precomputed lookup.
    Matches by district_id first, then by name, then by partial name.
    """
    lookup = _load_lookup()
    season_l = season.lower()

    # Try exact district_id match
    mask = (lookup["district_id"] == district_id) & (lookup["season"] == season_l)
    matches = lookup[mask]

    # Try district name match (for Kaggle DES districts vs centroid IDs)
    if matches.empty and district_name:
        name_l = district_name.lower().strip()
        mask2 = (lookup["district_name"].str.lower().str.strip() == name_l) & (lookup["season"] == season_l)
        matches = lookup[mask2]

    # Try partial name match
    if matches.empty and district_name:
        name_l = district_name.lower().strip()
        mask3 = (lookup["district_name"].str.lower().str.contains(name_l, na=False)) & (lookup["season"] == season_l)
        matches = lookup[mask3]
        if len(matches) > 1:
            matches = matches.iloc[:1]  # take first match

    # Try reverse partial (lookup name in search name)
    if matches.empty and district_name:
        name_l = district_name.lower().strip()
        mask4 = lookup["district_name"].str.lower().apply(lambda x: x.strip() in name_l or name_l in x.strip())
        mask4 = mask4 & (lookup["season"] == season_l)
        matches = lookup[mask4]
        if len(matches) > 1:
            matches = matches.iloc[:1]

    if matches.empty:
        return _build_coverage_error(district_id, season)

    row = matches.iloc[0].to_dict()
    coverage = _check_coverage(row)

    ph_source = "soilgrids_regional"
    if measured_ph is not None:
        if 3.0 <= measured_ph <= 9.5:
            row["soil_ph"] = measured_ph
            ph_source = "farmer_measured"
        else:
            ph_source = "invalid_input_fallback"

    return {
        "district_id":               district_id,
        "season":                    season_l,
        "temp_mean":                 float(row.get("temp_mean", 26.0)),
        "rainfall_total":            float(row.get("rainfall_total", 100.0)),
        "rainfall_variability":      float(row.get("rainfall_variability", 0.4)),
        "preceding_monsoon_rainfall":float(row.get("preceding_monsoon_rainfall", 0.0)),
        "soil_ph":                   float(row.get("soil_ph", 6.8)),
        "soil_oc":                   float(row.get("soil_oc", 4.5)),
        "soil_clay":                 float(row.get("soil_clay", 30.0)),
        "soil_cec":                  float(row.get("soil_cec", 20.0)),
        "lagged_irrigated_share":    float(row.get("lagged_irrigated_share", 0.40)),
        "season_kharif":             int(season_l == "kharif"),
        "data_quality_score":        row.get("data_quality_score", "good"),
        "coverage":                  coverage,
        "ph_source":                 ph_source,
        "district_name":             row.get("district_name", district_id),
        "state":                     row.get("state", ""),
        "climate_source":            row.get("climate_source", "open_meteo"),
        "climate_resolution":        "district_centroid",
        "soil_source":               row.get("soil_source", "nbss_state_fallback"),
        "soil_resolution":           row.get("soil_resolution", "state_fallback"),
        "soil_is_fallback":          bool(row.get("soil_is_fallback", True)),
        "irrigation_source":         row.get("irrigation_source", "des_state_level"),
        "irrigation_resolution":     row.get("irrigation_resolution", "state_level"),
        "irrigation_is_fallback":    bool(row.get("irrigation_is_fallback", True)),
    }


def _check_coverage(row: dict) -> dict:
    """Checks which data layers are present."""
    return {
        "has_climate":    pd.notna(row.get("temp_mean")),
        "has_soil":       pd.notna(row.get("soil_ph")),
        "has_irrigation": pd.notna(row.get("lagged_irrigated_share")),
        "complete":       all([
            pd.notna(row.get("temp_mean")),
            pd.notna(row.get("soil_ph")),
            pd.notna(row.get("lagged_irrigated_share")),
        ]),
    }


def _build_coverage_error(district_id: str, season: str) -> dict:
    return {
        "district_id":  district_id, "season": season,
        "error":        "district_not_found",
        "coverage":     {"has_climate": False, "has_soil": False, "has_irrigation": False, "complete": False},
        "data_quality_score": "unavailable",
    }


def list_supported_districts_in_lookup() -> list:
    """Returns list of district_ids present in the lookup table."""
    lookup = _load_lookup()
    return sorted(lookup["district_id"].unique().tolist())


def reload_lookup():
    """Force reload lookup table (call after model update)."""
    global _lookup_cache
    _lookup_cache = None
