"""
ALIP — Soil Feature Builder
Fetches soil properties from SoilGrids REST API (ISRIC).
Properties: pH (phh2o), Organic Carbon (soc), Clay (clay), CEC (cec).
Depth: 0-30cm (average of 0-5cm, 5-15cm, 15-30cm layers).
Soil features are STATIC (fetched once per district).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.data.district_centroids import DISTRICT_CENTROIDS

BASE_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "raw", "soil_cache.json")
OUT_PATH   = os.path.join(BASE_DIR, "processed", "soil_features.parquet")

SOILGRIDS_URL  = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPS     = ["phh2o", "soc", "clay", "cec"]
DEPTHS         = ["0-5cm", "5-15cm", "15-30cm"]

# Unit conversion factors (SoilGrids stores in scaled integer units)
UNIT_SCALE = {
    "phh2o": 0.1,    # pHx10 → pH
    "soc":   0.1,    # cg/kg → g/kg (dg/kg ×0.1)
    "clay":  0.1,    # g/kg → %
    "cec":   0.1,    # mmolc/kg × 0.1 → cmolc/kg
}

# Fallback regional values (if SoilGrids API is unavailable)
# Based on NBSS&LUP regional soil survey averages
FALLBACK_SOIL = {
    "Punjab":          {"soil_ph": 7.8, "soil_oc": 4.5, "soil_clay": 22.0, "soil_cec": 18.0},
    "Haryana":         {"soil_ph": 7.9, "soil_oc": 3.8, "soil_clay": 20.0, "soil_cec": 17.0},
    "Uttar Pradesh":   {"soil_ph": 7.5, "soil_oc": 5.0, "soil_clay": 25.0, "soil_cec": 20.0},
    "Bihar":           {"soil_ph": 7.2, "soil_oc": 6.0, "soil_clay": 28.0, "soil_cec": 22.0},
    "West Bengal":     {"soil_ph": 6.2, "soil_oc": 8.0, "soil_clay": 35.0, "soil_cec": 25.0},
    "Madhya Pradesh":  {"soil_ph": 7.1, "soil_oc": 4.2, "soil_clay": 40.0, "soil_cec": 28.0},
    "Maharashtra":     {"soil_ph": 7.3, "soil_oc": 4.0, "soil_clay": 45.0, "soil_cec": 30.0},
    "Telangana":       {"soil_ph": 6.5, "soil_oc": 3.8, "soil_clay": 38.0, "soil_cec": 22.0},
    "Andhra Pradesh":  {"soil_ph": 6.8, "soil_oc": 4.0, "soil_clay": 35.0, "soil_cec": 20.0},
    "Karnataka":       {"soil_ph": 6.2, "soil_oc": 5.0, "soil_clay": 40.0, "soil_cec": 24.0},
    "Rajasthan":       {"soil_ph": 8.1, "soil_oc": 2.5, "soil_clay": 12.0, "soil_cec": 12.0},
    "Gujarat":         {"soil_ph": 7.6, "soil_oc": 3.5, "soil_clay": 30.0, "soil_cec": 20.0},
    "Odisha":          {"soil_ph": 5.8, "soil_oc": 7.0, "soil_clay": 38.0, "soil_cec": 22.0},
    "Tamil Nadu":      {"soil_ph": 6.5, "soil_oc": 5.5, "soil_clay": 35.0, "soil_cec": 20.0},
    "default":         {"soil_ph": 6.8, "soil_oc": 4.5, "soil_clay": 30.0, "soil_cec": 20.0},
}


def _fetch_soilgrids(lat: float, lon: float) -> dict | None:
    """Fetches soil properties from SoilGrids REST API for a single point."""
    try:
        params = {
            "lat": lat, "lon": lon,
            "property": SOIL_PROPS,
            "depth": DEPTHS,
            "value": "mean",
        }
        r = requests.get(SOILGRIDS_URL, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()

        results = {}
        for prop_entry in data.get("properties", {}).get("layers", []):
            prop_name = prop_entry.get("name")
            if prop_name not in SOIL_PROPS:
                continue
            depths_data = prop_entry.get("depths", [])
            depth_values = []
            for depth in depths_data:
                v = depth.get("values", {}).get("mean")
                if v is not None:
                    scale = UNIT_SCALE.get(prop_name, 1.0)
                    depth_values.append(v * scale)
            if depth_values:
                results[prop_name] = float(np.mean(depth_values))

        if len(results) >= 3:
            return {
                "soil_ph":   round(results.get("phh2o", 6.8), 2),
                "soil_oc":   round(results.get("soc",   4.5), 2),
                "soil_clay": round(results.get("clay",  30.0), 1),
                "soil_cec":  round(results.get("cec",   20.0), 1),
                "data_source": "soilgrids_api",
            }
        return None
    except Exception as e:
        print(f"    [soil] SoilGrids error: {e}")
        return None


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)


def build_soil_features(test_districts: list = None, verbose: bool = True) -> pd.DataFrame:
    """Fetches SoilGrids data for all districts. Uses cache and fallback."""
    cache  = _load_cache()
    districts = test_districts or list(DISTRICT_CENTROIDS.keys())
    rows = []

    for i, did in enumerate(districts):
        meta = DISTRICT_CENTROIDS[did]
        lat, lon = meta["lat"], meta["lon"]
        state = meta["state"]

        if verbose:
            print(f"  [{i+1}/{len(districts)}] {meta['name']}", end=" ... ")

        if did in cache:
            soil = cache[did]
            if verbose:
                print(f"cached (pH={soil['soil_ph']})")
        else:
            soil = _fetch_soilgrids(lat, lon)
            if soil:
                cache[did] = soil
                _save_cache(cache)
                if verbose:
                    print(f"API (pH={soil['soil_ph']})")
                time.sleep(0.5)
            else:
                # Fallback to regional averages
                fb = FALLBACK_SOIL.get(state, FALLBACK_SOIL["default"]).copy()
                fb["data_source"] = "nbss_state_fallback"
                cache[did] = fb
                _save_cache(cache)
                soil = fb
                if verbose:
                    print(f"fallback (pH={soil['soil_ph']})")

        is_fb = soil.get("data_source") != "soilgrids_api"
        rows.append({
            "district_id": did,
            "district_name": meta["name"],
            "state": state,
            **{k: v for k, v in soil.items() if k != "data_source"},
            "soil_source": soil.get("data_source", "unknown"),
            "soil_is_fallback": is_fb,
            "soil_resolution": "district" if not is_fb else "state_fallback",
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    if verbose:
        print(f"\n[soil] Saved: {OUT_PATH}")
        print(f"  Shape: {df.shape}")
        print(f"  pH range: {df['soil_ph'].min():.1f} – {df['soil_ph'].max():.1f}")
        print(f"  OC range: {df['soil_oc'].min():.1f} – {df['soil_oc'].max():.1f}")
        api_count = (df['soil_source'] == 'soilgrids_api').sum()
        print(f"  From SoilGrids API: {api_count}/{len(df)}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        build_soil_features(test_districts=["MH001", "PB001", "TS001", "RJ001", "KA001"], verbose=True)
    else:
        build_soil_features(verbose=True)
