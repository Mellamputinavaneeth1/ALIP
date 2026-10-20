"""
ALIP — Training Table Builder
Joins crop outcomes + climate + soil + irrigation into the final
district_crop_training.parquet used for model training.

Also produces data_quality_report.json summarising coverage, missingness, etc.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.data.district_centroids import ALL_CROPS

BASE_DIR       = os.path.dirname(__file__)
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")
FEATURES_DIR   = os.path.join(BASE_DIR, "features")

OUT_PATH       = os.path.join(FEATURES_DIR, "district_crop_training.parquet")
QUALITY_PATH   = os.path.join(FEATURES_DIR, "data_quality_report.json")

# Feature columns used for ML
ENVIRONMENTAL_COLS = [
    "temp_mean", "rainfall_total", "rainfall_variability",
    "preceding_monsoon_rainfall",
    "soil_ph", "soil_oc", "soil_clay", "soil_cec",
    "lagged_irrigated_share",
    "season_kharif",   # 1=kharif, 0=rabi
]
CROP_COLS = [f"crop_{c}" for c in sorted(ALL_CROPS)]
FEATURE_COLS = ENVIRONMENTAL_COLS + CROP_COLS

LABEL_COL       = "is_major"
AREA_SHARE_COL  = "crop_area_share"


def build_training_table(verbose: bool = True) -> pd.DataFrame:
    """Joins all feature layers and validates the result."""

    # ── Load layers ───────────────────────────────────────────────────────────
    crops_path    = os.path.join(PROCESSED_DIR, "crop_outcomes.parquet")
    climate_path  = os.path.join(PROCESSED_DIR, "climate_features.parquet")
    soil_path     = os.path.join(PROCESSED_DIR, "soil_features.parquet")
    irr_path      = os.path.join(PROCESSED_DIR, "irrigation_features.parquet")

    for p in [crops_path, climate_path, soil_path, irr_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run build_* scripts first.")

    if verbose:
        print("[training_table] Loading data layers...")

    crops   = pd.read_parquet(crops_path)
    climate = pd.read_parquet(climate_path)
    soil    = pd.read_parquet(soil_path)
    irr     = pd.read_parquet(irr_path)

    if verbose:
        print(f"  Crop outcomes: {crops.shape}")
        print(f"  Climate:       {climate.shape}")
        print(f"  Soil:          {soil.shape}")
        print(f"  Irrigation:    {irr.shape}")

    # ── Keep only observed positive + observed zero + synthetic absence ──────
    crops = crops[crops["record_type"].isin(["observed_positive", "observed_zero", "synthetic_absence"])].copy()
    crops = crops[crops["year"] != 2015].copy()

    # ── Build Kaggle DES → climate district concordance ──────────────────────────
    from ml.data.district_centroids import DISTRICT_CENTROIDS
    climate_districts_meta = {
        did: DISTRICT_CENTROIDS[did]
        for did in climate["district_id"].unique()
        if did in DISTRICT_CENTROIDS
    }

    def _normalize_name(s: str) -> str:
        if not isinstance(s, str): return ""
        return s.strip().upper().replace(' ','_').replace('-','_')

    direct_lookup = {}
    fallback_lookup = {}
    for did, meta in climate_districts_meta.items():
        s_norm = _normalize_name(meta["state"])
        d_norm = _normalize_name(meta["name"])
        direct_lookup[(s_norm, d_norm)] = did
        if s_norm not in fallback_lookup:
            fallback_lookup[s_norm] = did

    climate_direct_match = 0
    climate_state_fallback = 0
    climate_missing = 0

    def _get_climate_id(row):
        nonlocal climate_direct_match, climate_state_fallback, climate_missing
        s_norm = _normalize_name(row["state"])
        d_norm = _normalize_name(row["district_name"])
        if (s_norm, d_norm) in direct_lookup:
            climate_direct_match += 1
            return direct_lookup[(s_norm, d_norm)]
        elif s_norm in fallback_lookup:
            climate_state_fallback += 1
            return fallback_lookup[s_norm]
        else:
            climate_missing += 1
            return None

    crops["climate_district_id"] = crops.apply(_get_climate_id, axis=1)
    
    # Exclude districts with no climate data
    crops = crops[crops["climate_district_id"].notna()].copy()

    if verbose:
        print(f"  Climate direct matches: {climate_direct_match}")
        print(f"  Climate state fallbacks: {climate_state_fallback}")
        print(f"  Climate missing (dropped): {climate_missing}")

    # ── Join climate ──────────────────────────────────────────────────────────
    climate_cols = ["district_id", "target_year", "season",
                    "temp_mean", "rainfall_total", "rainfall_variability",
                    "preceding_monsoon_rainfall"]
    for c_meta in ["climate_source", "climate_window_start", "climate_window_end", "window_years"]:
        if c_meta in climate.columns:
            climate_cols.append(c_meta)

    df = crops.merge(
        climate[climate_cols].rename(columns={
            "district_id": "climate_district_id",
            "target_year": "year"
        }),
        on=["climate_district_id", "year", "season"],
        how="left"
    )
    df = df.drop(columns=["climate_district_id"], errors="ignore")
    if "climate_source" not in df.columns:
        df["climate_source"] = "open_meteo"
    else:
        df["climate_source"] = df["climate_source"].fillna("open_meteo")

    # Note: Kaggle DES years 1997-2015, climate available 2005-2022.
    # For Kaggle DES years < 2005, climate features will be NaN → imputed below.
    # For serving, we use latest available climate (2022).

    # ── Join soil by state ────────────────────────────────────────────────────
    # soil parquet has district_id from our centroids (63 districts).
    # Map Kaggle DES state → centroid district_id via state_to_climate_id,
    # then join soil on that centroid district_id.
    soil_cols = ["district_id", "soil_ph", "soil_oc", "soil_clay", "soil_cec", "state"]
    # Create state-level soil averages (one value per state)
    soil_with_state = soil[soil["state"].notna()].copy() if "state" in soil.columns else soil.copy()
    # Add state info from DISTRICT_CENTROIDS
    from ml.data.district_centroids import DISTRICT_CENTROIDS
    soil_with_state = soil.copy()
    soil_with_state["centroid_state"] = soil_with_state["district_id"].map(
        {did: meta["state"] for did, meta in DISTRICT_CENTROIDS.items()}
    )
    # Aggregate to state-level median soil
    state_soil = soil_with_state.groupby("centroid_state")[
        ["soil_ph", "soil_oc", "soil_clay", "soil_cec"]
    ].median().reset_index()
    state_soil = state_soil.rename(columns={"centroid_state": "soil_state"})

    # Build Kaggle DES state → soil_state mapping
    def _match_soil_state(kaggle_des_state):
        s = kaggle_des_state.lower().strip()
        for ss in state_soil["soil_state"].tolist():
            if ss.lower() == s or ss.lower() in s or s in ss.lower():
                return ss
        return state_soil["soil_state"].iloc[0]  # national fallback

    df["soil_state"] = df["state"].apply(_match_soil_state)
    df = df.merge(state_soil, on="soil_state", how="left")
    df = df.drop(columns=["soil_state"], errors="ignore")
    df["soil_source"] = "nbss_state_fallback"
    df["soil_resolution"] = "state_fallback"
    df["soil_is_fallback"] = True

    # ── Join irrigation by state ───────────────────────────────────────────────
    # irrigation parquet has district_id from centroids + target_year.
    # Aggregate to state-level, then join by state+year.
    irr_with_state = irr.copy()
    irr_with_state["irr_state"] = irr_with_state["state"]
    state_irr = irr_with_state.groupby(["irr_state", "target_year"])[
        "lagged_irrigated_share"
    ].mean().reset_index()
    state_irr = state_irr.rename(columns={
        "irr_state": "irr_match_state", "target_year": "year"
    })

    def _match_irr_state(kaggle_des_state):
        s = kaggle_des_state.lower().strip()
        for ss in state_irr["irr_match_state"].unique():
            if ss.lower() == s or ss.lower() in s or s in ss.lower():
                return ss
        return state_irr["irr_match_state"].iloc[0]

    df["irr_match_state"] = df["state"].apply(_match_irr_state)
    df = df.merge(state_irr, on=["irr_match_state", "year"], how="left")
    df = df.drop(columns=["irr_match_state"], errors="ignore")
    df["irrigation_source"] = "des_state_level"
    df["irrigation_resolution"] = "state_level"
    df["irrigation_is_fallback"] = True

    # ── Feature engineering ───────────────────────────────────────────────────
    df["season_kharif"] = (df["season"] == "kharif").astype(int)

    # Fill preceding_monsoon_rainfall for kharif rows (not applicable)
    df["preceding_monsoon_rainfall"] = df["preceding_monsoon_rainfall"].fillna(
        df["rainfall_total"] * 0.0   # 0 for kharif (monsoon IS the season)
    )

    # One-hot encode crops
    df_crops = pd.get_dummies(df["crop"], prefix="crop", dtype=int)
    for c in CROP_COLS:
        if c not in df_crops.columns:
            df_crops[c] = 0
    df = pd.concat([df, df_crops], axis=1)

    # ── Validation checks ─────────────────────────────────────────────────────
    n_before = len(df)
    issues = []

    # No negative rainfall
    bad_rain = (df["rainfall_total"] < 0).sum()
    if bad_rain:
        issues.append(f"Negative rainfall in {bad_rain} rows")

    # pH range
    bad_ph = ((df["soil_ph"] < 3.0) | (df["soil_ph"] > 9.5)).sum()
    if bad_ph:
        issues.append(f"pH out of range [3,9.5] in {bad_ph} rows")

    # Duplicates
    dup_key = ["district_id", "year", "season", "crop"]
    n_dup = df.duplicated(subset=dup_key).sum()
    if n_dup:
        issues.append(f"{n_dup} duplicate district-year-season-crop rows")
        df = df.drop_duplicates(subset=dup_key, keep="first")

    # Drop rows missing critical features
    critical_cols = ["temp_mean", "rainfall_total", "soil_ph"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    if dropped:
        issues.append(f"Dropped {dropped} rows with missing critical features")

    if verbose:
        print(f"\n[training_table] Validation:")
        for issue in issues:
            print(f"  WARNING: {issue}")
        if not issues:
            print("  All checks passed.")

    # ── Impute remaining NAs ───────────────────────────────────────────────────
    for col in ["soil_oc", "soil_clay", "soil_cec", "lagged_irrigated_share",
                "rainfall_variability", "preceding_monsoon_rainfall"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    fallback_soil_count = int(df["soil_is_fallback"].sum()) if "soil_is_fallback" in df.columns else 0
    fallback_irr_count = int(df["irrigation_is_fallback"].sum()) if "irrigation_is_fallback" in df.columns else 0

    # ── Data quality report ────────────────────────────────────────────────────
    quality_report = {
        "total_rows":           int(len(df)),
        "rows_before_cleaning": int(n_before),
        "district_count":       int(df["district_id"].nunique()),
        "crop_count":           int(df["crop"].nunique()),
        "year_range":           [int(df["year"].min()) if not df.empty else 0,
                                 int(df["year"].max()) if not df.empty else 0],
        "seasons":              df["season"].unique().tolist(),
        "is_major_rate":        round(float(df["is_major"].mean()), 4),
        "soil_fallback_count":  fallback_soil_count,
        "soil_real_count":      int(len(df)) - fallback_soil_count,
        "irr_fallback_count":   fallback_irr_count,
        "irr_real_count":       int(len(df)) - fallback_irr_count,
        "climate_direct_match": climate_direct_match,
        "climate_state_fallback": climate_state_fallback,
        "climate_missing":      climate_missing,
        "missingness_pct":      {
            col: round(float(df[col].isna().mean() * 100), 2)
            for col in FEATURE_COLS if col in df.columns
        },
        "issues":               issues,
        "crop_coverage": {
            crop: int((df[df["crop"] == crop]["district_id"].nunique()))
            for crop in sorted(df["crop"].unique())
        },
        "crop_is_major_rate": {
            crop: round(float(df[df["crop"] == crop]["is_major"].mean()), 3)
            for crop in sorted(df["crop"].unique())
        },
    }

    os.makedirs(FEATURES_DIR, exist_ok=True)
    with open(QUALITY_PATH, "w") as f:
        json.dump(quality_report, f, indent=2)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUT_PATH, index=False)

    if verbose:
        print(f"\n[training_table] Saved: {OUT_PATH}")
        print(f"  Final shape: {df.shape}")
        print(f"  Feature columns: {FEATURE_COLS}")
        print(f"  is_major rate: {df['is_major'].mean():.2%}")
        print(f"  Quality report: {QUALITY_PATH}")

    return df


if __name__ == "__main__":
    build_training_table(verbose=True)
