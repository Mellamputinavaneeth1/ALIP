"""
ALIP — Dataset Builder v2
Builds Crop_recommendation_v2.csv by augmenting the baseline dataset with:
  - soil_type (derived from N, P, K, pH using pedological rules)
  - One-hot encoded soil type columns
  - Historical climate features (avg_temperature, avg_humidity, seasonal_rainfall)
    from crop's typical climate zone

Run directly:  python -m data.dataset_builder
"""

import pandas as pd
import numpy as np
import os

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, "Crop_recommendation.csv")
DATA_PATH_V2 = os.path.join(BASE_DIR, "Crop_recommendation_v2.csv")

# ─── Soil Type Mapping ───────────────────────────────────────────────────────
# Rules derived from NBSS&LUP (National Bureau of Soil Survey) pedological standards
# for Indian soils. Maps NPK+pH value ranges to one of 6 soil type categories.

SOIL_TYPES = ["alluvial", "black", "red", "laterite", "mountain", "sandy"]

# Crop-specific soil type ground truth (from ICAR / NBSS guidelines)
# Used as fallback when NPK+pH rules are ambiguous
CROP_SOIL_GROUND_TRUTH = {
    "rice":        "alluvial",
    "jute":        "alluvial",
    "maize":       "red",
    "cotton":      "black",
    "chickpea":    "black",
    "lentil":      "black",
    "pigeonpeas":  "red",
    "mungbean":    "red",
    "blackgram":   "red",
    "kidneybeans": "red",
    "mothbeans":   "sandy",
    "apple":       "mountain",
    "coffee":      "mountain",
    "coconut":     "laterite",
    "banana":      "alluvial",
    "papaya":      "alluvial",
    "mango":       "red",
    "orange":      "red",
    "grapes":      "black",
    "watermelon":  "sandy",
    "muskmelon":   "sandy",
    "pomegranate": "red",
}


def assign_soil_type(row) -> str:
    """
    Hybrid soil type assignment:
    1. Apply strict pedological rules for clear-cut cases
    2. Fall back to crop-specific ground truth for ambiguous NPK ranges
    Based on NBSS&LUP Indian Soil Classification standards.
    """
    n, p, k, ph = row['N'], row['P'], row['K'], row['ph']
    crop = row.get('label', '')

    # ── Rule 1: Mountain / Forest soils (apple, coffee) ──────────────────────
    # Very high K (>150), low pH, moderate-low N — distinct signal
    if k >= 150 and ph <= 6.5:
        return "mountain"

    # ── Rule 2: Laterite soils (coconut, rubber, coastal crops) ──────────────
    # Low N (<30), very acidic pH (<5.5), moderate K
    if n < 30 and ph < 5.5:
        return "laterite"

    # ── Rule 3: Black Soil (Vertisol) — cotton, chickpea, grapes ─────────────
    # High pH (>=7.2), moderate-high N, good K
    if ph >= 7.2 and n >= 35:
        return "black"

    # ── Rule 4: Alluvial soil — rice, jute, banana ───────────────────────────
    # High N (>=70), neutral-to-slightly alkaline pH, good K
    if n >= 70 and ph >= 6.5 and k >= 35:
        return "alluvial"

    # ── Rule 5: Sandy / Arid soils — mothbeans, watermelon, muskmelon ────────
    # Very low N (<25) AND low K (<25) — dryland crops
    if n < 25 and k < 25:
        return "sandy"

    # ── Rule 6: Red soil — moderate N, low-to-moderate K, slightly acidic ────
    if 30 <= n < 70 and ph < 7.0 and k < 50:
        return "red"

    # ── Fallback: use crop ground truth ──────────────────────────────────────
    return CROP_SOIL_GROUND_TRUTH.get(crop, "red")


# ─── Climate Feature Addition ────────────────────────────────────────────────
from data.climate_data import CROP_CLIMATE_ZONE, CLIMATE_ZONES


def get_crop_climate_features(crop: str) -> dict:
    """
    Returns average climate features across all 3 seasons for a crop's
    typical climate zone. We average across kharif/rabi/zaid to make
    the training data more representative.
    """
    zone_key = CROP_CLIMATE_ZONE.get(crop, "semi_arid")
    zone = CLIMATE_ZONES[zone_key]

    seasons = ["kharif", "rabi", "zaid"]
    avg_temp     = np.mean([zone[s]["avg_temperature"]   for s in seasons])
    avg_humid    = np.mean([zone[s]["avg_humidity"]      for s in seasons])
    avg_rain     = np.mean([zone[s]["seasonal_rainfall"] for s in seasons])

    return {
        "avg_temperature":   round(avg_temp, 1),
        "avg_humidity":      round(avg_humid, 1),
        "seasonal_rainfall": round(avg_rain, 1),
        "climate_zone":      zone_key,
    }


# ─── Build Dataset ───────────────────────────────────────────────────────────

def build_augmented_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Reads Crop_recommendation.csv, adds:
      - soil_type (string)
      - soil_type_alluvial, soil_type_black, soil_type_red,
        soil_type_laterite, soil_type_mountain, soil_type_sandy (OHE)
      - avg_temperature, avg_humidity, seasonal_rainfall (climate normals)
      - climate_zone (string, for reference)
    Saves result to Crop_recommendation_v2.csv.
    Returns the augmented DataFrame.
    """
    if verbose:
        print("[dataset_builder] Reading base dataset...")

    df = pd.read_csv(DATA_PATH)
    if verbose:
        print(f"  Base shape: {df.shape}")

    # Step 1: Assign soil type
    df["soil_type"] = df.apply(assign_soil_type, axis=1)
    if verbose:
        print("\n[dataset_builder] Soil type distribution:")
        for soil, count in df["soil_type"].value_counts().items():
            print(f"  {soil}: {count}")

    # Step 2: One-hot encode soil type (drop first=False, keep all 6)
    for stype in SOIL_TYPES:
        df[f"soil_type_{stype}"] = (df["soil_type"] == stype).astype(int)

    # Step 3: Add climate normals per crop
    climate_features = df["label"].apply(get_crop_climate_features)
    df["avg_temperature"]   = [f["avg_temperature"]   for f in climate_features]
    df["avg_humidity"]      = [f["avg_humidity"]      for f in climate_features]
    df["seasonal_rainfall"] = [f["seasonal_rainfall"] for f in climate_features]
    df["climate_zone"]      = [f["climate_zone"]      for f in climate_features]

    if verbose:
        print("\n[dataset_builder] Climate zone distribution:")
        for zone, count in df["climate_zone"].value_counts().items():
            print(f"  {zone}: {count}")
        print(f"\n[dataset_builder] Final dataset shape: {df.shape}")
        print(f"[dataset_builder] Columns: {list(df.columns)}")

    # Save
    df.to_csv(DATA_PATH_V2, index=False)
    if verbose:
        print(f"\n[dataset_builder] Saved to {DATA_PATH_V2}")

    return df


def get_feature_columns_v2() -> list:
    """Returns the ordered list of ML feature column names for v2 model."""
    return [
        "N", "P", "K", "ph",
        "soil_type_alluvial", "soil_type_black", "soil_type_red",
        "soil_type_laterite", "soil_type_mountain", "soil_type_sandy",
        "avg_temperature", "avg_humidity", "seasonal_rainfall",
    ]


def encode_soil_type_ohe(soil_type: str) -> dict:
    """
    Returns OHE dict for a given soil type string.
    Used during inference to build the feature vector.
    """
    st = soil_type.lower().strip()
    return {
        "soil_type_alluvial":  int(st == "alluvial"),
        "soil_type_black":     int(st == "black"),
        "soil_type_red":       int(st == "red"),
        "soil_type_laterite":  int(st == "laterite"),
        "soil_type_mountain":  int(st == "mountain"),
        "soil_type_sandy":     int(st == "sandy"),
    }


if __name__ == "__main__":
    build_augmented_dataset(verbose=True)
