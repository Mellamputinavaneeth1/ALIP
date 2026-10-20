"""
ALIP — Climate Feature Builder
Fetches historical seasonal climate from Open-Meteo Historical API.
No API key or registration required.

For each district centroid, fetches daily data (2005-2022),
aggregates to Kharif (Jun-Oct) and Rabi (Nov-Mar) seasonal averages.

Temporal leakage safety:
  For target year Y, uses climate from years [Y-10, Y-1] as predictors.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.data.district_centroids import DISTRICT_CENTROIDS

BASE_DIR = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "raw", "climate_cache.json")
OUT_PATH   = os.path.join(BASE_DIR, "processed", "climate_features.parquet")

TARGET_YEARS  = list(range(2005, 2023))
FETCH_START   = "1995-01-01"   # Full 10-year trailing window for 2005 onwards
FETCH_END     = "2022-12-31"


def _make_session():
    """Creates a requests session with automatic retry on 429/503."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=3,       # waits 3, 6, 12, 24, 48 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

_session = _make_session()


def _fetch_open_meteo(lat: float, lon: float) -> dict | None:
    """
    Fetches daily temperature + precipitation from Open-Meteo Historical API.
    Returns dict: {date: (temp_mean, precip_sum)}
    Handles 429 rate-limit with automatic exponential backoff.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": FETCH_START, "end_date": FETCH_END,
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "Asia/Kolkata",
    }
    try:
        r = _session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        dates  = daily.get("time", [])
        temps  = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])
        return {d: (t, p) for d, t, p in zip(dates, temps, precip)
                if t is not None and p is not None}
    except Exception as e:
        print(f"    [climate] API error: {e}")
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


def _aggregate_season(daily_data: dict, year: int, season: str) -> dict | None:
    """
    Extracts seasonal statistics from daily data.
    Kharif: Jun 1 – Oct 31 of given year
    Rabi:   Nov 1 of given year – Mar 31 of year+1
    """
    if season == "kharif":
        dates = [f"{year}-{m:02d}-{d:02d}"
                 for m in range(6, 11) for d in range(1, 32)
                 if f"{year}-{m:02d}-{d:02d}" in daily_data]
    else:  # rabi
        dates = (
            [f"{year}-{m:02d}-{d:02d}"
             for m in range(11, 13) for d in range(1, 32)
             if f"{year}-{m:02d}-{d:02d}" in daily_data] +
            [f"{year+1}-{m:02d}-{d:02d}"
             for m in range(1, 4) for d in range(1, 32)
             if f"{year+1}-{m:02d}-{d:02d}" in daily_data]
        )

    if len(dates) < 60:
        return None

    temps  = [daily_data[d][0] for d in dates if daily_data[d][0] is not None]
    precips = [daily_data[d][1] for d in dates if daily_data[d][1] is not None]

    if not temps or not precips:
        return None

    temp_mean    = float(np.mean(temps))
    rain_total   = float(np.sum(precips))
    # Monthly rainfall totals for CoV
    monthly = {}
    for d in dates:
        month = d[:7]
        monthly.setdefault(month, []).append(daily_data[d][1] or 0)
    monthly_totals = [sum(v) for v in monthly.values()]
    rain_cv = float(np.std(monthly_totals) / np.mean(monthly_totals)) if np.mean(monthly_totals) > 0 else 0.0

    return {
        "temp_mean":            round(temp_mean, 2),
        "rainfall_total":       round(rain_total, 1),
        "rainfall_variability": round(rain_cv, 3),
    }


def _get_preceding_monsoon(daily_data: dict, rabi_year: int) -> float:
    """
    For Rabi year Y, returns kharif rainfall of year Y-1.
    This captures soil moisture carryover critical for Rabi.
    """
    kharif = _aggregate_season(daily_data, rabi_year - 1, "kharif")
    return kharif["rainfall_total"] if kharif else None


def build_climate_features(test_districts: list = None, verbose: bool = True, use_cache_only: bool = True) -> pd.DataFrame:
    """
    Main entry point. Fetches or loads cached climate data for districts.
    Generates leakage-safe 10-year historical rolling averages as predictors.
    """
    cache = _load_cache()
    
    # Load additional districts from CSV
    csv_path = os.path.join(BASE_DIR, "raw", "india_district_coords.csv")
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        for _, row in df_csv.iterrows():
            state = str(row['state'])
            district = str(row['district'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            did = state[:3].upper() + '_' + district.strip().upper().replace(' ','_').replace('-','_')[:12]
            
            # If a CSV district is the same city as an existing centroid, the existing centroid wins
            existing_match = False
            for ex_id, ex_meta in DISTRICT_CENTROIDS.items():
                if ex_meta['name'].strip().upper().replace(' ', '_').replace('-', '_') == district.strip().upper().replace(' ', '_').replace('-', '_'):
                    existing_match = True
                    break
            
            if not existing_match and did not in DISTRICT_CENTROIDS:
                DISTRICT_CENTROIDS[did] = {"name": district, "state": state, "lat": lat, "lon": lon}
                
    if use_cache_only and not test_districts:
        districts = [did for did in DISTRICT_CENTROIDS.keys() if did in cache]
    else:
        districts = test_districts or list(DISTRICT_CENTROIDS.keys())
    rows = []

    for i, did in enumerate(districts):
        if did not in DISTRICT_CENTROIDS:
            continue
        meta = DISTRICT_CENTROIDS[did]
        lat, lon = meta["lat"], meta["lon"]
        name = meta["name"]

        if verbose:
            print(f"[{i+1}/{len(districts)}] District ({lat:.2f},{lon:.2f})", end=" ... ")

        # Fetch or load from cache
        if did not in cache:
            daily = _fetch_open_meteo(lat, lon)
            time.sleep(1.5)   # Respect Open-Meteo rate limits
            if daily:
                cache[did] = daily
                _save_cache(cache)
            else:
                print("WARNING: API failed (skipped)")
                continue
        else:
            daily = cache[did]

        if verbose:
            print(f"{len(daily)} daily records")

        # For each target year, build leakage-safe climate features
        # Using 10-year historical window ending at year-1
        for target_year in TARGET_YEARS:
            hist_years = list(range(target_year - 10, target_year))

            for season in ["kharif", "rabi"]:
                # Collect seasonal stats from 10 historical years
                hist_stats = []
                for hy in hist_years:
                    stat = _aggregate_season(daily, hy, season)
                    if stat:
                        hist_stats.append(stat)

                if not hist_stats:
                    continue

                # 10-year averages as features (leakage-safe — uses only past years)
                temp_mean    = float(np.mean([s["temp_mean"]            for s in hist_stats]))
                rain_total   = float(np.mean([s["rainfall_total"]       for s in hist_stats]))
                rain_var     = float(np.mean([s["rainfall_variability"] for s in hist_stats]))

                row = {
                    "district_id":              did,
                    "target_year":              target_year,
                    "season":                   season,
                    "temp_mean":                round(temp_mean, 2),
                    "rainfall_total":           round(rain_total, 1),
                    "rainfall_variability":     round(rain_var, 3),
                    "preceding_monsoon_rainfall": None,
                    "climate_window_start":     target_year - 10,
                    "climate_window_end":       target_year - 1,
                    "window_years":             len(hist_stats),
                    "climate_source":           "open_meteo",
                }

                # Preceding monsoon only relevant for Rabi
                if season == "rabi":
                    prev_monsoon = _get_preceding_monsoon(daily, target_year)
                    row["preceding_monsoon_rainfall"] = (
                        round(prev_monsoon, 1) if prev_monsoon else round(rain_total * 2.5, 1)
                    )
                else:
                    row["preceding_monsoon_rainfall"] = None

                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    if verbose:
        print(f"\n[climate] Saved: {OUT_PATH}")
        print(f"  Shape: {df.shape}")
        print(f"  Districts with data: {df['district_id'].nunique()}")
        print(f"  Year range: {df['target_year'].min()} – {df['target_year'].max()}")
        print(f"  Sample stats: temp={df['temp_mean'].mean():.1f}°C, rain={df['rainfall_total'].mean():.0f}mm")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on 3 districts only")
    parser.add_argument("--districts", nargs="+", help="Specific district IDs")
    args = parser.parse_args()

    if args.test:
        build_climate_features(test_districts=["MH001", "PB001", "TS001"], verbose=True)
    elif args.districts:
        build_climate_features(test_districts=args.districts, verbose=True)
    else:
        build_climate_features(verbose=True)
