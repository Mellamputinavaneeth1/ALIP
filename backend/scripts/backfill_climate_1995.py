"""
ALIP - Backfill 1995-1999 climate data for cached districts.
Eliminates 10-year rolling window truncation for target years 2005-2014.
"""
import os
import sys
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.data.district_centroids import DISTRICT_CENTROIDS

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'climate_cache.json')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'india_district_coords.csv')

def get_coords_map():
    coords_map = {}
    for did, meta in DISTRICT_CENTROIDS.items():
        coords_map[did] = (meta['lat'], meta['lon'])
    
    if os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        for _, row in df_csv.iterrows():
            state = str(row['state'])
            district = str(row['district'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            did = state[:3].upper() + '_' + district.strip().upper().replace(' ','_').replace('-','_')[:12]
            if did not in coords_map:
                coords_map[did] = (lat, lon)
    return coords_map

def make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

def fetch_1995_1999(session, lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1995-01-01",
        "end_date": "1999-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "Asia/Kolkata",
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    precip = daily.get("precipitation_sum", [])
    return {d: (t, p) for d, t, p in zip(dates, temps, precip) if t is not None and p is not None}

def backfill():
    if not os.path.exists(CACHE_PATH):
        print("No climate cache found at", CACHE_PATH)
        return

    with open(CACHE_PATH, 'r') as f:
        cache = json.load(f)

    coords_map = get_coords_map()
    session = make_session()

    districts_to_fetch = []
    for did, data in cache.items():
        if not any(d.startswith('1995') for d in data.keys()):
            districts_to_fetch.append(did)

    print(f"Total districts in cache: {len(cache)}")
    print(f"Districts needing 1995-1999 backfill: {len(districts_to_fetch)}")

    success_count = 0
    for idx, did in enumerate(districts_to_fetch, start=1):
        if did not in coords_map:
            print(f"[{idx}/{len(districts_to_fetch)}] {did}: Coordinates not found, skipping")
            continue

        lat, lon = coords_map[did]
        try:
            records = fetch_1995_1999(session, lat, lon)
            cache[did].update(records)
            success_count += 1
            print(f"[{idx}/{len(districts_to_fetch)}] {did} ({lat:.2f}, {lon:.2f}): Added {len(records)} records")
        except Exception as e:
            print(f"[{idx}/{len(districts_to_fetch)}] {did}: Error {e}")

        # Polite delay
        time.sleep(0.5)

        # Periodic save every 20 districts
        if idx % 20 == 0:
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)
            print("  [Auto-saved cache checkpoint]")

    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)

    print(f"\nBackfill complete! Successfully updated {success_count}/{len(districts_to_fetch)} districts.")

if __name__ == '__main__':
    backfill()
