"""
ALIP — District Service
Resolves farmer-provided location (name or lat/lon) to a canonical district_id.
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.data.district_centroids import (
    DISTRICT_CENTROIDS, NAME_TO_ID, resolve_district_id, get_district_meta, STATE_DISTRICTS
)


def resolve_district(location: str = None, lat: float = None, lon: float = None) -> dict:
    """
    Resolves a location to a canonical district.
    Priority: lat/lon > name string
    Returns: {district_id, district_name, state, lat, lon, found: bool, method: str}
    """
    # ── Try lat/lon spatial resolution ────────────────────────────────────────
    if lat is not None and lon is not None:
        result = _resolve_by_latlon(lat, lon)
        if result["found"]:
            return result

    # ── Try name resolution ───────────────────────────────────────────────────
    if location:
        did = resolve_district_id(location)
        if did:
            meta = get_district_meta(did)
            return {
                "district_id":   did,
                "district_name": meta["name"],
                "state":         meta["state"],
                "lat":           meta["lat"],
                "lon":           meta["lon"],
                "found":         True,
                "method":        "name_match",
            }

    return {
        "district_id":   None,
        "district_name": location or "Unknown",
        "state":         None,
        "lat":           lat,
        "lon":           lon,
        "found":         False,
        "method":        "unresolved",
    }


_spatial_index = None

def _load_spatial_index():
    global _spatial_index
    if _spatial_index is not None:
        return _spatial_index
    import pickle
    index_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'district_spatial_index.pkl')
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, 'rb') as f:
            _spatial_index = pickle.load(f)
        return _spatial_index
    except Exception:
        return None

def _resolve_by_polygon(lat: float, lon: float) -> dict:
    gdf = _load_spatial_index()
    if gdf is None:
        return {'found': False}
    try:
        from shapely.geometry import Point
        point = Point(lon, lat)
        matches = gdf[gdf.geometry.contains(point)]
        if len(matches) == 0:
            return {'found': False}
        
        row = matches.iloc[0]
        district_name = row['district_norm']
        state_norm = row.get('state_norm', '')
        
        did = resolve_district_id(district_name)
        if not did:
            did = resolve_district_id(district_name + ' ' + state_norm)
            
        return {
            "district_id":   did,
            "district_name": district_name,
            "state":         state_norm,
            "lat":           lat,
            "lon":           lon,
            "found":         True,
            "method":        "polygon"
        }
    except Exception:
        return {'found': False}

def _resolve_by_latlon(lat: float, lon: float) -> dict:
    """Find nearest district centroid by Haversine distance."""
    poly_res = _resolve_by_polygon(lat, lon)
    if poly_res.get('found'):
        return poly_res

    best_did, best_dist = None, float('inf')
    for did, meta in DISTRICT_CENTROIDS.items():
        dist = _haversine(lat, lon, meta["lat"], meta["lon"])
        if dist < best_dist:
            best_dist = dist
            best_did = did

    if best_did and best_dist <= 50.0:   # within 50 km -> confident spatial match
        meta = get_district_meta(best_did)
        return {
            "district_id":   best_did,
            "district_name": meta["name"],
            "state":         meta["state"],
            "lat":           meta["lat"],
            "lon":           meta["lon"],
            "found":         True,
            "method":        "haversine_nearest",
            "distance_km":   round(best_dist, 1),
        }
    return {"found": False}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def list_supported_districts() -> list:
    """Returns list of all supported districts with metadata."""
    return [
        {
            "district_id":   did,
            "district_name": meta["name"],
            "state":         meta["state"],
        }
        for did, meta in DISTRICT_CENTROIDS.items()
    ]


def get_district_info(district_id: str) -> dict | None:
    """Returns full district metadata including allowed crops per season."""
    from ml.data.district_centroids import DISTRICT_CROP_CALENDAR
    meta = get_district_meta(district_id)
    if not meta:
        return None
    calendar = DISTRICT_CROP_CALENDAR.get(district_id, {})
    return {
        "district_id":   district_id,
        "district_name": meta["name"],
        "state":         meta["state"],
        "lat":           meta["lat"],
        "lon":           meta["lon"],
        "kharif_crops":  calendar.get("kharif", []),
        "rabi_crops":    calendar.get("rabi", []),
    }
