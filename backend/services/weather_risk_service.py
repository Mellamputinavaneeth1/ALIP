"""
Weather Risk Service for ALIP Backend.

Assesses weather-related transport risk for agricultural commodity routes
using OpenWeatherMap 5-day/3-hour forecast data. Factors in crop
perishability to adjust scores and provides natural-language timing
recommendations (e.g. "Send TODAY — rain arrives tomorrow").
"""

import os
import math
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OWM_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
OWM_FORECAST_URL: str = "https://api.openweathermap.org/data/2.5/forecast"

# ---------------------------------------------------------------------------
# Perishability tiers  (shared with urgency_service)
#   Tier 1 – highly perishable   (tomato, sugarcane, banana)
#   Tier 2 – moderately durable  (maize, groundnut, potato, onion)
#   Tier 3 – storable            (rice, wheat, cotton, soybean)
# ---------------------------------------------------------------------------

PERISHABILITY_TIERS: Dict[str, int] = {
    "tomato": 1, "sugarcane": 1, "banana": 1,
    "maize": 2, "groundnut": 2, "potato": 2, "onion": 2,
    "rice": 3, "wheat": 3, "cotton": 3, "soybean": 3,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Route midpoint
# ═══════════════════════════════════════════════════════════════════════════

def compute_route_midpoint(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
) -> tuple:
    """Return the geographic midpoint between two coordinates.

    Uses a simple arithmetic average — accurate enough for short-to-medium
    domestic routes where great-circle distortion is negligible.

    Args:
        lat1: Source latitude.
        lng1: Source longitude.
        lat2: Destination latitude.
        lng2: Destination longitude.

    Returns:
        Tuple of (mid_lat, mid_lng).
    """
    mid_lat = (lat1 + lat2) / 2
    mid_lng = (lng1 + lng2) / 2
    return (mid_lat, mid_lng)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Fetch OWM 5-day / 3-hour forecast
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_forecast(lat: float, lng: float) -> list:
    """Fetch the 5-day / 3-hour weather forecast from OpenWeatherMap.

    Each returned entry is a dict with keys:
        dt          – datetime of the forecast slot
        temp        – temperature in °C
        rain_mm     – precipitation in mm for the 3-hour window
        description – short weather description (e.g. "light rain")
        wind_speed  – wind speed in m/s

    Args:
        lat: Latitude of the location.
        lng: Longitude of the location.

    Returns:
        List of forecast entry dicts.  Empty list on any API / network error.
    """
    if not OWM_API_KEY:
        print("[weather_risk] WARNING: WEATHER_API_KEY not set — returning empty forecast")
        return []

    params = {
        "lat": lat,
        "lon": lng,
        "appid": OWM_API_KEY,
        "units": "metric",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(OWM_FORECAST_URL, params=params)
            response.raise_for_status()
            data = response.json()

        entries: List[Dict] = []
        for item in data.get("list", []):
            entry = {
                "dt": datetime.utcfromtimestamp(item["dt"]),
                "temp": item.get("main", {}).get("temp", 0.0),
                "rain_mm": item.get("rain", {}).get("3h", 0.0),
                "description": (
                    item.get("weather", [{}])[0].get("description", "unknown")
                    if item.get("weather")
                    else "unknown"
                ),
                "wind_speed": item.get("wind", {}).get("speed", 0.0),
            }
            entries.append(entry)

        return entries

    except httpx.HTTPStatusError as exc:
        print(
            f"[weather_risk] OWM API HTTP error {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )
        return []
    except httpx.RequestError as exc:
        print(f"[weather_risk] OWM request error: {exc}")
        return []
    except Exception as exc:
        print(f"[weather_risk] Unexpected error fetching forecast: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Assess rain risk from forecast entries
# ═══════════════════════════════════════════════════════════════════════════

def _classify_daily_rain(rain_mm: float) -> Dict:
    """Classify a single day's cumulative rainfall into a risk level + score.

    Thresholds (mm over 24 h):
        < 5   → LOW       (score 90–100)
        5–15  → MEDIUM    (score 60–80)
        15–30 → HIGH      (score 30–55)
        > 30  → CRITICAL  (score 0–25)

    Args:
        rain_mm: Total rainfall for the day in millimetres.

    Returns:
        Dict with 'level' (str) and 'score' (int).
    """
    if rain_mm < 5:
        # Linear interpolation: 0mm → 100, 5mm → 90
        score = int(100 - (rain_mm / 5) * 10)
        return {"level": "LOW", "score": max(score, 90)}
    elif rain_mm < 15:
        # 5mm → 80, 15mm → 60
        score = int(80 - ((rain_mm - 5) / 10) * 20)
        return {"level": "MEDIUM", "score": max(score, 60)}
    elif rain_mm < 30:
        # 15mm → 55, 30mm → 30
        score = int(55 - ((rain_mm - 15) / 15) * 25)
        return {"level": "HIGH", "score": max(score, 30)}
    else:
        # 30mm → 25, 60mm+ → 0
        score = int(25 - ((rain_mm - 30) / 30) * 25)
        return {"level": "CRITICAL", "score": max(score, 0)}


def assess_rain_risk(forecast_entries: list, days: int = 3) -> dict:
    """Aggregate forecast entries into a daily rain-risk breakdown.

    Groups the 3-hour entries by calendar day, sums rainfall per day,
    classifies each day, and computes an overall average score.

    Args:
        forecast_entries: List of forecast dicts from ``fetch_forecast``.
        days: Number of days to consider (default 3).

    Returns:
        Dict containing:
            overall_score    – int 0-100
            overall_level    – LOW | MEDIUM | HIGH | CRITICAL
            daily_breakdown  – list of per-day dicts
    """
    if not forecast_entries:
        return {
            "overall_score": 50,
            "overall_level": "MEDIUM",
            "daily_breakdown": [],
        }

    # --- Group by calendar date ---
    daily_rain: Dict[str, float] = {}
    for entry in forecast_entries:
        day_str = entry["dt"].strftime("%Y-%m-%d")
        daily_rain.setdefault(day_str, 0.0)
        daily_rain[day_str] += entry.get("rain_mm", 0.0)

    # Keep only the first N days
    sorted_days = sorted(daily_rain.keys())[:days]

    daily_breakdown: List[Dict] = []
    total_score = 0

    for day_str in sorted_days:
        rain_mm = round(daily_rain[day_str], 2)
        classification = _classify_daily_rain(rain_mm)
        daily_breakdown.append({
            "day": day_str,
            "rain_mm": rain_mm,
            "level": classification["level"],
            "score": classification["score"],
        })
        total_score += classification["score"]

    overall_score = int(total_score / len(daily_breakdown)) if daily_breakdown else 50

    # Derive overall level from overall score
    if overall_score >= 80:
        overall_level = "LOW"
    elif overall_score >= 55:
        overall_level = "MEDIUM"
    elif overall_score >= 25:
        overall_level = "HIGH"
    else:
        overall_level = "CRITICAL"

    return {
        "overall_score": overall_score,
        "overall_level": overall_level,
        "daily_breakdown": daily_breakdown,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Full weather-risk assessment for a route + crop
# ═══════════════════════════════════════════════════════════════════════════

def _determine_optimal_timing(
    daily_breakdown: List[Dict],
    crop_tier: int,
) -> Dict:
    """Determine the best send-day and produce a human-readable recommendation.

    Decision matrix:
        • Today CLEAR, tomorrow RAIN  → send today
        • Today RAIN                   → find first clear day
        • Tier-1 crop & rain > 2 days  → cannot wait
        • Otherwise                    → send within normal urgency window

    Args:
        daily_breakdown: List of per-day rain-risk dicts (must be date-sorted).
        crop_tier: Perishability tier of the crop (1, 2, or 3).

    Returns:
        Dict with 'optimal_timing' (str) and 'send_day_recommendation' (int).
    """
    if not daily_breakdown:
        return {
            "optimal_timing": "Send within urgency window",
            "send_day_recommendation": 0,
        }

    today = daily_breakdown[0] if len(daily_breakdown) > 0 else None
    tomorrow = daily_breakdown[1] if len(daily_breakdown) > 1 else None

    today_clear = today and today["level"] in ("LOW", "MEDIUM")
    today_rain = today and today["level"] in ("HIGH", "CRITICAL")
    tomorrow_rain = tomorrow and tomorrow["level"] in ("HIGH", "CRITICAL")

    # --- Today clear, rain coming tomorrow → send NOW ---
    if today_clear and tomorrow_rain:
        return {
            "optimal_timing": "Send TODAY — rain arrives tomorrow",
            "send_day_recommendation": 0,
        }

    # --- Today is rainy → find first clear day ---
    if today_rain:
        first_clear_day: Optional[int] = None
        for idx, day_info in enumerate(daily_breakdown):
            if day_info["level"] in ("LOW", "MEDIUM"):
                first_clear_day = idx
                break

        # Tier-1 crop cannot wait more than 2 days
        if crop_tier == 1:
            if first_clear_day is None or first_clear_day > 2:
                return {
                    "optimal_timing": (
                        "Cannot wait — send in rain or find closer market"
                    ),
                    "send_day_recommendation": 0,
                }
            return {
                "optimal_timing": (
                    f"Wait {first_clear_day} day(s) — weather clears"
                ),
                "send_day_recommendation": first_clear_day,
            }

        if first_clear_day is not None:
            return {
                "optimal_timing": (
                    f"Wait {first_clear_day} day(s) — weather clears"
                ),
                "send_day_recommendation": first_clear_day,
            }

        # No clear day in forecast window
        return {
            "optimal_timing": "Persistent rain expected — plan for protection",
            "send_day_recommendation": 0,
        }

    # --- Default: conditions are acceptable ---
    return {
        "optimal_timing": "Send within urgency window",
        "send_day_recommendation": 0,
    }


async def assess_weather_risk(
    source_lat: float,
    source_lng: float,
    dest_lat: float,
    dest_lng: float,
    crop: str = "rice",
) -> dict:
    """Perform a full weather-risk assessment for a crop transport route.

    Steps:
        1. Compute the geographic midpoint of the route.
        2. Fetch weather forecasts for the midpoint AND the destination.
        3. Assess rain risk for both locations.
        4. Apply perishability adjustments to the score.
        5. Determine optimal send-day timing.

    Perishability adjustments:
        • Tier 1 (highly perishable): score × 0.5  if HIGH/CRITICAL rain
        • Tier 2 (moderate):          score × 0.7  if HIGH rain
        • Tier 3 (storable):          no adjustment

    Args:
        source_lat: Source location latitude.
        source_lng: Source location longitude.
        dest_lat:   Destination location latitude.
        dest_lng:   Destination location longitude.
        crop:       Crop name (lowercased key into PERISHABILITY_TIERS).

    Returns:
        Dict with weather_score, weather_level, route_risk,
        destination_risk, optimal_timing, send_day_recommendation,
        crop_tier, and perishability_adjustment flag.
    """
    crop_key = crop.lower().strip()
    crop_tier: int = PERISHABILITY_TIERS.get(crop_key, 3)

    # 1. Route midpoint
    mid_lat, mid_lng = compute_route_midpoint(
        source_lat, source_lng, dest_lat, dest_lng,
    )

    # 2. Fetch forecasts (midpoint + destination) concurrently
    try:
        midpoint_forecast = await fetch_forecast(mid_lat, mid_lng)
        dest_forecast = await fetch_forecast(dest_lat, dest_lng)
    except Exception as exc:
        print(f"[weather_risk] Error fetching forecasts: {exc}")
        midpoint_forecast = []
        dest_forecast = []

    # 3. Assess rain risk for each location
    route_risk = assess_rain_risk(midpoint_forecast, days=3)
    destination_risk = assess_rain_risk(dest_forecast, days=3)

    # 4. Combined score = average of midpoint & destination
    combined_score = int(
        (route_risk["overall_score"] + destination_risk["overall_score"]) / 2
    )
    perishability_adjustment = False

    # 5. Apply perishability adjustment
    worst_level = (
        route_risk["overall_level"]
        if route_risk["overall_score"] <= destination_risk["overall_score"]
        else destination_risk["overall_level"]
    )

    if crop_tier == 1 and worst_level in ("HIGH", "CRITICAL"):
        combined_score = int(combined_score * 0.5)
        perishability_adjustment = True
    elif crop_tier == 2 and worst_level == "HIGH":
        combined_score = int(combined_score * 0.7)
        perishability_adjustment = True
    # Tier 3 — no adjustment

    # Clamp to [0, 100]
    combined_score = max(0, min(100, combined_score))

    # 6. Derive final weather level
    if combined_score >= 80:
        weather_level = "LOW"
    elif combined_score >= 55:
        weather_level = "MEDIUM"
    elif combined_score >= 25:
        weather_level = "HIGH"
    else:
        weather_level = "CRITICAL"

    # 7. Determine optimal timing (use destination breakdown as primary)
    primary_breakdown = (
        destination_risk["daily_breakdown"]
        if destination_risk["daily_breakdown"]
        else route_risk["daily_breakdown"]
    )
    timing = _determine_optimal_timing(primary_breakdown, crop_tier)

    return {
        "weather_score": combined_score,
        "weather_level": weather_level,
        "route_risk": route_risk,
        "destination_risk": destination_risk,
        "optimal_timing": timing["optimal_timing"],
        "send_day_recommendation": timing["send_day_recommendation"],
        "crop_tier": crop_tier,
        "perishability_adjustment": perishability_adjustment,
    }
