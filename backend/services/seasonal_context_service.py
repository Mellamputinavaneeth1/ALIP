"""
ALIP — Seasonal Context Engine
Provides seasonal intelligence for agricultural commodities
based on ICAR/APMC published seasonal patterns.
"""

import datetime
from typing import Dict, Optional


# ─── Season Determination ──────────────────────────────
SEASON_MONTHS = {
    1: "rabi", 2: "rabi", 3: "rabi", 4: "zaid",
    5: "zaid", 6: "kharif", 7: "kharif", 8: "kharif",
    9: "kharif", 10: "kharif", 11: "rabi", 12: "rabi"
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# ─── Seasonal Patterns Database ────────────────────────
# Based on ICAR/APMC published seasonal data for Indian crops
SEASONAL_PATTERNS = {
    "rice": {
        "kharif": {
            "peak_supply_months": [9, 10, 11],
            "typical_oversupply": "Oct-Nov peak arrivals from kharif harvest",
            "self_correction": True,
            "self_correction_note": "Eases by December as harvest season ends",
            "historical_note": "Oct prices typically drop 15-25% from September baseline"
        },
        "rabi": {
            "peak_supply_months": [3, 4],
            "typical_oversupply": "March-April rabi harvest arrivals",
            "self_correction": True,
            "self_correction_note": "Eases by May as planting season begins",
            "historical_note": "March prices drop 10-15% from February baseline"
        },
        "zaid": {
            "peak_supply_months": [5, 6],
            "typical_oversupply": "Late clearing of kharif stockpiles",
            "self_correction": True,
            "self_correction_note": "Supply reduces as kharif planting begins June-July",
            "historical_note": "May-June is transition period, prices stabilize"
        }
    },
    "wheat": {
        "rabi": {
            "peak_supply_months": [3, 4, 5],
            "typical_oversupply": "March-May rabi harvest peak",
            "self_correction": True,
            "self_correction_note": "Eases by June as storage absorbs excess",
            "historical_note": "April mandi prices drop 10-20% from March peak"
        }
    },
    "maize": {
        "kharif": {
            "peak_supply_months": [9, 10],
            "typical_oversupply": "September-October harvest flood",
            "self_correction": True,
            "self_correction_note": "Stabilizes by November",
            "historical_note": "Prices drop 15-30% during peak harvest"
        }
    },
    "cotton": {
        "kharif": {
            "peak_supply_months": [10, 11, 12],
            "typical_oversupply": "Oct-Dec picking season peak",
            "self_correction": True,
            "self_correction_note": "Eases by January as picking ends",
            "historical_note": "November prices typically 10-15% below MSP"
        }
    },
    "sugarcane": {
        "kharif": {
            "peak_supply_months": [11, 12, 1, 2],
            "typical_oversupply": "Nov-Feb crushing season",
            "self_correction": False,
            "self_correction_note": "Requires mill coordination for absorption",
            "historical_note": "Persistent oversupply during crushing season"
        }
    },
    "groundnut": {
        "kharif": {
            "peak_supply_months": [10, 11],
            "typical_oversupply": "Oct-Nov harvest arrivals",
            "self_correction": True,
            "self_correction_note": "Stabilizes by December",
            "historical_note": "Prices drop 20-30% during peak arrivals"
        }
    },
    "soybean": {
        "kharif": {
            "peak_supply_months": [10, 11],
            "typical_oversupply": "Oct-Nov harvest peak",
            "self_correction": True,
            "self_correction_note": "Eases by December as processing absorbs supply",
            "historical_note": "Prices drop 15-25% from September baseline"
        }
    },
    "tomato": {
        "kharif": {
            "peak_supply_months": [1, 2, 3],
            "typical_oversupply": "Jan-Mar winter harvest glut",
            "self_correction": True,
            "self_correction_note": "Corrects quickly as crop is highly perishable",
            "historical_note": "Prices can crash 50-70% during peak glut"
        }
    },
    "onion": {
        "rabi": {
            "peak_supply_months": [4, 5],
            "typical_oversupply": "April-May rabi harvest",
            "self_correction": True,
            "self_correction_note": "Eases as storage absorbs; prices rise post-monsoon",
            "historical_note": "April prices can drop 40-60% before storage absorption"
        }
    }
}


def get_seasonal_context(
    crop: str,
    supply_trend: str = "stable",
    region: str = "",
    month: int = None
) -> dict:
    """
    Provide seasonal intelligence for a crop.

    Args:
        crop: Crop name (e.g., "rice")
        supply_trend: Current trend ("rapidly_increasing", "increasing",
                      "stable", "decreasing", "rapidly_decreasing")
        region: Region name for context (optional)
        month: Month number (1-12), defaults to current month

    Returns:
        dict with seasonal summary, self_correction analysis,
        and actionable reasoning.
    """
    if month is None:
        month = datetime.datetime.now().month

    crop_key = crop.strip().lower()
    season = SEASON_MONTHS.get(month, "kharif")
    month_name = MONTH_NAMES.get(month, "Unknown")

    # Get patterns for this crop + season
    crop_patterns = SEASONAL_PATTERNS.get(crop_key, {})
    season_data = crop_patterns.get(season)

    if not season_data:
        # No pattern data — check other seasons
        for s, data in crop_patterns.items():
            if month in data.get("peak_supply_months", []):
                season_data = data
                season = s
                break

    if not season_data:
        return {
            "crop": crop_key,
            "season": season,
            "month": month,
            "month_name": month_name,
            "has_pattern": False,
            "summary": (
                f"No specific seasonal pattern data available for "
                f"{crop_key} in {month_name} ({season} season)."
            ),
            "self_correction_likely": False,
            "reasoning": "Insufficient historical data for this crop-season combination.",
            "recommendation": "Monitor supply levels and act based on urgency score."
        }

    is_peak = month in season_data.get("peak_supply_months", [])
    self_corrects = season_data.get("self_correction", False)

    # Build reasoning based on trend + seasonality
    trend_growing = supply_trend in ("rapidly_increasing", "increasing")

    if is_peak and self_corrects and not trend_growing:
        reasoning = (
            f"This is a seasonal peak for {crop_key} ({season_data['typical_oversupply']}). "
            f"{season_data['self_correction_note']}. "
            f"Current trend is {supply_trend}, suggesting the peak may be passing. "
            f"Self-correction is likely."
        )
        self_correction_likely = True
        recommendation = (
            "Monitor for 3-5 days. If supply stabilizes or decreases, "
            "self-correction is occurring. Redistribute only if urgency is CRITICAL+."
        )
    elif is_peak and self_corrects and trend_growing:
        reasoning = (
            f"This is a seasonal peak for {crop_key} ({season_data['typical_oversupply']}). "
            f"However, supply is still {supply_trend}. "
            f"The peak has NOT been reached yet. "
            f"Do not wait for self-correction — act within urgency window."
        )
        self_correction_likely = False
        recommendation = "Act within urgency window. Supply still growing despite seasonal peak."
    elif is_peak and not self_corrects:
        reasoning = (
            f"This is a seasonal peak for {crop_key} ({season_data['typical_oversupply']}). "
            f"{season_data['self_correction_note']}. "
            f"Self-correction is NOT expected. Active redistribution required."
        )
        self_correction_likely = False
        recommendation = "Immediate redistribution recommended. This pattern does not self-correct."
    else:
        # Not currently in peak month
        reasoning = (
            f"{month_name} is {season} season for {crop_key}. "
            f"Peak supply typically occurs in months "
            f"{', '.join(MONTH_NAMES[m] for m in season_data['peak_supply_months'])}. "
            f"Current oversupply may be unusual for this time. "
            f"Historical: {season_data['historical_note']}."
        )
        self_correction_likely = False
        recommendation = "This oversupply is unusual for the season. Investigate and act proactively."

    summary = (
        f"{month_name} is {season} season for {crop_key}"
        f"{f' in {region.title()}' if region else ''}. "
        f"{season_data['typical_oversupply']}. "
        f"{season_data['historical_note']}."
    )

    return {
        "crop": crop_key,
        "season": season,
        "month": month,
        "month_name": month_name,
        "has_pattern": True,
        "is_peak_month": is_peak,
        "peak_months": season_data.get("peak_supply_months", []),
        "summary": summary,
        "self_correction_likely": self_correction_likely,
        "self_correction_note": season_data.get("self_correction_note", ""),
        "historical_note": season_data.get("historical_note", ""),
        "reasoning": reasoning,
        "recommendation": recommendation,
        "supply_trend": supply_trend
    }
