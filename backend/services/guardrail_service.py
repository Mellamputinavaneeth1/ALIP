"""
ALIP — Guardrail + Ranking Service
Applies agronomic rules, confidence checks, and abstrains when uncertain.
Ranks calibrated probabilities into final top-3 recommendations.
"""

GUARDRAIL_VERSION = "1.0.0"

# ── Agronomic guardrails (from ICAR / State Agronomy guidelines) ─────────────
# Note: rainfall values represent cumulative seasonal totals (mm), not monthly.
AGRONOMIC_GUARDRAILS = {
    "rice": {
        "rule_id": "GR-RICE-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 450, "max_rainfall": 3000,
        "min_temp": 18, "max_temp": 42,
        "min_ph": 4.5, "max_ph": 8.5
    },
    "wheat": {
        "rule_id": "GR-WHEAT-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["rabi"],
        "min_rainfall": 20, "max_rainfall": 500,
        "min_temp": 8, "max_temp": 32,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "maize": {
        "rule_id": "GR-MAIZE-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif", "rabi"],
        "min_rainfall": 250, "max_rainfall": 1600,
        "min_temp": 14, "max_temp": 40,
        "min_ph": 5.2, "max_ph": 8.2
    },
    "soybean": {
        "rule_id": "GR-SOY-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 350, "max_rainfall": 1800,
        "min_temp": 18, "max_temp": 38,
        "min_ph": 5.5, "max_ph": 8.0
    },
    "cotton": {
        "rule_id": "GR-COTN-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 350, "max_rainfall": 1600,
        "min_temp": 18, "max_temp": 42,
        "min_ph": 5.5, "max_ph": 8.8
    },
    "groundnut": {
        "rule_id": "GR-GNUT-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 300, "max_rainfall": 1400,
        "min_temp": 18, "max_temp": 40,
        "min_ph": 5.3, "max_ph": 8.2
    },
    "pigeonpeas": {
        "rule_id": "GR-TUR-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 350, "max_rainfall": 1500,
        "min_temp": 18, "max_temp": 40,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "mungbean": {
        "rule_id": "GR-MUNG-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif", "rabi"],
        "min_rainfall": 150, "max_rainfall": 1000,
        "min_temp": 18, "max_temp": 40,
        "min_ph": 5.5, "max_ph": 8.2
    },
    "chickpea": {
        "rule_id": "GR-GRAM-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["rabi"],
        "min_rainfall": 20, "max_rainfall": 450,
        "min_temp": 8, "max_temp": 32,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "mustard": {
        "rule_id": "GR-MUST-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["rabi"],
        "min_rainfall": 20, "max_rainfall": 400,
        "min_temp": 8, "max_temp": 30,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "lentil": {
        "rule_id": "GR-LENT-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["rabi"],
        "min_rainfall": 20, "max_rainfall": 400,
        "min_temp": 8, "max_temp": 30,
        "min_ph": 5.5, "max_ph": 8.2
    },
    "barley": {
        "rule_id": "GR-BARL-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["rabi"],
        "min_rainfall": 20, "max_rainfall": 450,
        "min_temp": 8, "max_temp": 32,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "jowar": {
        "rule_id": "GR-JOW-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif", "rabi"],
        "min_rainfall": 200, "max_rainfall": 1200,
        "min_temp": 15, "max_temp": 40,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "bajra": {
        "rule_id": "GR-BAJ-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif"],
        "min_rainfall": 150, "max_rainfall": 900,
        "min_temp": 18, "max_temp": 42,
        "min_ph": 5.5, "max_ph": 8.5
    },
    "ragi": {
        "rule_id": "GR-RAGI-01",
        "version": "1.0.0",
        "source": "ICAR general guidelines (approximate)",
        "seasons": ["kharif", "rabi"],
        "min_rainfall": 300, "max_rainfall": 1500,
        "min_temp": 15, "max_temp": 38,
        "min_ph": 5.0, "max_ph": 8.2
    },
}

# Confidence thresholds based on validation calibration curves
HIGH_CONF_THRESHOLD   = 0.60
MEDIUM_CONF_THRESHOLD = 0.35
ABSTAIN_THRESHOLD     = 0.20


def calculate_confidence_band(score: float) -> str:
    if score >= HIGH_CONF_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONF_THRESHOLD:
        return "medium"
    return "low"


def apply_guardrails(ranked_crops: list, features: dict, season: str) -> tuple[list, list]:
    """
    Filters and flags crops based on agronomic rules.
    Returns: (filtered_crops, flags)
    """
    flags = []
    filtered = []

    for item in ranked_crops:
        crop = item["crop"]
        score = item["score"]
        guards = AGRONOMIC_GUARDRAILS.get(crop, {})
        warnings = []

        # Season check - hard exclude
        if guards.get("seasons") and season not in guards["seasons"]:
            flags.append({
                "rule_id": guards.get("rule_id", "unknown"),
                "crop": crop,
                "action": "exclude",
                "severity": "hard",
                "reason": "%s is not typically grown in %s season" % (crop, season),
                "version": guards.get("version", GUARDRAIL_VERSION),
            })
            continue

        # Temperature check - soft flag
        temp = features.get("temp_mean", 25.0)
        if guards.get("min_temp") and temp < guards["min_temp"]:
            w = "Temperature (%.0f C) below minimum for %s" % (temp, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})
        if guards.get("max_temp") and temp > guards["max_temp"]:
            w = "Temperature (%.0f C) above maximum for %s" % (temp, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})

        # Rainfall check - soft flag
        rain = features.get("rainfall_total", 100.0)
        if guards.get("min_rainfall") and rain < guards["min_rainfall"]:
            w = "Rainfall (%.0f mm) insufficient for %s" % (rain, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})
        if guards.get("max_rainfall") and rain > guards["max_rainfall"]:
            w = "Rainfall (%.0f mm) too high for %s" % (rain, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})

        # pH check - soft flag
        ph = features.get("soil_ph", 6.8)
        if guards.get("max_ph") and ph > guards["max_ph"]:
            w = "Soil pH (%.1f) above range for %s" % (ph, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})
        if guards.get("min_ph") and ph < guards["min_ph"]:
            w = "Soil pH (%.1f) below range for %s" % (ph, crop)
            warnings.append(w)
            flags.append({"rule_id": guards.get("rule_id"), "crop": crop, "action": "flag", "severity": "soft", "reason": w, "version": guards.get("version", GUARDRAIL_VERSION)})

        item["guardrail_warnings"] = warnings
        item["guardrail_status"] = "pass" if not warnings else "flagged"
        filtered.append(item)

    return filtered, flags


def should_abstain(top_score: float, coverage: dict) -> tuple[bool, str]:
    """
    Decides whether to abstain from a high-confidence recommendation.
    Returns: (abstain: bool, reason: str)
    """
    if not coverage.get("has_climate"):
        return True, "Climate data not available for this district"
    if not coverage.get("has_soil"):
        return True, "Soil data not available for this district"
    if top_score < ABSTAIN_THRESHOLD:
        return True, "No crop shows strong historical association with this region and season"
    return False, ""


def rank_and_top3(ranked_crops: list, features: dict, season: str, top_n: int = 3) -> tuple[list, list, bool, str]:
    """
    Full ranking pipeline: guardrails -> sort -> top-N -> confidence -> abstain check.
    Returns: (top_crops, flags, abstain, abstain_reason)
    """
    coverage = features.get("coverage", {
        "has_climate": True, "has_soil": True
    })

    # Apply guardrails
    filtered, flags = apply_guardrails(ranked_crops, features, season)

    if not filtered:
        return [], flags, True, "No suitable crops found for this season and environment"

    top_score = filtered[0]["score"]
    abstain, reason = should_abstain(top_score, coverage)

    if abstain:
        # Still return candidates but flag uncertainty
        top = filtered[:top_n]
        for item in top:
            item["confidence_band"] = "low"
            if "guardrail_status" not in item:
                item["guardrail_status"] = "pass" if not item.get("guardrail_warnings") else "flagged"
        return top, flags, True, reason

    top = filtered[:top_n]
    for item in top:
        item["confidence_band"] = calculate_confidence_band(item["score"])
        if "guardrail_status" not in item:
            item["guardrail_status"] = "pass" if not item.get("guardrail_warnings") else "flagged"

    return top, flags, False, ""

