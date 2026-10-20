"""
Urgency Scorer Service
======================

Combines multiple supply-side signals into a single 1-10 actionability score
that tells stakeholders how urgently they need to respond to an oversupply
situation.

Signals consumed
----------------
* **oversupply_ratio** – current supply ÷ threshold (e.g. 2.04 means 204 %).
* **z_score** – statistical deviation from historical mean
  (from ``supply_history_service.compute_zscore``).
* **supply_trend** – directional momentum of supply arrivals.
* **crop** – used to look up perishability tier (tier 1 = most perishable).
* **days_since_detected** – persistence of the oversupply event.

Output
------
A dict containing the numeric score, human-readable label, recommended
act-within window, component breakdown, and a natural-language reasoning
string.
"""

from __future__ import annotations

from typing import Dict, Union

# ---------------------------------------------------------------------------
# Perishability tiers  (1 = highly perishable, 3 = storable)
# ---------------------------------------------------------------------------
# Try to reuse the canonical mapping from demand_ml_service; fall back to a
# local copy so this module stays self-contained.
try:
    from services.demand_ml_service import PERISHABILITY_TIERS  # type: ignore[import-untyped]
except ImportError:
    PERISHABILITY_TIERS: Dict[str, int] = {
        # Tier 1 – highly perishable
        "tomato": 1,
        "sugarcane": 1,
        "banana": 1,
        # Tier 2 – moderately perishable
        "maize": 2,
        "groundnut": 2,
        "potato": 2,
        "onion": 2,
        # Tier 3 – storable
        "rice": 3,
        "wheat": 3,
        "cotton": 3,
        "soybean": 3,
    }

DEFAULT_TIER: int = 3

# ---------------------------------------------------------------------------
# Tier multiplier & trend boost look-ups
# ---------------------------------------------------------------------------
_TIER_MULTIPLIER: Dict[int, float] = {
    1: 2.0,   # perishable crops escalate urgency fast
    2: 1.3,
    3: 1.0,   # storable crops – no multiplier
}

_TREND_BOOST: Dict[str, float] = {
    "rapidly_increasing": 1.5,
    "increasing": 0.8,
    "stable": 0.0,
    "decreasing": -0.5,
    "rapidly_decreasing": -1.0,
}

# ---------------------------------------------------------------------------
# Label mapping & recommended action windows
# ---------------------------------------------------------------------------
_ACT_WITHIN_DAYS: Dict[str, int] = {
    "EMERGENCY": 1,
    "CRITICAL": 2,
    "HIGH": 5,
    "WATCH": 14,
}


def _score_to_label(score: int) -> tuple[str, str]:
    """Map a 1-10 score to a human-readable label and display colour.

    Returns
    -------
    tuple[str, str]
        ``(label, colour)`` where *label* is one of
        ``WATCH | HIGH | CRITICAL | EMERGENCY`` and *colour* is a CSS-safe
        colour name.
    """
    if score <= 3:
        return "WATCH", "green"
    if score <= 6:
        return "HIGH", "orange"
    if score <= 8:
        return "CRITICAL", "red"
    return "EMERGENCY", "darkred"


def _perishability_description(tier: int) -> str:
    """Return a short human-friendly perishability descriptor."""
    return {1: "highly perishable", 2: "moderately perishable", 3: "storable"}.get(
        tier, "storable"
    )


def _z_score_description(z: float) -> str:
    """Describe how unusual the z-score is in plain English."""
    abs_z = abs(z)
    if abs_z >= 3.0:
        return "extreme spike"
    if abs_z >= 2.0:
        return "unusual spike"
    if abs_z >= 1.0:
        return "moderate deviation"
    return "within normal range"


def _trend_description(trend: str) -> str:
    """Convert a trend slug to a human-friendly phrase."""
    return trend.replace("_", " ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_urgency(
    oversupply_ratio: float,
    z_score: float,
    supply_trend: str,
    crop: str,
    days_since_detected: int = 0,
) -> Dict[str, Union[int, float, str, Dict[str, float]]]:
    """Compute a composite urgency score for an oversupply event.

    Parameters
    ----------
    oversupply_ratio:
        Ratio of current supply to the oversupply threshold.
        A value of ``2.04`` means supply is 204 % of the threshold.
    z_score:
        Statistical z-score of today's supply relative to historical data,
        as returned by ``supply_history_service.compute_zscore``.
    supply_trend:
        Directional momentum label – one of ``"rapidly_increasing"``,
        ``"increasing"``, ``"stable"``, ``"decreasing"``, or
        ``"rapidly_decreasing"``.
    crop:
        Crop name (case-insensitive).  Used for perishability look-up.
    days_since_detected:
        Number of consecutive days the oversupply condition has persisted.
        Adds a time-pressure boost after 3 days.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``score`` (int) – 1-10 actionability score.
        * ``label`` (str) – ``WATCH | HIGH | CRITICAL | EMERGENCY``.
        * ``color`` (str) – CSS-safe colour for UI rendering.
        * ``act_within_days`` (int) – recommended action window.
        * ``perishability_tier`` (int) – 1 (perishable) to 3 (storable).
        * ``reasoning`` (str) – natural-language explanation.
        * ``components`` (dict) – breakdown of individual scoring factors.

    Examples
    --------
    >>> result = compute_urgency(
    ...     oversupply_ratio=2.04,
    ...     z_score=2.8,
    ...     supply_trend="rapidly_increasing",
    ...     crop="rice",
    ...     days_since_detected=3,
    ... )
    >>> result["label"]
    'CRITICAL'
    """
    # -- Component calculations ---------------------------------------------
    base: float = min(9.0, oversupply_ratio * 3.0)
    z_boost: float = min(2.0, z_score * 0.5)

    tier: int = PERISHABILITY_TIERS.get(crop.lower(), DEFAULT_TIER)
    tier_multiplier: float = _TIER_MULTIPLIER.get(tier, 1.0)

    trend_boost: float = _TREND_BOOST.get(supply_trend, 0.0)

    time_boost: float = 1.0 if days_since_detected >= 3 else 0.0

    # -- Composite score ----------------------------------------------------
    raw_score: float = (base + z_boost + trend_boost + time_boost) * tier_multiplier
    final_score: int = max(1, min(10, round(raw_score)))

    # -- Label & action window ----------------------------------------------
    label, color = _score_to_label(final_score)
    act_within_days: int = _ACT_WITHIN_DAYS[label]

    # -- Reasoning string ---------------------------------------------------
    perish_desc = _perishability_description(tier)
    z_desc = _z_score_description(z_score)
    trend_desc = _trend_description(supply_trend)

    parts: list[str] = [
        f"{crop.capitalize()} ({perish_desc})",
        f"with Z-score {z_score:.1f} ({z_desc})",
    ]
    if supply_trend != "stable":
        parts.append(f"+ {trend_desc} trend")
    if days_since_detected > 0:
        day_word = "day" if days_since_detected == 1 else "days"
        parts.append(f"+ {days_since_detected} {day_word} since detection")

    reasoning = (
        f"{' '.join(parts)} = {label} urgency. "
        f"Act within {act_within_days} {'day' if act_within_days == 1 else 'days'}."
    )

    return {
        "score": final_score,
        "label": label,
        "color": color,
        "act_within_days": act_within_days,
        "perishability_tier": tier,
        "reasoning": reasoning,
        "components": {
            "base": round(base, 4),
            "z_boost": round(z_boost, 4),
            "trend_boost": round(trend_boost, 4),
            "time_boost": round(time_boost, 4),
            "tier_multiplier": round(tier_multiplier, 4),
            "raw_score": round(raw_score, 4),
        },
    }
