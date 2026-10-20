"""
ALIP — Price Trajectory Service
Predicts 7-day price decay for crops under oversupply conditions.
Uses analytical polynomial decay calibrated by perishability tier.
"""

from typing import Dict, List, Optional


# ─── Perishability Tiers ───────────────────────────────
PERISHABILITY_TIERS = {
    "tomato": 1, "sugarcane": 1, "banana": 1,
    "maize": 2, "groundnut": 2, "potato": 2, "onion": 2,
    "rice": 3, "wheat": 3, "cotton": 3, "soybean": 3,
}
DEFAULT_TIER = 3

# ─── Base Prices (₹/kg) ───────────────────────────────
BASE_PRICES = {
    "rice": 22, "wheat": 21, "maize": 18, "cotton": 65,
    "sugarcane": 3, "groundnut": 55, "soybean": 43,
    "tomato": 30, "banana": 25, "potato": 15, "onion": 20,
    "default": 25
}

# ─── Price Elasticity (how fast prices drop) ───────────
ELASTICITY = {
    "rice": 0.35, "wheat": 0.30, "maize": 0.40, "cotton": 0.25,
    "sugarcane": 0.20, "groundnut": 0.45, "soybean": 0.38,
    "tomato": 0.60, "banana": 0.55, "potato": 0.45, "onion": 0.50,
    "default": 0.35
}

# Acceleration factor by perishability tier
TIER_ACCELERATION = {1: 1.15, 2: 1.05, 3: 1.00}

# Price floor: never predict below this fraction of base
PRICE_FLOOR_PCT = 0.30


def compute_price_trajectory(
    crop: str,
    base_price: float = 0.0,
    oversupply_ratio: float = 1.0,
    transport_cost_per_100kg: float = 0.0,
    days: int = 7
) -> dict:
    """
    Predict how crop prices decay over a multi-day window.

    Uses polynomial decay calibrated by crop elasticity,
    oversupply ratio, and perishability tier.

    Args:
        crop: Crop name (e.g. "rice")
        base_price: Current market price per kg.
                    If 0, uses default from BASE_PRICES.
        oversupply_ratio: Current supply / threshold (e.g. 2.04)
        transport_cost_per_100kg: Transport cost for break-even calc
        days: Number of days to project (default 7)

    Returns:
        dict with day_prices, losses, break_even_day,
        and natural language recommendation.
    """
    crop_key = crop.strip().lower()

    # Resolve defaults
    if base_price <= 0:
        base_price = BASE_PRICES.get(crop_key, BASE_PRICES["default"])

    tier = PERISHABILITY_TIERS.get(crop_key, DEFAULT_TIER)
    elasticity = ELASTICITY.get(crop_key, ELASTICITY["default"])
    acceleration = TIER_ACCELERATION.get(tier, 1.0)

    # Decay rate scales with oversupply
    decay_rate = elasticity * max(0, oversupply_ratio - 1.0)

    # Price floor
    price_floor = base_price * PRICE_FLOOR_PCT

    # ─── Compute day-by-day prices ─────────────────────
    day_prices = []
    day_losses_per_100kg = []

    for d in range(days):
        if d == 0:
            price_d = base_price
        else:
            # Polynomial decay with tier acceleration
            t_frac = d / max(1, days - 1)
            exponent = 1 + 0.15 * max(0, oversupply_ratio - 1)
            decay = decay_rate * (t_frac ** exponent) * acceleration
            price_d = base_price * (1 - decay)
            price_d = max(price_floor, price_d)

        day_prices.append(round(price_d, 2))

        # Loss per 100kg compared to selling at base price today
        loss = round((base_price - price_d) * 100, 2)
        day_losses_per_100kg.append(loss)

    # ─── Break-even calculation ────────────────────────
    # First day where cost of waiting > transport cost
    break_even_day = None
    if transport_cost_per_100kg > 0:
        for d in range(days):
            if day_losses_per_100kg[d] > transport_cost_per_100kg:
                break_even_day = d
                break

    # ─── Summary stats ─────────────────────────────────
    total_loss_7d = day_losses_per_100kg[-1] if day_losses_per_100kg else 0
    loss_per_day_avg = round(total_loss_7d / max(1, days - 1), 2)

    # ─── Natural language recommendation ───────────────
    tier_name = {1: "highly perishable", 2: "moderately durable", 3: "storable"}
    tier_label = tier_name.get(tier, "storable")

    if break_even_day is not None and break_even_day <= 2:
        recommendation = (
            f"Act immediately — {crop_key} ({tier_label}) prices will drop "
            f"₹{day_losses_per_100kg[min(2, days-1)]:.0f}/100kg by day 3. "
            f"Transport cost breaks even on day {break_even_day}."
        )
    elif break_even_day is not None and break_even_day <= 4:
        recommendation = (
            f"Act within {break_even_day} days — prices decay "
            f"₹{loss_per_day_avg:.0f}/100kg per day. "
            f"After day {break_even_day}, waiting costs more than transport."
        )
    elif total_loss_7d > 500:
        recommendation = (
            f"Significant decay expected — ₹{total_loss_7d:.0f}/100kg loss "
            f"over {days} days. Redistribute to preserve value."
        )
    elif total_loss_7d > 200:
        recommendation = (
            f"Moderate decay — ₹{total_loss_7d:.0f}/100kg over {days} days. "
            f"Monitor and redistribute if urgency score rises."
        )
    else:
        recommendation = (
            f"Low decay risk — ₹{total_loss_7d:.0f}/100kg over {days} days. "
            f"Price stable enough to hold if storage permits."
        )

    return {
        "crop":                          crop_key,
        "base_price":                    base_price,
        "oversupply_ratio":              round(oversupply_ratio, 3),
        "perishability_tier":            tier,
        "perishability_label":           tier_label,
        "day_prices":                    day_prices,
        "day_losses_per_100kg":          day_losses_per_100kg,
        "break_even_day":                break_even_day,
        "total_projected_loss_7d_per_100kg": round(total_loss_7d, 2),
        "loss_per_day_avg":              loss_per_day_avg,
        "transport_cost_per_100kg":      transport_cost_per_100kg,
        "recommendation":               recommendation
    }
