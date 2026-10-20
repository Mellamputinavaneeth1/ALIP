"""
ALIP — Weighted Market Ranking Engine
The final assembly layer that combines all 7 intelligence components
into a unified composite score per destination, then ranks them.
Includes destination saturation, storage validation, and temporal timing.
"""

import datetime
from typing import Dict, List, Optional

from services.region_service import (
    REGIONS, haversine_distance, get_region_info, _find_region
)
from services.demand_ml_service import (
    predict_dynamic_threshold,
    TRANSPORT_RATES, DEFAULT_TRUCK_TYPE, PERISHABILITY_TIERS
)
from services.shipment_service import (
    calculate_transport_cost, get_incoming_to_destination
)
from services.supply_history_service import (
    get_supply_history, compute_zscore
)
from services.urgency_service import compute_urgency
from services.price_trajectory_service import compute_price_trajectory
from services.reliability_service import compute_reliability
from services.seasonal_context_service import get_seasonal_context
from services.weather_risk_service import assess_weather_risk


# ─── Factor Weights ────────────────────────────────────
WEIGHTS = {
    "demand_gap":       0.20,
    "price_advantage":  0.15,
    "transport":        0.15,
    "reliability":      0.20,
    "storage":          0.10,
    "weather":          0.10,
    "saturation":       0.10,
}

MAX_REASONABLE_TRANSPORT = 500.0   # ₹500/100kg = very far
COLD_STORAGE_REQUIRED_TIERS = {1}  # Tier 1 crops REQUIRE cold storage
DRY_STORAGE_PREFERRED_TIERS = {2}  # Tier 2 crops PREFER dry storage


# ─── Saturation Score (P3.1) ──────────────────────────
async def compute_saturation_score(
    destination: str,
    crop: str,
    destination_deficit_kg: float,
    hours: int = 24
) -> dict:
    """
    Compute destination saturation before recommending.
    Prevents moving oversupply from one place to another.

    Args:
        destination: Target region name
        crop: Crop name
        destination_deficit_kg: How much the destination needs
        hours: Look-back window for active shipments

    Returns:
        dict with saturation_ratio, score (0-100), label,
        and safe_to_send quantity.
    """
    try:
        incoming = await get_incoming_to_destination(destination, crop, hours)
    except Exception:
        incoming = {"pending_incoming_kg": 0, "shipment_count": 0}
    pending_kg = incoming.get("pending_incoming_kg", 0)

    if destination_deficit_kg <= 0:
        # Already at or above threshold — extremely saturated
        return {
            "saturation_ratio": 10.0,
            "score": 0,
            "label": "CRITICAL",
            "safe_to_send_kg": 0,
            "pending_incoming_kg": pending_kg,
            "shipment_count": incoming.get("shipment_count", 0)
        }

    ratio = pending_kg / destination_deficit_kg
    score = max(0, round(100 - ratio * 100))

    if ratio < 0.4:
        label = "LOW"
    elif ratio < 0.8:
        label = "MEDIUM"
    elif ratio < 1.2:
        label = "HIGH"
    else:
        label = "CRITICAL"

    safe_to_send = max(0, round(destination_deficit_kg - pending_kg, 2))

    return {
        "saturation_ratio": round(ratio, 3),
        "score": score,
        "label": label,
        "safe_to_send_kg": safe_to_send,
        "pending_incoming_kg": pending_kg,
        "shipment_count": incoming.get("shipment_count", 0)
    }


# ─── Storage Validation (P3.3) ────────────────────────
async def validate_storage(
    destination_broker_id: str,
    crop: str,
    quantity_kg: float
) -> dict:
    """
    Validate destination broker has appropriate storage.

    Returns:
        dict with storage_score (0-100), storage_type_match,
        max_send_kg, and any exclusion flag.
    """
    from config.database import users_collection
    from bson import ObjectId

    tier = PERISHABILITY_TIERS.get(crop.strip().lower(), 3)

    # Try to fetch broker profile
    try:
        broker = await users_collection.find_one(
            {"_id": ObjectId(destination_broker_id)}
        )
    except Exception:
        broker = None

    if not broker or "storage_capacity" not in broker:
        # No storage data — use neutral default
        return {
            "storage_score": 50,
            "storage_type_match": "unknown",
            "max_send_kg": quantity_kg,
            "excluded": False,
            "reason": "No storage data available — using default"
        }

    storage = broker["storage_capacity"]
    cold = storage.get("cold_storage_kg", 0)
    dry = storage.get("dry_storage_kg", 0)
    yard = storage.get("open_yard_kg", 0)

    # Determine relevant storage by tier
    if tier == 1:
        # REQUIRES cold storage
        if cold <= 0:
            return {
                "storage_score": 0,
                "storage_type_match": "EXCLUDED",
                "max_send_kg": 0,
                "excluded": True,
                "reason": f"No cold storage at destination — excluded for tier 1 crop"
            }
        relevant = cold
        storage_type = "cold_storage"
    elif tier == 2:
        # PREFERS dry storage, open_yard acceptable
        if dry > 0:
            relevant = dry
            storage_type = "dry_storage"
        elif yard > 0:
            relevant = yard
            storage_type = "open_yard"
        else:
            return {
                "storage_score": 0,
                "storage_type_match": "EXCLUDED",
                "max_send_kg": 0,
                "excluded": True,
                "reason": "No dry or open yard storage — excluded for tier 2 crop"
            }
    else:
        # Tier 3: any storage type works
        relevant = dry + yard + cold
        storage_type = "any"
        if relevant <= 0:
            return {
                "storage_score": 0,
                "storage_type_match": "EXCLUDED",
                "max_send_kg": 0,
                "excluded": True,
                "reason": "No storage available — excluded"
            }

    max_send = min(quantity_kg, relevant)
    score = min(100, round((relevant / max(1, quantity_kg)) * 100))

    reason = None
    if relevant < quantity_kg:
        reason = f"Storage limited — max send: {relevant}kg"

    return {
        "storage_score": score,
        "storage_type_match": storage_type,
        "max_send_kg": round(max_send, 2),
        "excluded": False,
        "reason": reason,
        "available_kg": round(relevant, 2)
    }


# ─── Temporal Recommendation (P4.2) ──────────────────
def compute_temporal_recommendation(
    weather_data: dict,
    urgency_data: dict,
    crop: str
) -> dict:
    """
    Combine weather timing + urgency to give exact send timing.

    Returns:
        dict with timing string, recommended day, and reasoning.
    """
    tier = PERISHABILITY_TIERS.get(crop.strip().lower(), 3)
    act_within = urgency_data.get("act_within_days", 14)

    weather_level = weather_data.get("weather_level", "LOW")
    send_day = weather_data.get("send_day_recommendation", 0)
    weather_timing = weather_data.get("optimal_timing", "Send when ready")

    # Decision logic
    if weather_level in ("LOW", "MEDIUM") and send_day == 0:
        timing = "Send TODAY — weather is clear"
        recommended_day = 0
    elif weather_level == "HIGH" and tier == 1:
        timing = "Cannot wait — send in rain or find closer market"
        recommended_day = 0
    elif send_day <= act_within:
        timing = f"Wait {send_day} day(s) — {weather_timing}"
        recommended_day = send_day
    else:
        timing = f"Send within {act_within} day(s) (urgency window)"
        recommended_day = min(send_day, act_within)

    return {
        "timing": timing,
        "recommended_day": recommended_day,
        "weather_consideration": weather_timing,
        "urgency_window_days": act_within
    }


# ─── Main Ranking Engine (P4.1) ──────────────────────
async def rank_destinations(
    source_region: str,
    crop: str,
    current_supply_kg: float,
    quantity_to_send_kg: float = 0,
    broker_id: str = "",
    truck_type: str = "medium",
    max_distance_km: float = 500.0,
    top_n: int = 5
) -> dict:
    """
    The main orchestrator: ranks all candidate destinations
    using the 7-factor weighted composite score.

    Args:
        source_region: Source region name
        crop: Crop name
        current_supply_kg: Current supply at source
        quantity_to_send_kg: Amount broker wants to send (0 = auto)
        broker_id: Broker ID for shipment tracking
        truck_type: Truck type for transport cost
        max_distance_km: Max distance to consider
        top_n: Return top N destinations

    Returns:
        Full analysis dict with destinations, urgency,
        price trajectory, seasonal context, and allocation summary.
    """
    crop_key = crop.strip().lower()
    src_key = _find_region(source_region)

    if not src_key:
        return {"success": False, "error": f"Region '{source_region}' not found"}

    src_data = REGIONS[src_key]
    now = datetime.datetime.utcnow()

    # ─── 1. Dynamic Threshold ─────────────────────────
    threshold_result = predict_dynamic_threshold(
        crop=crop_key,
        region_type=src_data.get("region_type", "semi-urban"),
        population_density=src_data.get("population_density", 3000),
        market_size=src_data.get("market_size", "medium")
    )
    threshold_kg = threshold_result["predicted_threshold_kg"]
    oversupply_ratio = round(current_supply_kg / max(1, threshold_kg), 3)

    # Auto-calculate quantity if not specified
    if quantity_to_send_kg <= 0:
        quantity_to_send_kg = max(0, current_supply_kg - threshold_kg)

    # ─── Early return if NO oversupply ────────────────
    if oversupply_ratio <= 1.0:
        seasonal = get_seasonal_context(crop=crop_key, supply_trend="stable", region=src_key)
        return {
            "success": True,
            "region": src_key,
            "crop": crop_key,
            "analysis_timestamp": now.isoformat() + "Z",
            "current_state": {
                "current_supply_kg": current_supply_kg,
                "dynamic_threshold_kg": threshold_kg,
                "oversupply_ratio": oversupply_ratio,
                "z_score": 0,
                "anomaly_level": "normal",
                "supply_trend": "stable",
                "listing_count": 0
            },
            "urgency": {
                "score": 1,
                "label": "WATCH",
                "color": "green",
                "act_within_days": 14,
                "reasoning": f"Supply ({current_supply_kg}kg) is within the healthy threshold ({threshold_kg}kg). No redistribution needed.",
                "components": {}
            },
            "price_trajectory": compute_price_trajectory(
                crop=crop_key, oversupply_ratio=oversupply_ratio
            ),
            "destinations": [],
            "excluded_destinations": [],
            "allocation_summary": {
                "total_excess_kg": 0,
                "recommended_send_kg": 0,
                "keep_local_kg": current_supply_kg,
                "net_savings_vs_holding": 0,
                "destinations_count": 0
            },
            "seasonal_context": seasonal,
            "execution_ready": False,
            "message": "Supply is within healthy threshold — no redistribution needed."
        }

    # ─── Quick DB Connectivity Check ──────────────────
    # Prevents 5s timeout × 40 destinations cascade when MongoDB is down
    db_available = False
    try:
        from config.database import db
        await db.command("ping", maxTimeMS=1500)
        db_available = True
    except Exception:
        pass  # DB unreachable — use offline fallbacks

    # ─── 2. Z-Score ───────────────────────────────────
    history = []
    if db_available:
        try:
            history = await get_supply_history(src_key, crop_key, days=30)
        except Exception:
            history = []
    history_values = [h["total_supply_kg"] for h in history]
    z_result = compute_zscore(history_values, current_supply_kg)

    # ─── 3. Urgency ───────────────────────────────────
    # Estimate days since detected (simplified: check history)
    days_oversupply = 0
    for h in reversed(history):
        if h["total_supply_kg"] > threshold_kg:
            days_oversupply += 1
        else:
            break

    # Get supply trend from history
    if len(history_values) >= 2:
        recent_change = history_values[-1] - history_values[-2] if len(history_values) >= 2 else 0
        pct = (recent_change / max(1, history_values[-2])) * 100 if len(history_values) >= 2 else 0
        if pct > 15:
            supply_trend = "rapidly_increasing"
        elif pct > 5:
            supply_trend = "increasing"
        elif pct < -15:
            supply_trend = "rapidly_decreasing"
        elif pct < -5:
            supply_trend = "decreasing"
        else:
            supply_trend = "stable"
    else:
        supply_trend = "stable"

    urgency = compute_urgency(
        oversupply_ratio=oversupply_ratio,
        z_score=z_result.get("z_score", 0),
        supply_trend=supply_trend,
        crop=crop_key,
        days_since_detected=days_oversupply
    )

    # ─── 4. Price Trajectory ──────────────────────────
    # Guard transport cost against zero quantity
    effective_qty = max(1, quantity_to_send_kg)
    avg_transport = calculate_transport_cost(200, effective_qty, truck_type)
    price_traj = compute_price_trajectory(
        crop=crop_key,
        oversupply_ratio=oversupply_ratio,
        transport_cost_per_100kg=avg_transport.get("cost_per_100kg", 0)
    )

    # ─── 5. Seasonal Context ─────────────────────────
    seasonal = get_seasonal_context(
        crop=crop_key,
        supply_trend=supply_trend,
        region=src_key
    )

    # ─── 6. Score Each Destination ────────────────────
    destinations = []
    excluded = []

    for dest_name, dest_data in REGIONS.items():
        if dest_name == src_key:
            continue

        # Distance filter
        dist = haversine_distance(
            src_data["lat"], src_data["lng"],
            dest_data["lat"], dest_data["lng"]
        )
        if dist > max_distance_km:
            continue

        # Destination threshold
        dest_threshold = predict_dynamic_threshold(
            crop=crop_key,
            region_type=dest_data.get("region_type", "semi-urban"),
            population_density=dest_data.get("population_density", 3000),
            market_size=dest_data.get("market_size", "medium")
        )["predicted_threshold_kg"]

        # Destination deficit (how much they need)
        # For now estimate 50% of their threshold as current supply
        est_dest_supply = dest_threshold * 0.5
        dest_deficit = max(0, dest_threshold - est_dest_supply)

        # ─── Factor Scores ────────────────────────────

        # a. Demand Gap (0-100)
        demand_gap_score = min(100, round(
            (dest_deficit / max(1, quantity_to_send_kg)) * 100
        ))

        # b. Price Advantage (0-100)
        # Price at day 3 (typical transit time)
        price_day3 = price_traj["day_prices"][min(3, len(price_traj["day_prices"]) - 1)]
        base_price = price_traj["base_price"]
        price_adv_score = min(100, max(0, round(
            ((base_price - price_day3) / max(0.01, base_price)) * 1000
        )))

        # c. Transport (0-100, lower cost = higher score)
        transport = calculate_transport_cost(dist, max(1, quantity_to_send_kg), truck_type)
        transport_cost_100kg = transport.get("cost_per_100kg", 0)
        transport_score = max(0, round(
            100 - (transport_cost_100kg / MAX_REASONABLE_TRANSPORT * 100)
        ))

        # d. Reliability (0-100) — use default for now
        reliability_score = 65  # Default neutral score

        # e. Saturation (0-100)
        if db_available:
            try:
                sat = await compute_saturation_score(
                    dest_name, crop_key, dest_deficit
                )
            except Exception:
                sat = {"score": 80, "label": "LOW", "safe_to_send_kg": quantity_to_send_kg,
                       "pending_incoming_kg": 0, "shipment_count": 0, "saturation_ratio": 0}
        else:
            sat = {"score": 80, "label": "LOW", "safe_to_send_kg": quantity_to_send_kg,
                   "pending_incoming_kg": 0, "shipment_count": 0, "saturation_ratio": 0}
        saturation_score = sat["score"]
        safe_to_send = sat["safe_to_send_kg"]

        # f. Weather (0-100)
        # Skip external API if no key configured (avoids per-dest timeout)
        import os
        if not os.getenv("WEATHER_API_KEY", ""):
            weather = {"weather_score": 75, "weather_level": "LOW",
                       "optimal_timing": "Weather data unavailable — no API key",
                       "send_day_recommendation": 0}
            weather_score = 75
            weather_level = "LOW"
        else:
            try:
                weather = await assess_weather_risk(
                    src_data["lat"], src_data["lng"],
                    dest_data["lat"], dest_data["lng"],
                    crop_key
                )
                weather_score = weather.get("weather_score", 75)
                weather_level = weather.get("weather_level", "LOW")
            except Exception:
                weather = {"weather_score": 75, "weather_level": "LOW",
                           "optimal_timing": "Weather data unavailable",
                           "send_day_recommendation": 0}
                weather_score = 75
                weather_level = "LOW"

        # g. Storage — use default (no broker ID for destination)
        storage_score = 50  # Neutral default

        # ─── Hard Exclusions ──────────────────────────
        exclusion_reason = None
        if storage_score == 0:
            exclusion_reason = "Storage unavailable"
        elif sat["label"] == "CRITICAL":
            exclusion_reason = f"Saturation CRITICAL — already {sat['pending_incoming_kg']}kg incoming"
        elif weather_level == "CRITICAL" and PERISHABILITY_TIERS.get(crop_key, 3) == 1:
            exclusion_reason = "Weather CRITICAL for perishable crop"
        elif dest_deficit <= 0:
            exclusion_reason = "Destination already at/above threshold"

        if exclusion_reason:
            excluded.append({
                "region": dest_name,
                "state": dest_data["state"],
                "distance_km": dist,
                "reason": exclusion_reason
            })
            continue

        # ─── Composite Score ──────────────────────────
        composite = round(
            demand_gap_score * WEIGHTS["demand_gap"] +
            price_adv_score * WEIGHTS["price_advantage"] +
            transport_score * WEIGHTS["transport"] +
            reliability_score * WEIGHTS["reliability"] +
            storage_score * WEIGHTS["storage"] +
            weather_score * WEIGHTS["weather"] +
            saturation_score * WEIGHTS["saturation"]
        )

        # Star rating
        if composite >= 80:
            star_rating, rec_label = 5, "STRONGLY RECOMMENDED"
        elif composite >= 65:
            star_rating, rec_label = 4, "RECOMMENDED"
        elif composite >= 50:
            star_rating, rec_label = 3, "FEASIBLE"
        elif composite >= 35:
            star_rating, rec_label = 2, "MARGINAL"
        else:
            star_rating, rec_label = 1, "NOT RECOMMENDED"

        # ROI calculation
        price_gain = price_traj["day_losses_per_100kg"][min(3, len(price_traj["day_losses_per_100kg"])-1)]
        roi = round(price_gain / max(0.01, transport_cost_100kg), 2) if transport_cost_100kg > 0 else 0
        roi_verdict = (
            "STRONGLY RECOMMENDED" if roi > 3
            else "RECOMMENDED" if roi > 1.5
            else "MARGINAL" if roi > 1
            else "NOT RECOMMENDED"
        )

        # Temporal recommendation
        temporal = compute_temporal_recommendation(weather, urgency, crop_key)

        # Quantity to send (capped by saturation)
        send_qty = min(quantity_to_send_kg, safe_to_send)

        destinations.append({
            "region": dest_name,
            "state": dest_data["state"],
            "distance_km": dist,
            "composite_score": composite,
            "star_rating": star_rating,
            "recommendation": rec_label,
            "quantity_to_send": round(send_qty, 2),
            "timing": temporal["timing"],
            "send_day": temporal["recommended_day"],
            "transport_cost_per_100kg": transport_cost_100kg,
            "transport_total": transport["total_cost"],
            "roi": roi,
            "roi_verdict": roi_verdict,
            "saturation": sat["label"],
            "weather_risk": weather_level,
            "breakdown": {
                "demand_gap_score": demand_gap_score,
                "price_advantage_score": price_adv_score,
                "transport_score": transport_score,
                "reliability_score": reliability_score,
                "storage_score": storage_score,
                "weather_score": weather_score,
                "saturation_score": saturation_score
            },
            "flags": []
        })

        # Add flags
        if weather_level in ("HIGH", "CRITICAL"):
            destinations[-1]["flags"].append(f"Weather {weather_level} — {temporal['weather_consideration']}")
        if sat["label"] in ("MEDIUM", "HIGH"):
            destinations[-1]["flags"].append(f"Saturation {sat['label']} — safe to send: {safe_to_send}kg")
        if roi > 3:
            destinations[-1]["flags"].append(f"High ROI: {roi}x return vs transport cost")

    # Sort by composite score descending
    destinations.sort(key=lambda x: x["composite_score"], reverse=True)
    top_destinations = destinations[:top_n]

    # ─── Allocation Summary ───────────────────────────
    total_excess = max(0, current_supply_kg - threshold_kg)
    recommended_send = min(total_excess, sum(d["quantity_to_send"] for d in top_destinations))
    net_savings = round(
        recommended_send / 100 * price_traj.get("total_projected_loss_7d_per_100kg", 0), 2
    )

    return {
        "success": True,
        "region": src_key,
        "crop": crop_key,
        "analysis_timestamp": now.isoformat() + "Z",

        "current_state": {
            "current_supply_kg": current_supply_kg,
            "dynamic_threshold_kg": threshold_kg,
            "oversupply_ratio": oversupply_ratio,
            "z_score": z_result.get("z_score", 0),
            "anomaly_level": z_result.get("anomaly_level", "unknown"),
            "supply_trend": supply_trend,
            "listing_count": len(history)
        },

        "urgency": urgency,

        "price_trajectory": price_traj,

        "destinations": top_destinations,
        "excluded_destinations": excluded[:5],

        "allocation_summary": {
            "total_excess_kg": round(total_excess, 2),
            "recommended_send_kg": round(recommended_send, 2),
            "keep_local_kg": round(current_supply_kg - recommended_send, 2),
            "net_savings_vs_holding": net_savings,
            "destinations_count": len(top_destinations)
        },

        "seasonal_context": seasonal,

        "execution_ready": len(top_destinations) > 0
    }
