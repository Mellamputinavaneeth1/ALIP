"""
ALIP — Broker Reliability Score Service
Scores brokers 0-100 based on deal completion, response time,
and adherence. Uses Bayesian adjustment for fair cold-start handling.
"""

import datetime
from typing import Dict, Optional
from config.database import db
from bson import ObjectId

# ─── Collections ───────────────────────────────────────
deal_ratings_collection = db["deal_ratings"]
shipment_log_collection = db["shipment_log"]

# ─── Constants ─────────────────────────────────────────
MIN_DEALS_FOR_SCORE = 5       # Below this, use default score
DEFAULT_RELIABILITY = 65      # Neutral score for new brokers

# Weight distribution
COMPLETION_WEIGHT = 0.50
RESPONSE_WEIGHT   = 0.25
ADHERENCE_WEIGHT  = 0.25


def _star_rating(score: float) -> dict:
    """Convert reliability score to star display."""
    if score >= 80:
        stars, display = 5, "★★★★★"
    elif score >= 60:
        stars, display = 4, "★★★★☆"
    elif score >= 40:
        stars, display = 3, "★★★☆☆"
    elif score >= 20:
        stars, display = 2, "★★☆☆☆"
    else:
        stars, display = 1, "★☆☆☆☆"
    return {"stars": stars, "display": display}


def _response_time_score(avg_hours: float) -> int:
    """Score based on average response time in hours."""
    if avg_hours < 4:
        return 100
    elif avg_hours < 12:
        return 75
    elif avg_hours < 24:
        return 50
    else:
        return 25


async def compute_reliability(broker_id: str) -> dict:
    """
    Compute composite reliability score for a broker.

    Three sub-components:
    a. Completion rate (50% weight) — Bayesian adjusted
    b. Response time score (25% weight)
    c. Deal adherence score (25% weight) — from ratings

    Returns:
        dict with score (0-100), star_rating, label, components,
        and whether the broker has sufficient history.
    """
    # ─── Completion Rate (from shipment_log) ───────────
    total_deals = await shipment_log_collection.count_documents(
        {"broker_id": broker_id}
    )
    completed_deals = await shipment_log_collection.count_documents(
        {"broker_id": broker_id, "status": "delivered"}
    )

    # Bayesian adjusted: (completed + 2) / (total + 4)
    # Prevents 2/2 = 100% outranking 48/50 = 96%
    adjusted_completion = (
        (completed_deals + 2) / (total_deals + 4) * 100
        if total_deals > 0 else DEFAULT_RELIABILITY
    )

    # ─── Response Time (avg hours from initiation to first status change) ─
    # For now, estimate from shipments that went to in_transit or delivered
    pipeline = [
        {"$match": {
            "broker_id": broker_id,
            "status": {"$in": ["in_transit", "delivered"]},
            "actual_delivery": {"$ne": None}
        }},
        {"$project": {
            "response_hours": {
                "$divide": [
                    {"$subtract": [
                        {"$ifNull": ["$actual_delivery", "$expected_delivery"]},
                        "$initiated_at"
                    ]},
                    3600000  # ms to hours
                ]
            }
        }},
        {"$group": {
            "_id": None,
            "avg_response_hours": {"$avg": "$response_hours"}
        }}
    ]
    response_result = await shipment_log_collection.aggregate(
        pipeline
    ).to_list(1)

    avg_response_hours = (
        response_result[0]["avg_response_hours"]
        if response_result else 12.0  # default: 12 hours
    )
    response_score = _response_time_score(avg_response_hours)

    # ─── Adherence Score (from deal_ratings) ───────────
    ratings_pipeline = [
        {"$match": {"rated_broker_id": broker_id}},
        {"$group": {
            "_id": None,
            "avg_stars": {"$avg": "$stars"},
            "count": {"$sum": 1}
        }}
    ]
    ratings_result = await deal_ratings_collection.aggregate(
        ratings_pipeline
    ).to_list(1)

    if ratings_result and ratings_result[0]["count"] > 0:
        avg_stars = ratings_result[0]["avg_stars"]
        adherence_score = (avg_stars / 5.0) * 100
        rating_count = ratings_result[0]["count"]
    else:
        adherence_score = DEFAULT_RELIABILITY
        avg_stars = 0
        rating_count = 0

    # ─── Composite Score ───────────────────────────────
    sufficient_history = total_deals >= MIN_DEALS_FOR_SCORE

    if sufficient_history:
        composite = (
            adjusted_completion * COMPLETION_WEIGHT +
            response_score * RESPONSE_WEIGHT +
            adherence_score * ADHERENCE_WEIGHT
        )
    else:
        # Cold start: use default with slight adjustment
        composite = DEFAULT_RELIABILITY
        if total_deals > 0:
            # Blend default with actual data
            blend_factor = total_deals / MIN_DEALS_FOR_SCORE
            actual = (
                adjusted_completion * COMPLETION_WEIGHT +
                response_score * RESPONSE_WEIGHT +
                adherence_score * ADHERENCE_WEIGHT
            )
            composite = (
                DEFAULT_RELIABILITY * (1 - blend_factor) +
                actual * blend_factor
            )

    composite = round(min(100, max(0, composite)), 1)
    stars = _star_rating(composite)

    # Label
    if not sufficient_history:
        label = "New — limited history"
    elif composite >= 80:
        label = "Highly Reliable"
    elif composite >= 60:
        label = "Reliable"
    elif composite >= 40:
        label = "Moderate"
    else:
        label = "Unreliable"

    return {
        "broker_id":          broker_id,
        "reliability_score":  composite,
        "star_rating":        stars,
        "label":              label,
        "sufficient_history": sufficient_history,
        "components": {
            "completion": {
                "total_deals":     total_deals,
                "completed_deals": completed_deals,
                "adjusted_rate":   round(adjusted_completion, 1),
                "weight":          COMPLETION_WEIGHT
            },
            "response_time": {
                "avg_hours":  round(avg_response_hours, 1),
                "score":      response_score,
                "weight":     RESPONSE_WEIGHT
            },
            "adherence": {
                "avg_stars":     round(avg_stars, 1) if avg_stars else 0,
                "rating_count":  rating_count,
                "score":         round(adherence_score, 1),
                "weight":        ADHERENCE_WEIGHT
            }
        }
    }


async def rate_broker(
    deal_id: str,
    rater_broker_id: str,
    rated_broker_id: str,
    stars: int,
    comment: str = ""
) -> dict:
    """
    Submit a rating for a broker after a completed deal.

    Args:
        deal_id: The shipment/deal ID
        rater_broker_id: Who is rating
        rated_broker_id: Who is being rated
        stars: 1-5 star rating
        comment: Optional text feedback

    Returns:
        dict with the created rating document info
    """
    if not 1 <= stars <= 5:
        raise ValueError("Stars must be between 1 and 5")

    rating_doc = {
        "deal_id":          deal_id,
        "rater_broker_id":  rater_broker_id,
        "rated_broker_id":  rated_broker_id,
        "stars":            stars,
        "comment":          comment,
        "rated_at":         datetime.datetime.utcnow()
    }

    result = await deal_ratings_collection.insert_one(rating_doc)

    return {
        "success":    True,
        "rating_id":  str(result.inserted_id),
        "deal_id":    deal_id,
        "stars":      stars,
        "message":    f"Rating of {stars} stars submitted successfully"
    }


async def get_broker_ratings(
    broker_id: str,
    limit: int = 20
) -> list:
    """Get ratings received by a broker."""
    cursor = deal_ratings_collection.find(
        {"rated_broker_id": broker_id}
    ).sort("rated_at", -1).limit(limit)

    ratings = []
    async for doc in cursor:
        ratings.append({
            "rating_id":        str(doc["_id"]),
            "deal_id":          doc.get("deal_id"),
            "rater_broker_id":  doc.get("rater_broker_id"),
            "stars":            doc.get("stars"),
            "comment":          doc.get("comment", ""),
            "rated_at":         str(doc.get("rated_at", ""))
        })
    return ratings
