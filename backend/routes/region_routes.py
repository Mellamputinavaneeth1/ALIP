from fastapi import APIRouter
from services.demand_ml_service import predict_dynamic_threshold
from services.region_service import get_region_info
from services.region_service import (
    get_nearest_regions,
    get_distance_between,
    get_all_regions,
    get_region_info
)

router = APIRouter()

# Get nearest regions for redistribution
@router.get("/regions/nearest/{region}/{crop}")
async def nearest_regions(region: str, crop: str,
                           max_km: float = 500.0):
    crop = crop.strip()
    region = region.strip()
    results = get_nearest_regions(
        region, crop,
        n=5,
        max_distance_km=max_km
    )
    if not results:
        return {
            "success": False,
            "message": f"Region '{region}' not found"
        }
    return {
        "success":        True,
        "source_region":  region,
        "crop":           crop,
        "recommendations": results,
        "total_found":    len(results)
    }

# Get distance between two regions
@router.get("/regions/distance/{region1}/{region2}")
async def region_distance(region1: str, region2: str):
    dist = get_distance_between(region1, region2)
    if dist < 0:
        return {
            "success": False,
            "message": "One or both regions not found"
        }
    return {
        "success":    True,
        "region1":    region1,
        "region2":    region2,
        "distance_km": dist
    }

# Get all available regions
@router.get("/regions/all")
async def all_regions():
    return {
        "success": True,
        "regions": get_all_regions(),
        "total":   len(get_all_regions())
    }

# Get single region info
@router.get("/regions/info/{region}")
async def region_info(region: str):
    info = get_region_info(region)
    if not info:
        return {
            "success": False,
            "message": f"Region '{region}' not found"
        }
    return {
        "success": True,
        "region":  region,
        "info":    info
    }
@router.get("/demand/threshold/{region}/{crop}")
async def dynamic_threshold(region: str, crop: str):
    region = region.strip()
    crop   = crop.strip()

    # Get region data
    info = get_region_info(region)
    if not info:
        return {"success": False,
                "message": f"Region '{region}' not found"}

    result = predict_dynamic_threshold(
        crop              = crop,
        region_type       = info.get("region_type","semi-urban"),
        population_density= info.get("population_density", 3000),
        market_size       = info.get("market_size", "medium")
    )
    return {"success": True, **result}
from services.demand_ml_service import (
    predict_dynamic_threshold,
    analyze_supply_trend
)
from fastapi import Body

@router.post("/demand/trend/{region}/{crop}")
async def supply_trend(
    region: str,
    crop: str,
    supply_history: list = Body(
        ...,
        example=[300, 450, 600, 800, 1100]
    )
):
    region = region.strip()
    crop   = crop.strip()

    # Get dynamic threshold first
    info = get_region_info(region)
    if not info:
        return {"success": False,
                "message": f"Region '{region}' not found"}

    threshold_result = predict_dynamic_threshold(
        crop               = crop,
        region_type        = info.get("region_type", "semi-urban"),
        population_density = info.get("population_density", 3000),
        market_size        = info.get("market_size", "medium")
    )
    threshold = threshold_result["predicted_threshold_kg"]

    # Analyze trend
    trend = analyze_supply_trend(supply_history, threshold)

    return {
        "success":          True,
        "region":           region,
        "crop":             crop,
        "dynamic_threshold":threshold,
        "trend_analysis":   trend
    }
from services.demand_ml_service import (
    predict_dynamic_threshold,
    analyze_supply_trend,
    estimate_price_impact
)

@router.get("/demand/price-impact/{region}/{crop}/{supply_kg}")
async def price_impact(
    region: str,
    crop: str,
    supply_kg: float
):
    region = region.strip()
    crop   = crop.strip()

    # Get dynamic threshold
    info = get_region_info(region)
    if not info:
        return {"success": False,
                "message": f"Region '{region}' not found"}

    threshold_result = predict_dynamic_threshold(
        crop               = crop,
        region_type        = info.get("region_type","semi-urban"),
        population_density = info.get("population_density",3000),
        market_size        = info.get("market_size","medium")
    )
    threshold = threshold_result["predicted_threshold_kg"]

    # Price impact
    impact = estimate_price_impact(crop, supply_kg, threshold)

    # Nearest regions for redistribution
    from services.region_service import get_nearest_regions
    nearby = get_nearest_regions(region, crop, n=3)

    return {
        "success":           True,
        "region":            region,
        "crop":              crop,
        "current_supply_kg": supply_kg,
        "dynamic_threshold": threshold,
        "price_impact":      impact,
        "redistribute_to":   nearby
    }


# ─── NEW: Full Unified Demand Analysis ─────────────────
from services.market_ranking_service import rank_destinations
from services.supply_history_service import record_supply
from services.shipment_service import (
    create_shipment, calculate_transport_cost,
    get_broker_shipments, update_shipment_status
)
from services.reliability_service import rate_broker, compute_reliability
from pydantic import BaseModel
from typing import Optional


class ShipmentRequest(BaseModel):
    broker_id: str
    source_region: str
    destination_region: str
    crop: str
    quantity_kg: float
    agreed_price: float = 0.0
    truck_type: str = "medium"
    expected_delivery_days: int = 2
    notes: str = ""


class RatingRequest(BaseModel):
    deal_id: str
    rater_broker_id: str
    rated_broker_id: str
    stars: int
    comment: str = ""


class SupplyRecordRequest(BaseModel):
    region: str
    crop: str
    total_supply_kg: float
    listing_count: int = 0


@router.get("/demand/analysis/{region}/{crop}/{supply_kg}")
async def full_demand_analysis(
    region: str,
    crop: str,
    supply_kg: float,
    quantity_to_send: float = 0,
    broker_id: str = "",
    truck_type: str = "medium",
    max_distance: float = 500.0,
    top_n: int = 5
):
    """
    Full unified demand analysis with weighted market ranking.

    Returns comprehensive analysis including:
    - Current state (threshold, oversupply ratio, Z-score)
    - Urgency assessment (1-10 score)
    - Price trajectory (7-day projection)
    - Top N ranked destinations with composite scores
    - Allocation summary
    - Seasonal context
    """
    region = region.strip()
    crop = crop.strip()

    # Also record this supply snapshot for Z-score history
    try:
        await record_supply(region, crop, supply_kg, 0)
    except Exception:
        pass  # Non-critical — don't fail the analysis

    try:
        result = await rank_destinations(
            source_region=region,
            crop=crop,
            current_supply_kg=supply_kg,
            quantity_to_send_kg=quantity_to_send,
            broker_id=broker_id,
            truck_type=truck_type,
            max_distance_km=max_distance,
            top_n=top_n
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis engine error: {str(e)}"
        }


@router.post("/demand/initiate-shipment")
async def initiate_shipment(req: ShipmentRequest):
    """
    Execute redistribution — broker initiates a shipment.

    Creates a shipment log entry, computes transport cost,
    and returns confirmation with waybill reference.
    """
    from services.region_service import get_distance_between

    # Calculate distance
    distance = get_distance_between(req.source_region, req.destination_region)
    if distance < 0:
        return {
            "success": False,
            "message": "One or both regions not found"
        }

    # Calculate transport cost
    transport = calculate_transport_cost(
        distance, req.quantity_kg, req.truck_type
    )

    # Create shipment
    shipment = await create_shipment(
        broker_id=req.broker_id,
        source_region=req.source_region,
        destination_region=req.destination_region,
        crop=req.crop,
        quantity_kg=req.quantity_kg,
        agreed_price=req.agreed_price,
        transport_cost=transport["total_cost"],
        expected_delivery_days=req.expected_delivery_days,
        notes=req.notes
    )

    return {
        "success": True,
        "message": "Shipment initiated successfully",
        "shipment": shipment,
        "transport": transport,
        "distance_km": distance
    }


@router.patch("/demand/shipment/{shipment_id}/status/{new_status}")
async def change_shipment_status(shipment_id: str, new_status: str):
    """Update shipment status (pending → in_transit → delivered / cancelled)."""
    result = await update_shipment_status(shipment_id, new_status)
    return {"success": True, "shipment": result}


@router.get("/demand/shipments/{broker_id}")
async def broker_shipments(
    broker_id: str,
    status: str = None,
    limit: int = 20
):
    """Get shipments for a broker."""
    shipments = await get_broker_shipments(broker_id, status, limit)
    return {
        "success": True,
        "broker_id": broker_id,
        "shipments": shipments,
        "total": len(shipments)
    }


@router.post("/demand/record-supply")
async def record_supply_snapshot(req: SupplyRecordRequest):
    """Record a supply snapshot for Z-score history building."""
    result = await record_supply(
        req.region, req.crop, req.total_supply_kg, req.listing_count
    )
    return {"success": True, **result}


@router.get("/demand/transport-cost/{region1}/{region2}/{quantity_kg}")
async def transport_cost(
    region1: str, region2: str, quantity_kg: float,
    truck_type: str = "medium"
):
    """Calculate transport cost between two regions."""
    from services.region_service import get_distance_between
    distance = get_distance_between(region1, region2)
    if distance < 0:
        return {"success": False, "message": "Region not found"}
    cost = calculate_transport_cost(distance, quantity_kg, truck_type)
    return {"success": True, **cost}


@router.post("/demand/rate-broker")
async def rate_broker_endpoint(req: RatingRequest):
    """Submit a rating for a broker after a completed deal."""
    result = await rate_broker(
        deal_id=req.deal_id,
        rater_broker_id=req.rater_broker_id,
        rated_broker_id=req.rated_broker_id,
        stars=req.stars,
        comment=req.comment
    )
    return result


@router.get("/demand/broker-reliability/{broker_id}")
async def broker_reliability(broker_id: str):
    """Get reliability score for a broker."""
    result = await compute_reliability(broker_id)
    return {"success": True, **result}