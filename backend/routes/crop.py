from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.crop_service import recommend_crop, recommend_crop_manual

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])


class CropInput(BaseModel):
    location: str
    season: str


class ManualCropInput(BaseModel):
    nitrogen: float
    phosphorous: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    season: str = "kharif"
    location: Optional[str] = ""


class RegionalCropRequest(BaseModel):
    location:    Optional[str]   = None    # District name, e.g. "Nashik"
    latitude:    Optional[float] = None    # GPS lat
    longitude:   Optional[float] = None    # GPS lon
    season:      str                       # "kharif" or "rabi"
    measured_ph: Optional[float] = None    # Optional farmer soil-test pH


@router.post("/recommend")
async def get_crop_recommendation(data: CropInput):
    """Auto mode — uses location to look up soil data and fetch live weather."""
    result = await recommend_crop(
        location=data.location,
        season=data.season
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/recommend/manual")
async def get_crop_recommendation_manual(data: ManualCropInput):
    """
    Manual mode — farmer enters N, P, K, pH, temperature, humidity, rainfall
    directly from a soil test report / Soil Health Card.
    """
    result = await recommend_crop_manual(
        nitrogen=data.nitrogen,
        phosphorous=data.phosphorous,
        potassium=data.potassium,
        temperature=data.temperature,
        humidity=data.humidity,
        ph=data.ph,
        rainfall=data.rainfall,
        season=data.season,
        location=data.location or "",
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# ── Regional Recommendation (v2 — District-based) ─────────────────────────────

@router.post("/recommend/regional")
async def regional_crop_recommendation(req: RegionalCropRequest):
    """
    Production regional crop ranking endpoint.
    Uses real district-level soil, climate, and irrigation data
    from precomputed feature lookup. Returns top-3 crop recommendations
    with confidence bands and SHAP-derived reasons.

    Inputs: location name OR lat/lon + season + optional measured pH.
    """
    if not req.location and (req.latitude is None or req.longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'location' (district name) or 'latitude'+'longitude'"
        )
    if req.season.lower() not in ("kharif", "rabi"):
        raise HTTPException(status_code=400, detail="season must be 'kharif' or 'rabi'")

    from services.regional_crop_service import recommend_regional
    result = await recommend_regional(
        location=req.location,
        latitude=req.latitude,
        longitude=req.longitude,
        season=req.season.lower(),
        measured_ph=req.measured_ph,
    )
    if not result.get("success"):
        raise HTTPException(status_code=404 if "not found" in result.get("error","").lower() else 500,
                            detail=result.get("error", "Unknown error"))
    return result


@router.get("/districts")
async def list_districts():
    """Returns all supported districts for the regional recommendation model."""
    from services.district_service import list_supported_districts
    return {"districts": list_supported_districts(), "count": len(list_supported_districts())}


@router.get("/district/{district_id}")
async def get_district(district_id: str):
    """Returns environmental context and crop calendar for a specific district."""
    from services.district_service import get_district_info
    info = get_district_info(district_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"District '{district_id}' not supported")
    return info