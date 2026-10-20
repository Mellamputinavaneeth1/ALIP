from fastapi import APIRouter, HTTPException
from typing import List, Optional
from services.weather_service import get_farm_dashboard

router = APIRouter(prefix="/weather", tags=["Weather Advisory"])


@router.get("/dashboard/{location}")
async def get_weather_dashboard(
    location: str,
    crops: Optional[str] = None  # comma-separated: "tomato,potato"
):
    """
    Main weather dashboard for the farmer portal.
    
    Args:
        location: City name, e.g. 'Hyderabad, IN'
        crops: Optional comma-separated list of crops the farmer grows,
               e.g. 'tomato,potato'. Used to generate crop-specific disease risk.
               Defaults to all supported crops (tomato, potato, corn, pepper).
    """
    crop_list: Optional[List[str]] = None
    if crops:
        crop_list = [c.strip().lower() for c in crops.split(",") if c.strip()]

    result = await get_farm_dashboard(location, crops=crop_list)

    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result