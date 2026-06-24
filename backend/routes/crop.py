from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.crop_service import recommend_crop

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])

class CropInput(BaseModel):
    location: str
    season: str

@router.post("/recommend")
async def get_crop_recommendation(data: CropInput):
    result = await recommend_crop(
        location=data.location,
        season=data.season
    )

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result["error"]
        )

    return result