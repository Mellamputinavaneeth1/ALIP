from fastapi import APIRouter, UploadFile, File, HTTPException
from services.disease_service import predict_disease

router = APIRouter(prefix="/disease", tags=["Disease Detection"])

@router.post("/predict")
async def detect_disease(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed"
        )

    # Read image bytes
    image_bytes = await file.read()

    # Run prediction
    result = await predict_disease(image_bytes)

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {result['error']}"
        )

    return result