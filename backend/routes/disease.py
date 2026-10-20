from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.disease_service import predict_disease
from typing import Optional, List
from config.database import disease_scans_collection
import datetime
import uuid

router = APIRouter(prefix="/disease", tags=["Disease Detection"])

# In-memory store for instant fallback & fast lookups
_IN_MEMORY_SCANS = []


async def save_scan_record(record: dict):
    """Save scan to MongoDB and in-memory cache."""
    _IN_MEMORY_SCANS.insert(0, record)
    # Keep memory bounded
    if len(_IN_MEMORY_SCANS) > 100:
        _IN_MEMORY_SCANS.pop()

    try:
        doc = dict(record)
        doc["_id"] = record["scan_id"]
        await disease_scans_collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    except Exception as e:
        print(f"[WARN] Failed to persist disease scan to MongoDB: {e}")


async def get_active_scans_for_location(location: Optional[str] = None) -> List[dict]:
    """Retrieve active disease scans matching location or recent active ones."""
    scans = []
    try:
        query = {"active": True}
        if location and location.lower() != "all":
            query["location"] = {"$regex": location.split(",")[0].strip(), "$options": "i"}
        cursor = disease_scans_collection.find(query).sort("created_at", -1).limit(20)
        async for doc in cursor:
            doc_id = doc.get("scan_id") or str(doc.get("_id"))
            doc["scan_id"] = doc_id
            doc.pop("_id", None)
            scans.append(doc)
    except Exception as e:
        print(f"[WARN] MongoDB active scans query failed, using in-memory: {e}")

    if not scans:
        loc_term = location.split(",")[0].strip().lower() if location else ""
        for s in _IN_MEMORY_SCANS:
            if s.get("active", True):
                s_loc = (s.get("location") or "").lower()
                if not loc_term or loc_term == "all" or loc_term in s_loc or s_loc in loc_term:
                    scans.append(s)

    if not scans and _IN_MEMORY_SCANS:
        scans = [s for s in _IN_MEMORY_SCANS if s.get("active", True)][:5]

    return scans


@router.post("/predict")
async def detect_disease(
    file: UploadFile = File(...),
    location: Optional[str] = Form(None)
):
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed"
        )

    # Read image bytes
    image_bytes = await file.read()

    # Run prediction (with optional location for weather-aware analysis)
    result = await predict_disease(image_bytes, location=location)

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {result['error']}"
        )

    # If a disease was diagnosed, persist the scan record for Weather Advisory integration
    if result.get("status") == "diseased":
        scan_id = str(uuid.uuid4())
        scan_record = {
            "scan_id": scan_id,
            "location": (location or "Unknown").strip(),
            "crop": (result.get("crop") or "").lower(),
            "disease": result.get("disease", ""),
            "raw_class": result.get("raw_class", ""),
            "status": "diseased",
            "severity": result.get("severity", "Medium"),
            "confidence": result.get("confidence", 0),
            "treatment": result.get("treatment", ""),
            "prevention": result.get("prevention", ""),
            "created_at": datetime.datetime.utcnow().isoformat(),
            "active": True
        }
        await save_scan_record(scan_record)
        result["scan_id"] = scan_id
        result["persisted"] = True

    return result


@router.get("/active-scans/{location}")
async def get_active_scans(location: str):
    """Retrieve active disease scans for a specific location."""
    scans = await get_active_scans_for_location(location)
    return {"success": True, "count": len(scans), "scans": scans}


@router.get("/all-active-scans")
async def get_all_active_scans():
    """Retrieve all active disease scans across all locations."""
    scans = await get_active_scans_for_location("all")
    return {"success": True, "count": len(scans), "scans": scans}


@router.post("/scan/{scan_id}/resolve")
async def resolve_disease_scan(scan_id: str):
    """Mark an active disease scan as resolved/treated."""
    updated = False
    try:
        res = await disease_scans_collection.update_one(
            {"$or": [{"_id": scan_id}, {"scan_id": scan_id}]},
            {"$set": {"active": False, "resolved_at": datetime.datetime.utcnow().isoformat()}}
        )
        if res.modified_count > 0:
            updated = True
    except Exception as e:
        print(f"[WARN] Failed to mark scan resolved in MongoDB: {e}")

    for s in _IN_MEMORY_SCANS:
        if s.get("scan_id") == scan_id:
            s["active"] = False
            s["resolved_at"] = datetime.datetime.utcnow().isoformat()
            updated = True

    return {"success": True, "scan_id": scan_id, "active": False, "updated": updated}
