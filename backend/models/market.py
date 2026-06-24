from pydantic import BaseModel
from typing import Optional
from enum import Enum

class CropStatus(str, Enum):
    available = "available"
    sold = "sold"
    reserved = "reserved"

class CropListing(BaseModel):
    farmer_name: str
    location: str
    crop_name: str
    quantity_kg: float
    price_per_kg: float
    status: CropStatus = CropStatus.available
    contact: Optional[str] = None

class DemandQuery(BaseModel):
    region: str
    crop_name: Optional[str] = None