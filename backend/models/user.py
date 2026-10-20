from pydantic import BaseModel
from typing import Optional
from enum import Enum
from datetime import datetime

class UserRole(str, Enum):
    farmer = "farmer"
    broker = "broker"

# Storage capacity model for broker profiles
class StorageCapacity(BaseModel):
    cold_storage_kg: float = 0.0
    dry_storage_kg: float = 0.0
    open_yard_kg: float = 0.0
    last_updated: Optional[datetime] = None

# Register request model
class UserRegister(BaseModel):
    full_name: str
    email: str
    password: str
    role: UserRole
    phone: Optional[str] = None
    location: Optional[str] = None
    storage_capacity: Optional[StorageCapacity] = None

# Login request model
class UserLogin(BaseModel):
    email: str
    password: str

# Response model (never expose password)
class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    role: str
    phone: Optional[str] = None
    location: Optional[str] = None
    storage_capacity: Optional[StorageCapacity] = None

# Profile update model (partial updates)
class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    storage_capacity: Optional[StorageCapacity] = None