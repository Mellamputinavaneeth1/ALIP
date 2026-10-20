from fastapi import APIRouter, HTTPException, status
from models.user import UserRegister, UserLogin, UserResponse, ProfileUpdate
from config.database import users_collection
from config.security import hash_password, verify_password, create_access_token
from services import local_auth_store
from bson import ObjectId
from pymongo.errors import PyMongoError
import datetime

router = APIRouter(prefix="/auth", tags=["Authentication"])


def database_unavailable_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Database connection unavailable. Check MongoDB Atlas network "
            "access, cluster status, and MONGODB_URL."
        ),
    )


def build_user_doc(user: UserRegister) -> dict:
    doc = {
        "full_name": user.full_name,
        "email": user.email,
        "password": hash_password(user.password),
        "role": user.role.value,
        "phone": user.phone,
        "location": user.location,
        "created_at": datetime.datetime.utcnow()
    }
    # Add storage capacity for broker profiles
    if user.storage_capacity:
        doc["storage_capacity"] = {
            "cold_storage_kg": user.storage_capacity.cold_storage_kg,
            "dry_storage_kg": user.storage_capacity.dry_storage_kg,
            "open_yard_kg": user.storage_capacity.open_yard_kg,
            "last_updated": datetime.datetime.utcnow()
        }
    return doc


def build_login_response(db_user: dict) -> dict:
    token = create_access_token(data={
        "sub": str(db_user["_id"]),
        "email": db_user["email"],
        "role": db_user["role"]
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(db_user["_id"]),
        "role": db_user["role"],
        "full_name": db_user["full_name"],
        "email": db_user["email"],
        "phone": db_user.get("phone", ""),
        "location": db_user.get("location", "")
    }


async def register_with_local_store(user: UserRegister) -> dict:
    existing_user = await local_auth_store.find_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    user_doc = build_user_doc(user)
    inserted_id = await local_auth_store.insert_user(user_doc)

    return {
        "message": "User registered successfully",
        "id": inserted_id,
        "role": user.role.value,
        "storage": "local"
    }


async def login_with_local_store(user: UserLogin) -> dict:
    db_user = await local_auth_store.find_user_by_email(user.email)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    response = build_login_response(db_user)
    response["storage"] = "local"
    return response

@router.post("/register")
async def register(user: UserRegister):
    try:
        # Check if email already exists
        existing_user = await users_collection.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user document
        user_doc = build_user_doc(user)
        
        # Insert into database
        result = await users_collection.insert_one(user_doc)
        
        return {
            "message": "User registered successfully",
            "id": str(result.inserted_id),
            "role": user.role.value
        }
    except HTTPException:
        raise
    except PyMongoError:
        return await register_with_local_store(user)

@router.post("/login")
async def login(user: UserLogin):
    try:
        # Find user by email
        db_user = await users_collection.find_one({"email": user.email})
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(user.password, db_user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        return build_login_response(db_user)
    except HTTPException:
        raise
    except PyMongoError:
        return await login_with_local_store(user)


@router.patch("/profile/{user_id}")
async def update_profile(user_id: str, update: ProfileUpdate):
    """Update user profile (storage capacity, name, phone, location)."""
    try:
        update_fields = {}
        if update.full_name is not None:
            update_fields["full_name"] = update.full_name
        if update.phone is not None:
            update_fields["phone"] = update.phone
        if update.location is not None:
            update_fields["location"] = update.location
        if update.storage_capacity is not None:
            update_fields["storage_capacity"] = {
                "cold_storage_kg": update.storage_capacity.cold_storage_kg,
                "dry_storage_kg": update.storage_capacity.dry_storage_kg,
                "open_yard_kg": update.storage_capacity.open_yard_kg,
                "last_updated": datetime.datetime.utcnow()
            }

        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        result = await users_collection.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": update_fields},
            return_document=True
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return {
            "message": "Profile updated successfully",
            "user_id": str(result["_id"]),
            "updated_fields": list(update_fields.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile including storage capacity."""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {
            "id": str(user["_id"]),
            "full_name": user.get("full_name"),
            "email": user.get("email"),
            "role": user.get("role"),
            "phone": user.get("phone"),
            "location": user.get("location"),
            "storage_capacity": user.get("storage_capacity"),
            "created_at": str(user.get("created_at", ""))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
