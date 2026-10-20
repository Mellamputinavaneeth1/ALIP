import json
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock


STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "local_users.json"
_lock = Lock()


def _load_users() -> list[dict]:
    if not STORE_PATH.exists():
        return []

    with STORE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_users(users: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


async def find_user_by_email(email: str) -> dict | None:
    normalized_email = email.lower()
    with _lock:
        for user in _load_users():
            if user.get("email", "").lower() == normalized_email:
                return user

    return None


async def insert_user(user_doc: dict) -> str:
    with _lock:
        users = _load_users()
        user_id = str(uuid.uuid4())
        local_user = {
            **_json_safe(user_doc),
            "_id": user_id,
            "role": str(user_doc["role"]),
        }
        users.append(local_user)
        _save_users(users)
        return user_id
