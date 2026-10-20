"""
Shipment Log Service
====================

Manages the `shipment_log` MongoDB collection that tracks crop redistribution
shipments between brokers across different regions.

Each shipment records source/destination regions, crop type, quantity, pricing,
transport cost, and lifecycle status (pending → in_transit → delivered / cancelled).
"""

from config.database import db
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
shipment_log_collection = db["shipment_log"]

# ---------------------------------------------------------------------------
# Transport rate configuration
# ---------------------------------------------------------------------------
TRANSPORT_RATES: dict = {
    "small":  {"label": "Small (< 3 tonnes)",  "rate_per_km": 25.0},
    "medium": {"label": "Medium (3-10 tonnes)", "rate_per_km": 35.0},
    "large":  {"label": "Large (> 10 tonnes)",  "rate_per_km": 50.0},
}
DEFAULT_TRUCK_TYPE: str = "medium"

VALID_STATUSES: set = {"pending", "in_transit", "delivered", "cancelled"}


# ---------------------------------------------------------------------------
# 1. Transport cost calculator (sync – pure math)
# ---------------------------------------------------------------------------
def calculate_transport_cost(
    distance_km: float,
    quantity_kg: float,
    truck_type: str = DEFAULT_TRUCK_TYPE,
) -> dict:
    """Calculate the transport cost for a shipment.

    Args:
        distance_km: Distance between source and destination in kilometres.
        quantity_kg: Total cargo weight in kilograms.
        truck_type: One of ``"small"``, ``"medium"``, or ``"large"``.

    Returns:
        dict with cost breakdown including total cost and cost per 100 kg.

    Raises:
        ValueError: If *truck_type* is not recognised or *quantity_kg* <= 0.
    """
    if truck_type not in TRANSPORT_RATES:
        raise ValueError(
            f"Invalid truck_type '{truck_type}'. "
            f"Choose from: {list(TRANSPORT_RATES.keys())}"
        )
    if quantity_kg <= 0:
        raise ValueError("quantity_kg must be greater than 0")

    rate_info = TRANSPORT_RATES[truck_type]
    rate_per_km: float = rate_info["rate_per_km"]
    total_cost: float = round(rate_per_km * distance_km, 2)
    cost_per_100kg: float = round((total_cost / quantity_kg) * 100, 2)

    return {
        "distance_km": distance_km,
        "truck_type": truck_type,
        "truck_label": rate_info["label"],
        "rate_per_km": rate_per_km,
        "total_cost": total_cost,
        "cost_per_100kg": cost_per_100kg,
        "quantity_kg": quantity_kg,
    }


# ---------------------------------------------------------------------------
# 2. Create a new shipment
# ---------------------------------------------------------------------------
async def create_shipment(
    broker_id: str,
    source_region: str,
    destination_region: str,
    crop: str,
    quantity_kg: float,
    agreed_price: float,
    transport_cost: float,
    expected_delivery_days: int = 2,
    notes: str = "",
) -> dict:
    """Insert a new shipment record into the ``shipment_log`` collection.

    The shipment is created with ``status="pending"`` and timestamps for
    ``initiated_at`` (now) and ``expected_delivery`` (now + *expected_delivery_days*).

    Args:
        broker_id: ObjectId (as string) of the initiating broker.
        source_region: Origin region name (stored lowercase).
        destination_region: Target region name (stored lowercase).
        crop: Crop name (stored lowercase).
        quantity_kg: Weight of the shipment in kilograms.
        agreed_price: Agreed price per kilogram.
        transport_cost: Total transport cost for the shipment.
        expected_delivery_days: Estimated days until delivery (default 2).
        notes: Optional free-text notes.

    Returns:
        The created document as a dict with ``_id`` converted to string.
    """
    now = datetime.utcnow()

    shipment_doc: dict = {
        "broker_id": broker_id,
        "source_region": source_region.strip().lower(),
        "destination_region": destination_region.strip().lower(),
        "crop": crop.strip().lower(),
        "quantity_kg": float(quantity_kg),
        "status": "pending",
        "initiated_at": now,
        "expected_delivery": now + timedelta(days=expected_delivery_days),
        "actual_delivery": None,
        "agreed_price": float(agreed_price),
        "transport_cost": float(transport_cost),
        "notes": notes.strip(),
    }

    result = await shipment_log_collection.insert_one(shipment_doc)
    shipment_doc["_id"] = str(result.inserted_id)
    return shipment_doc


# ---------------------------------------------------------------------------
# 3. Update shipment status
# ---------------------------------------------------------------------------
async def update_shipment_status(shipment_id: str, new_status: str) -> dict:
    """Transition a shipment to a new lifecycle status.

    When *new_status* is ``"delivered"``, ``actual_delivery`` is automatically
    set to the current UTC time.

    Args:
        shipment_id: The ``_id`` of the shipment document (as string).
        new_status: Target status – one of
            ``"pending"``, ``"in_transit"``, ``"delivered"``, ``"cancelled"``.

    Returns:
        The updated document as a dict with ``_id`` as string.

    Raises:
        ValueError: If *new_status* is not a recognised status value.
        Exception: If the shipment is not found.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. "
            f"Must be one of: {VALID_STATUSES}"
        )

    update_fields: dict = {"status": new_status}
    if new_status == "delivered":
        update_fields["actual_delivery"] = datetime.utcnow()

    updated = await shipment_log_collection.find_one_and_update(
        {"_id": ObjectId(shipment_id)},
        {"$set": update_fields},
        return_document=True,
    )

    if not updated:
        raise Exception(f"Shipment not found: {shipment_id}")

    updated["_id"] = str(updated["_id"])
    return updated


# ---------------------------------------------------------------------------
# 4. Get shipments for a broker
# ---------------------------------------------------------------------------
async def get_broker_shipments(
    broker_id: str,
    status: Optional[str] = None,
    limit: int = 20,
) -> list:
    """Retrieve shipments initiated by a specific broker.

    Args:
        broker_id: The broker's ObjectId as a string.
        status: Optional filter – return only shipments with this status.
        limit: Maximum number of documents to return (default 20).

    Returns:
        A list of shipment dicts sorted by ``initiated_at`` descending,
        with ``_id`` converted to string.
    """
    query: dict = {"broker_id": broker_id}
    if status:
        query["status"] = status

    cursor = shipment_log_collection.find(query).sort(
        "initiated_at", -1
    ).limit(limit)

    shipments: list = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        shipments.append(doc)

    return shipments


# ---------------------------------------------------------------------------
# 5. Get incoming shipments to a destination
# ---------------------------------------------------------------------------
async def get_incoming_to_destination(
    destination_region: str,
    crop: str,
    hours: int = 24,
) -> dict:
    """Summarise pending/in-transit shipments heading to a destination region.

    Args:
        destination_region: Target region name (case-insensitive).
        crop: Crop name to filter on (case-insensitive).
        hours: Look-back window in hours (default 24).

    Returns:
        A summary dict containing total pending incoming weight, shipment
        count, and individual shipment summaries.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    query: dict = {
        "destination_region": destination_region.strip().lower(),
        "crop": crop.strip().lower(),
        "status": {"$in": ["pending", "in_transit"]},
        "initiated_at": {"$gt": cutoff},
    }

    shipments: list = []
    total_kg: float = 0.0

    async for doc in shipment_log_collection.find(query):
        doc["_id"] = str(doc["_id"])
        total_kg += doc.get("quantity_kg", 0.0)
        shipments.append(doc)

    return {
        "destination": destination_region.strip().lower(),
        "crop": crop.strip().lower(),
        "pending_incoming_kg": round(total_kg, 2),
        "shipment_count": len(shipments),
        "shipments": shipments,
    }


# ---------------------------------------------------------------------------
# 6. Expire stale shipments
# ---------------------------------------------------------------------------
async def expire_stale_shipments(days: int = 7) -> int:
    """Cancel shipments that have been pending for longer than *days*.

    Finds all documents where ``status == "pending"`` and ``initiated_at``
    is older than ``now - days``, then sets their status to ``"cancelled"``.

    Args:
        days: Age threshold in days (default 7).

    Returns:
        The number of shipments that were cancelled.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    result = await shipment_log_collection.update_many(
        {
            "status": "pending",
            "initiated_at": {"$lt": cutoff},
        },
        {"$set": {"status": "cancelled"}},
    )

    return result.modified_count
