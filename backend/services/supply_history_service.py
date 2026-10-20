"""
Supply History Service
======================
Tracks daily supply levels per region/crop in the ``supply_history``
MongoDB collection.  Provides helpers for recording snapshots,
querying historical trends, computing z-score anomaly metrics, and
cleaning up stale records.

Document schema
---------------
{
    "region":          str   – lowercase region name
    "crop":            str   – lowercase crop name
    "date":            datetime – date only (midnight, no time component)
    "total_supply_kg": float – aggregate supply for the day
    "listing_count":   int   – number of active listings
    "recorded_at":     datetime – full UTC timestamp of last upsert
}
"""

import datetime
from typing import Any

import numpy as np
from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from config.database import db

# ---------------------------------------------------------------------------
# Collection handle
# ---------------------------------------------------------------------------
supply_history_collection = db["supply_history"]


# ---------------------------------------------------------------------------
# Index creation (called once at module load via startup event)
# ---------------------------------------------------------------------------
async def _create_indexes() -> None:
    """Create a compound unique index on (region, crop, date).

    This ensures one document per region+crop+day and makes
    historical range queries efficient.
    """
    await supply_history_collection.create_index(
        [("region", ASCENDING), ("crop", ASCENDING), ("date", ASCENDING)],
        unique=True,
    )


# ---------------------------------------------------------------------------
# 1. record_supply
# ---------------------------------------------------------------------------
async def record_supply(
    region: str,
    crop: str,
    total_supply_kg: float,
    listing_count: int,
) -> dict[str, Any]:
    """Upsert a daily supply snapshot for *region* / *crop*.

    The unique key is ``(region, crop, date)`` where *date* is today at
    midnight UTC.  If a document already exists for today it is updated
    with the latest values; otherwise a new document is inserted.

    Parameters
    ----------
    region:
        Region name (will be lower-cased).
    crop:
        Crop name (will be lower-cased).
    total_supply_kg:
        Total supply in kilograms for the day.
    listing_count:
        Number of active crop listings contributing to the supply.

    Returns
    -------
    dict
        Operation result with ``success``, ``region``, ``crop``,
        ``date``, and ``upserted`` flag.
    """
    try:
        region_key = region.strip().lower()
        crop_key = crop.strip().lower()

        today = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        filter_doc = {
            "region": region_key,
            "crop": crop_key,
            "date": today,
        }
        update_doc = {
            "$set": {
                "total_supply_kg": total_supply_kg,
                "listing_count": listing_count,
                "recorded_at": datetime.datetime.utcnow(),
            },
            "$setOnInsert": {
                "region": region_key,
                "crop": crop_key,
                "date": today,
            },
        }

        result = await supply_history_collection.update_one(
            filter_doc, update_doc, upsert=True,
        )

        return {
            "success": True,
            "region": region_key,
            "crop": crop_key,
            "date": today.isoformat(),
            "upserted": result.upserted_id is not None,
            "message": (
                "New supply record created"
                if result.upserted_id
                else "Existing supply record updated"
            ),
        }
    except PyMongoError as e:
        return {"success": False, "error": f"Database error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 2. get_supply_history
# ---------------------------------------------------------------------------
async def get_supply_history(
    region: str,
    crop: str,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Return the last *days* of supply history for *region* / *crop*.

    Parameters
    ----------
    region:
        Region name (will be lower-cased).
    crop:
        Crop name (will be lower-cased).
    days:
        Number of days of history to retrieve (default 30).

    Returns
    -------
    list[dict]
        List of ``{date, total_supply_kg, listing_count}`` dicts sorted
        by date ascending.
    """
    try:
        region_key = region.strip().lower()
        crop_key = crop.strip().lower()

        cutoff = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) - datetime.timedelta(days=days)

        cursor = supply_history_collection.find(
            {
                "region": region_key,
                "crop": crop_key,
                "date": {"$gte": cutoff},
            },
            {
                "_id": 0,
                "date": 1,
                "total_supply_kg": 1,
                "listing_count": 1,
            },
        ).sort("date", ASCENDING)

        results: list[dict[str, Any]] = []
        async for doc in cursor:
            results.append({
                "date": doc["date"].isoformat(),
                "total_supply_kg": doc["total_supply_kg"],
                "listing_count": doc["listing_count"],
            })

        return results
    except PyMongoError:
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 3. compute_zscore  (synchronous – pure math)
# ---------------------------------------------------------------------------
def compute_zscore(
    supply_history: list[float],
    current_supply: float,
) -> dict[str, Any]:
    """Compute z-score anomaly metrics for *current_supply* against
    a list of historical supply values.

    Parameters
    ----------
    supply_history:
        Historical supply values (e.g. ``total_supply_kg`` for each day).
    current_supply:
        Today's supply value to evaluate.

    Returns
    -------
    dict
        Keys: ``mean_30d``, ``std_30d``, ``z_score``, ``anomaly_level``,
        ``interpretation``, ``data_points``, ``sufficient_data``.

    Notes
    -----
    Anomaly levels based on absolute z-score:

    * ``< 1.0`` → **normal** – "Supply within normal variation"
    * ``1.0 – 2.0`` → **elevated** – "Supply above average, worth monitoring"
    * ``2.0 – 3.0`` → **significant** – "Supply significantly above normal — HIGH anomaly"
    * ``> 3.0`` → **critical** – "Extreme supply spike — CRITICAL anomaly"

    If fewer than 7 data points are available the function returns
    ``sufficient_data=False`` with ``z_score=0`` and
    ``anomaly_level="insufficient_data"``.
    """
    data_points = len(supply_history)

    # ---- Insufficient data guard ----
    if data_points < 7:
        return {
            "mean_30d": 0.0,
            "std_30d": 0.0,
            "z_score": 0.0,
            "anomaly_level": "insufficient_data",
            "interpretation": (
                f"Only {data_points} data point(s) available; "
                "need at least 7 for meaningful analysis"
            ),
            "data_points": data_points,
            "sufficient_data": False,
        }

    # ---- Compute statistics ----
    arr = np.array(supply_history, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))  # population std

    if std_val == 0:
        z_score = 0.0
    else:
        z_score = float((current_supply - mean_val) / std_val)

    # ---- Classify anomaly ----
    abs_z = abs(z_score)
    if abs_z >= 3.0:
        anomaly_level = "critical"
        interpretation = "Extreme supply spike — CRITICAL anomaly"
    elif abs_z >= 2.0:
        anomaly_level = "significant"
        interpretation = "Supply significantly above normal — HIGH anomaly"
    elif abs_z >= 1.0:
        anomaly_level = "elevated"
        interpretation = "Supply above average, worth monitoring"
    else:
        anomaly_level = "normal"
        interpretation = "Supply within normal variation"

    return {
        "mean_30d": round(mean_val, 2),
        "std_30d": round(std_val, 2),
        "z_score": round(z_score, 2),
        "anomaly_level": anomaly_level,
        "interpretation": interpretation,
        "data_points": data_points,
        "sufficient_data": True,
    }


# ---------------------------------------------------------------------------
# 4. cleanup_old_records
# ---------------------------------------------------------------------------
async def cleanup_old_records(days_to_keep: int = 90) -> int:
    """Delete supply-history records older than *days_to_keep* days.

    Parameters
    ----------
    days_to_keep:
        Number of days of history to retain (default 90).

    Returns
    -------
    int
        Number of documents deleted.
    """
    try:
        cutoff = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) - datetime.timedelta(days=days_to_keep)

        result = await supply_history_collection.delete_many(
            {"date": {"$lt": cutoff}},
        )
        return result.deleted_count
    except PyMongoError:
        return 0
    except Exception:
        return 0
