import datetime

from config.database import market_alerts_collection, market_collection
from models.market import CropListing
from pymongo.errors import PyMongoError
from services.demand_analysis import (
    DEMAND_THRESHOLDS,
    add_supply,
    build_market_signal,
    empty_supply_index,
    normalize_key,
)
from services.demand_ml_service import (
    predict_dynamic_threshold,
    analyze_supply_trend,
    estimate_price_impact,
)
from services.region_service import (
    get_nearest_regions,
    get_region_info,
)
from services import local_market_store


def _listing_to_doc(listing: CropListing) -> dict:
    return {
        "farmer_name": listing.farmer_name,
        "location": normalize_key(listing.location),
        "crop_name": normalize_key(listing.crop_name),
        "quantity_kg": listing.quantity_kg,
        "price_per_kg": listing.price_per_kg,
        "status": listing.status.value,
        "contact": listing.contact,
        "created_at": datetime.datetime.utcnow(),
    }


def _build_supply_summary_from_listings(listings: list[dict]) -> dict:
    supply = {}
    for doc in listings:
        crop = doc["crop_name"]
        qty = doc["quantity_kg"]
        if crop not in supply:
            supply[crop] = {
                "total_quantity_kg": 0,
                "listings_count": 0,
                "avg_price_per_kg": 0,
                "prices": [],
            }
        supply[crop]["total_quantity_kg"] += qty
        supply[crop]["listings_count"] += 1
        supply[crop]["prices"].append(doc["price_per_kg"])

    for crop in supply:
        prices = supply[crop]["prices"]
        supply[crop]["avg_price_per_kg"] = round(sum(prices) / len(prices), 2)
        del supply[crop]["prices"]

    return supply


def _build_regional_data_from_listings(listings: list[dict]) -> dict:
    regional_data = empty_supply_index()
    for doc in listings:
        add_supply(
            regional_data,
            doc.get("location"),
            doc.get("crop_name"),
            doc.get("quantity_kg", 0),
        )
    return regional_data


async def _load_local_regional_supply_index(query: dict = None) -> dict:
    listings = await local_market_store.find_listings(query or {"status": "available"})
    return _build_regional_data_from_listings(listings)


async def _load_regional_supply_index(query: dict = None) -> dict:
    cursor = market_collection.find(query or {"status": "available"})
    regional_data = empty_supply_index()

    async for doc in cursor:
        add_supply(
            regional_data,
            doc.get("location"),
            doc.get("crop_name"),
            doc.get("quantity_kg", 0),
        )

    return regional_data


async def _save_alert_history(signals: list) -> None:
    now = datetime.datetime.utcnow()
    for signal in signals:
        await market_alerts_collection.update_one(
            {"id": signal["id"]},
            {
                "$set": {
                    **signal,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )


async def add_crop_listing(listing: CropListing) -> dict:
    doc = _listing_to_doc(listing)
    try:
        result = await market_collection.insert_one(doc)
        return {
            "success": True,
            "message": "Crop listing added successfully",
            "id": str(result.inserted_id),
        }
    except PyMongoError:
        inserted_id = await local_market_store.insert_listing(doc)
        return {
            "success": True,
            "message": "Crop listing added successfully",
            "id": inserted_id,
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_all_listings() -> dict:
    try:
        cursor = market_collection.find({"status": "available"})
        listings = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["created_at"] = str(doc["created_at"])
            listings.append(doc)
        return {
            "success": True,
            "total": len(listings),
            "listings": listings,
        }
    except PyMongoError:
        listings = await local_market_store.find_listings({"status": "available"})
        return {
            "success": True,
            "total": len(listings),
            "listings": listings,
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_regional_supply(region: str) -> dict:
    try:
        region_key = normalize_key(region)
        cursor = market_collection.find({
            "location": region_key,
            "status": "available",
        })

        listings = []
        async for doc in cursor:
            listings.append(doc)
        supply = _build_supply_summary_from_listings(listings)

        return {
            "success": True,
            "region": region_key,
            "crops_available": len(supply),
            "supply": supply,
        }
    except PyMongoError:
        listings = await local_market_store.find_listings({
            "location": region_key,
            "status": "available",
        })
        supply = _build_supply_summary_from_listings(listings)
        return {
            "success": True,
            "region": region_key,
            "crops_available": len(supply),
            "supply": supply,
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_demand(region: str, crop_name: str = None) -> dict:
    try:
        region_key = normalize_key(region)
        regional_data = await _load_regional_supply_index()
        supply_summary = dict(regional_data.get(region_key, {}))

        if crop_name:
            crop_key = normalize_key(crop_name)
            supply_summary = {
                crop_key: supply_summary.get(crop_key, 0)
            }

        alerts = []
        recommendations = []
        signals = []

        for crop, total_qty in supply_summary.items():
            signal = build_market_signal(
                region=region_key,
                crop=crop,
                quantity=total_qty,
                regional_data=regional_data,
            )
            signals.append(signal)

            if signal["type"] in ["oversupply", "undersupply"]:
                alerts.append(signal)
            else:
                recommendations.append({
                    "crop": crop,
                    "status": "balanced",
                    "threshold": signal["threshold"],
                    "supply_ratio": signal["supply_ratio"],
                    "message": signal["message"],
                })

        await _save_alert_history(signals)

        return {
            "success": True,
            "region": region_key,
            "analysis": supply_summary,
            "alerts": alerts,
            "recommendations": recommendations,
            "thresholds": DEMAND_THRESHOLDS,
            "total_alerts": len(alerts),
        }
    except PyMongoError:
        regional_data = await _load_local_regional_supply_index()
        supply_summary = dict(regional_data.get(region_key, {}))

        if crop_name:
            crop_key = normalize_key(crop_name)
            supply_summary = {
                crop_key: supply_summary.get(crop_key, 0)
            }

        alerts = []
        recommendations = []

        for crop, total_qty in supply_summary.items():
            signal = build_market_signal(
                region=region_key,
                crop=crop,
                quantity=total_qty,
                regional_data=regional_data,
            )

            if signal["type"] in ["oversupply", "undersupply"]:
                alerts.append(signal)
            else:
                recommendations.append({
                    "crop": crop,
                    "status": "balanced",
                    "threshold": signal["threshold"],
                    "supply_ratio": signal["supply_ratio"],
                    "message": signal["message"],
                })

        return {
            "success": True,
            "region": region_key,
            "analysis": supply_summary,
            "alerts": alerts,
            "recommendations": recommendations,
            "thresholds": DEMAND_THRESHOLDS,
            "total_alerts": len(alerts),
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_market_summary() -> dict:
    try:
        cursor = market_collection.find({"status": "available"})

        regional_data = {}
        total_listings = 0

        async for doc in cursor:
            region = doc["location"]
            crop = doc["crop_name"]
            qty = doc["quantity_kg"]
            total_listings += 1

            if region not in regional_data:
                regional_data[region] = {}
            if crop not in regional_data[region]:
                regional_data[region][crop] = 0
            regional_data[region][crop] += qty

        return {
            "success": True,
            "total_listings": total_listings,
            "regions_active": len(regional_data),
            "regional_summary": regional_data,
            "thresholds": DEMAND_THRESHOLDS,
        }
    except PyMongoError:
        listings = await local_market_store.find_listings({"status": "available"})
        regional_data = _build_regional_data_from_listings(listings)
        return {
            "success": True,
            "total_listings": len(listings),
            "regions_active": len(regional_data),
            "regional_summary": regional_data,
            "thresholds": DEMAND_THRESHOLDS,
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_all_market_alerts() -> dict:
    try:
        regional_data = await _load_regional_supply_index()
        signals = []

        for region, crops in regional_data.items():
            for crop, total_qty in crops.items():
                signals.append(
                    build_market_signal(
                        region=region,
                        crop=crop,
                        quantity=total_qty,
                        regional_data=regional_data,
                    )
                )

        await _save_alert_history(signals)

        return {
            "success": True,
            "total": len(signals),
            "high_alerts": len([
                signal for signal in signals
                if signal["severity"] == "high"
            ]),
            "medium_alerts": len([
                signal for signal in signals
                if signal["severity"] == "medium"
            ]),
            "alerts": signals,
        }
    except PyMongoError:
        regional_data = await _load_local_regional_supply_index()
        signals = []

        for region, crops in regional_data.items():
            for crop, total_qty in crops.items():
                signals.append(
                    build_market_signal(
                        region=region,
                        crop=crop,
                        quantity=total_qty,
                        regional_data=regional_data,
                    )
                )

        return {
            "success": True,
            "total": len(signals),
            "high_alerts": len([
                signal for signal in signals
                if signal["severity"] == "high"
            ]),
            "medium_alerts": len([
                signal for signal in signals
                if signal["severity"] == "medium"
            ]),
            "alerts": signals,
            "storage": "local",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_enhanced_demand_analysis(
    region: str,
    crop: str,
    current_supply: float,
    supply_history: list = None,
) -> dict:
    """
    Enhanced broker market analysis combining
    all 4 intelligence modules.
    """
    try:
        info = get_region_info(region) or {}
        month = datetime.datetime.now().month

        # 1. Dynamic ML threshold
        threshold_data = predict_dynamic_threshold(
            crop=crop,
            region_type=info.get("region_type", "semi-urban"),
            population_density=info.get("population_density", 3000),
            market_size=info.get("market_size", "medium"),
            month=month,
        )
        threshold = threshold_data["predicted_threshold_kg"]

        # 2. Supply status
        if current_supply > 1.5 * threshold:
            status = "Oversupply"
        elif current_supply < 0.5 * threshold:
            status = "Undersupply"
        else:
            status = "Balanced"

        # 3. Nearby regions
        nearby = get_nearest_regions(region, crop, n=3)

        # 4. Trend analysis
        trend = None
        if supply_history and len(supply_history) >= 2:
            trend = analyze_supply_trend(supply_history, threshold)

        # 5. Price impact
        price = None
        if status == "Oversupply":
            price = estimate_price_impact(crop, current_supply, threshold)

        return {
            "success": True,
            "region": region,
            "crop": crop,
            "current_supply_kg": current_supply,
            "supply_status": status,
            "dynamic_threshold": threshold_data,
            "nearest_regions": nearby,
            "trend_analysis": trend,
            "price_impact": price,
            "analysis_engine": "ALIP Enhanced Broker Intelligence v2.0",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
