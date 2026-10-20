"""Audit dynamic thresholds across regions and crops."""
import sys
sys.path.insert(0, ".")
from services.demand_ml_service import predict_dynamic_threshold
from services.region_service import get_region_info

regions = ["Karimnagar", "Hyderabad", "Warangal", "Mumbai", "Chennai", "Vijayawada"]
crops = ["rice", "wheat", "tomato", "cotton"]

print(f"{'Region':15s} {'Crop':8s} {'Threshold':>12s}  {'Type':12s} {'Pop':>6s} {'Size':8s}")
print("-" * 75)

for r in regions:
    info = get_region_info(r)
    if not info:
        print(f"{r:15s} NOT FOUND")
        continue
    for c in crops:
        t = predict_dynamic_threshold(
            crop=c,
            region_type=info.get("region_type", "semi-urban"),
            population_density=info.get("population_density", 3000),
            market_size=info.get("market_size", "medium")
        )
        threshold = t["predicted_threshold_kg"]
        rtype = info.get("region_type", "?")
        pop = info.get("population_density", 0)
        size = info.get("market_size", "?")
        print(f"{r:15s} {c:8s} {threshold:>10,.0f} kg  {rtype:12s} {pop:>6} {size:8s}")
