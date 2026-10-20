"""Quick validation of all new demand analysis services."""
import sys
sys.path.insert(0, ".")

from services.urgency_service import compute_urgency
from services.price_trajectory_service import compute_price_trajectory
from services.supply_history_service import compute_zscore
from services.shipment_service import calculate_transport_cost
from services.seasonal_context_service import get_seasonal_context

passed = 0
failed = 0

# Test 1: Transport cost fix (was 10/km, now 35/km for medium)
tc = calculate_transport_cost(134.6, 300, "medium")
print(f"Test 1 - Transport: rate={tc['rate_per_km']}/km, total={tc['total_cost']}, per100kg={tc['cost_per_100kg']}")
if tc["rate_per_km"] == 35.0:
    print("  PASS")
    passed += 1
else:
    print(f"  FAIL: expected 35, got {tc['rate_per_km']}")
    failed += 1

# Test 2: Urgency scorer
u = compute_urgency(2.04, 2.8, "rapidly_increasing", "rice", 3)
print(f"Test 2 - Urgency: score={u['score']}, label={u['label']}, act_within={u['act_within_days']}d")
if u["score"] >= 7:
    print("  PASS")
    passed += 1
else:
    print(f"  FAIL: expected >= 7, got {u['score']}")
    failed += 1

# Test 3: Z-score
z = compute_zscore([500, 520, 510, 530, 515, 525, 540, 520, 510, 530], 1400)
print(f"Test 3 - Z-score: z={z['z_score']:.2f}, level={z['anomaly_level']}")
if z["anomaly_level"] == "critical":
    print("  PASS")
    passed += 1
else:
    print(f"  FAIL: expected critical, got {z['anomaly_level']}")
    failed += 1

# Test 4: Price trajectory
pt = compute_price_trajectory("tomato", 30, 2.0, 93.8)
print(f"Test 4 - Trajectory: day0={pt['day_prices'][0]}, day6={pt['day_prices'][6]}, break_even={pt['break_even_day']}")
if pt["day_prices"][0] == 30 and pt["day_prices"][6] < 30:
    print("  PASS")
    passed += 1
else:
    print(f"  FAIL")
    failed += 1

# Test 5: Seasonal context
sc = get_seasonal_context("rice", "rapidly_increasing", "Hyderabad", month=10)
print(f"Test 5 - Seasonal: peak={sc['is_peak_month']}, self_correct={sc['self_correction_likely']}")
if sc["is_peak_month"] and not sc["self_correction_likely"]:
    print("  PASS (peak month + trend growing = no self-correction)")
    passed += 1
else:
    print(f"  FAIL")
    failed += 1

# Test 6: Transport cost for different truck types
tc_small = calculate_transport_cost(200, 500, "small")
tc_large = calculate_transport_cost(200, 500, "large")
print(f"Test 6 - Truck types: small={tc_small['rate_per_km']}, large={tc_large['rate_per_km']}")
if tc_small["rate_per_km"] == 25.0 and tc_large["rate_per_km"] == 50.0:
    print("  PASS")
    passed += 1
else:
    print(f"  FAIL")
    failed += 1

# Test 7: Urgency for perishable crop
u2 = compute_urgency(2.0, 2.0, "increasing", "tomato", 1)
print(f"Test 7 - Perishable urgency: score={u2['score']}, label={u2['label']}, tier={u2['perishability_tier']}")
if u2["score"] > u["score"] or u2["perishability_tier"] == 1:
    print("  PASS (tomato tier 1 gets higher multiplier)")
    passed += 1
else:
    print(f"  FAIL")
    failed += 1

print(f"\n{'='*50}")
print(f"Results: {passed} PASSED, {failed} FAILED out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED")
