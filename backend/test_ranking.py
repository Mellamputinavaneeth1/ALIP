"""Test both undersupply and oversupply scenarios."""
import asyncio, sys
sys.path.insert(0, ".")
from services.market_ranking_service import rank_destinations

async def test():
    print("=== TEST 1: Undersupply (70kg rice in Karimnagar) ===")
    r1 = await rank_destinations("Karimnagar", "rice", 70, top_n=3)
    print(f"Success: {r1['success']}")
    print(f"Oversupply ratio: {r1['current_state']['oversupply_ratio']}")
    print(f"Urgency: {r1['urgency']['score']}/10 — {r1['urgency']['label']}")
    print(f"Message: {r1.get('message', 'N/A')}")
    print(f"Destinations: {len(r1.get('destinations', []))}")
    print()

    print("=== TEST 2: Oversupply (3333kg wheat in Hyderabad) ===")
    r2 = await rank_destinations("Hyderabad", "wheat", 3333, truck_type="large", top_n=3)
    print(f"Success: {r2['success']}")
    print(f"Oversupply ratio: {r2['current_state']['oversupply_ratio']}")
    print(f"Urgency: {r2['urgency']['score']}/10 — {r2['urgency']['label']}")
    print(f"Destinations: {len(r2.get('destinations', []))}")
    for d in r2.get("destinations", [])[:3]:
        print(f"  - {d['region']} score={d['composite_score']} dist={d['distance_km']:.0f}km")
    print()

    print("=== TEST 3: Perishable oversupply (5000kg tomato) ===")
    r3 = await rank_destinations("Hyderabad", "tomato", 5000, top_n=3)
    print(f"Success: {r3['success']}")
    print(f"Oversupply ratio: {r3['current_state']['oversupply_ratio']}")
    print(f"Urgency: {r3['urgency']['score']}/10 — {r3['urgency']['label']}")
    print(f"Destinations: {len(r3.get('destinations', []))}")
    print(f"Price day0: {r3['price_trajectory']['day_prices'][0]}")
    print(f"Price day6: {r3['price_trajectory']['day_prices'][6]}")

asyncio.run(test())
