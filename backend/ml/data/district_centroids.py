"""
ALIP — District Centroids
60 key Indian agricultural districts with lat/lon centroids and LGD metadata.
Covers all major crop zones: Gangetic plains, Deccan, Coastal, Arid, Hill.
Expand to full 680 districts by placing Kaggle DES data in ml/data/raw/kaggle_des/.
"""

# district_id → metadata
# lat/lon = district centroid (used for climate and soil API calls)
DISTRICT_CENTROIDS = {
    # ── Punjab (Subtropical Alluvial — Wheat, Rice) ────────────────────────
    "PB001": {"name": "Ludhiana",    "state": "Punjab",         "lat": 30.900, "lon": 75.857},
    "PB002": {"name": "Amritsar",    "state": "Punjab",         "lat": 31.634, "lon": 74.872},
    "PB003": {"name": "Bathinda",    "state": "Punjab",         "lat": 30.211, "lon": 74.945},
    "PB004": {"name": "Jalandhar",   "state": "Punjab",         "lat": 31.326, "lon": 75.576},
    "PB005": {"name": "Patiala",     "state": "Punjab",         "lat": 30.340, "lon": 76.386},

    # ── Haryana (Subtropical Alluvial — Wheat, Rice, Mustard) ─────────────
    "HR001": {"name": "Hisar",       "state": "Haryana",        "lat": 29.151, "lon": 75.722},
    "HR002": {"name": "Rohtak",      "state": "Haryana",        "lat": 28.895, "lon": 76.607},
    "HR003": {"name": "Karnal",      "state": "Haryana",        "lat": 29.685, "lon": 76.990},

    # ── Uttar Pradesh (Subtropical Alluvial — Wheat, Rice, Sugarcane) ──────
    "UP001": {"name": "Lucknow",     "state": "Uttar Pradesh",  "lat": 26.847, "lon": 80.947},
    "UP002": {"name": "Varanasi",    "state": "Uttar Pradesh",  "lat": 25.317, "lon": 82.973},
    "UP003": {"name": "Kanpur",      "state": "Uttar Pradesh",  "lat": 26.449, "lon": 80.331},
    "UP004": {"name": "Agra",        "state": "Uttar Pradesh",  "lat": 27.177, "lon": 78.008},
    "UP005": {"name": "Aligarh",     "state": "Uttar Pradesh",  "lat": 27.882, "lon": 78.082},
    "UP006": {"name": "Gorakhpur",   "state": "Uttar Pradesh",  "lat": 26.760, "lon": 83.373},
    "UP007": {"name": "Allahabad",   "state": "Uttar Pradesh",  "lat": 25.435, "lon": 81.846},
    "UP008": {"name": "Moradabad",   "state": "Uttar Pradesh",  "lat": 28.839, "lon": 78.776},

    # ── Bihar (Subtropical — Rice, Wheat, Maize) ───────────────────────────
    "BR001": {"name": "Patna",       "state": "Bihar",          "lat": 25.594, "lon": 85.137},
    "BR002": {"name": "Muzaffarpur", "state": "Bihar",          "lat": 26.120, "lon": 85.391},
    "BR003": {"name": "Bhagalpur",   "state": "Bihar",          "lat": 25.244, "lon": 86.983},

    # ── West Bengal (Tropical Humid — Rice, Jute, Mustard) ────────────────
    "WB001": {"name": "Bardhaman",   "state": "West Bengal",    "lat": 23.232, "lon": 87.860},
    "WB002": {"name": "Murshidabad", "state": "West Bengal",    "lat": 24.177, "lon": 88.275},
    "WB003": {"name": "Midnapore",   "state": "West Bengal",    "lat": 22.423, "lon": 87.319},

    # ── Madhya Pradesh (Semi-Arid Black/Red — Soybean, Wheat, Chickpea) ───
    "MP001": {"name": "Indore",      "state": "Madhya Pradesh", "lat": 22.719, "lon": 75.857},
    "MP002": {"name": "Bhopal",      "state": "Madhya Pradesh", "lat": 23.259, "lon": 77.413},
    "MP003": {"name": "Jabalpur",    "state": "Madhya Pradesh", "lat": 23.181, "lon": 79.987},
    "MP004": {"name": "Sagar",       "state": "Madhya Pradesh", "lat": 23.838, "lon": 78.737},
    "MP005": {"name": "Hoshangabad", "state": "Madhya Pradesh", "lat": 22.754, "lon": 77.726},
    "MP006": {"name": "Ujjain",      "state": "Madhya Pradesh", "lat": 23.182, "lon": 75.771},

    # ── Maharashtra (Semi-Arid Black — Cotton, Soybean, Jowar) ────────────
    "MH001": {"name": "Nashik",      "state": "Maharashtra",    "lat": 20.011, "lon": 73.790},
    "MH002": {"name": "Nagpur",      "state": "Maharashtra",    "lat": 21.146, "lon": 79.089},
    "MH003": {"name": "Pune",        "state": "Maharashtra",    "lat": 18.520, "lon": 73.856},
    "MH004": {"name": "Aurangabad",  "state": "Maharashtra",    "lat": 19.877, "lon": 75.343},
    "MH005": {"name": "Amravati",    "state": "Maharashtra",    "lat": 20.932, "lon": 77.751},
    "MH006": {"name": "Latur",       "state": "Maharashtra",    "lat": 18.400, "lon": 76.560},
    "MH007": {"name": "Nanded",      "state": "Maharashtra",    "lat": 19.162, "lon": 77.299},
    "MH008": {"name": "Solapur",     "state": "Maharashtra",    "lat": 17.686, "lon": 75.901},

    # ── Telangana (Semi-Arid Red — Rice, Cotton, Maize, Jowar) ────────────
    "TS001": {"name": "Hyderabad",   "state": "Telangana",      "lat": 17.385, "lon": 78.487},
    "TS002": {"name": "Warangal",    "state": "Telangana",      "lat": 17.977, "lon": 79.594},
    "TS003": {"name": "Karimnagar",  "state": "Telangana",      "lat": 18.438, "lon": 79.129},
    "TS004": {"name": "Nizamabad",   "state": "Telangana",      "lat": 18.672, "lon": 78.094},
    "TS005": {"name": "Khammam",     "state": "Telangana",      "lat": 17.247, "lon": 80.150},

    # ── Andhra Pradesh (Semi-Arid/Coastal — Rice, Cotton, Groundnut) ──────
    "AP001": {"name": "Guntur",      "state": "Andhra Pradesh", "lat": 16.307, "lon": 80.436},
    "AP002": {"name": "Kurnool",     "state": "Andhra Pradesh", "lat": 15.828, "lon": 78.037},
    "AP003": {"name": "Kadapa",      "state": "Andhra Pradesh", "lat": 14.466, "lon": 78.821},
    "AP004": {"name": "Nellore",     "state": "Andhra Pradesh", "lat": 14.442, "lon": 79.987},
    "AP005": {"name": "Krishna",     "state": "Andhra Pradesh", "lat": 16.610, "lon": 81.084},

    # ── Karnataka (Semi-Arid — Jowar, Ragi, Cotton, Maize) ────────────────
    "KA001": {"name": "Dharwad",     "state": "Karnataka",      "lat": 15.459, "lon": 75.008},
    "KA002": {"name": "Tumkur",      "state": "Karnataka",      "lat": 13.341, "lon": 77.102},
    "KA003": {"name": "Raichur",     "state": "Karnataka",      "lat": 16.202, "lon": 77.356},
    "KA004": {"name": "Hassan",      "state": "Karnataka",      "lat": 13.003, "lon": 76.100},

    # ── Rajasthan (Arid — Bajra, Jowar, Mustard, Barley) ──────────────────
    "RJ001": {"name": "Jodhpur",     "state": "Rajasthan",      "lat": 26.295, "lon": 73.017},
    "RJ002": {"name": "Bikaner",     "state": "Rajasthan",      "lat": 28.018, "lon": 73.316},
    "RJ003": {"name": "Jaipur",      "state": "Rajasthan",      "lat": 26.912, "lon": 75.787},
    "RJ004": {"name": "Kota",        "state": "Rajasthan",      "lat": 25.182, "lon": 75.828},
    "RJ005": {"name": "Alwar",       "state": "Rajasthan",      "lat": 27.553, "lon": 76.634},

    # ── Gujarat (Semi-Arid — Cotton, Groundnut, Wheat, Bajra) ─────────────
    "GJ001": {"name": "Ahmedabad",   "state": "Gujarat",        "lat": 23.023, "lon": 72.572},
    "GJ002": {"name": "Rajkot",      "state": "Gujarat",        "lat": 22.303, "lon": 70.802},
    "GJ003": {"name": "Junagadh",    "state": "Gujarat",        "lat": 21.522, "lon": 70.458},

    # ── Odisha (Tropical Humid — Rice, Maize, Groundnut) ──────────────────
    "OD001": {"name": "Cuttack",     "state": "Odisha",         "lat": 20.462, "lon": 85.883},
    "OD002": {"name": "Bhubaneswar", "state": "Odisha",         "lat": 20.296, "lon": 85.824},

    # ── Tamil Nadu (Semi-Arid/Coastal — Rice, Groundnut, Jowar, Ragi) ─────
    "TN001": {"name": "Coimbatore",  "state": "Tamil Nadu",     "lat": 11.017, "lon": 76.955},
    "TN002": {"name": "Salem",       "state": "Tamil Nadu",     "lat": 11.664, "lon": 78.146},
    "TN003": {"name": "Madurai",     "state": "Tamil Nadu",     "lat":  9.925, "lon": 78.119},
}

# district_id → crop set allowed per season (from ICAR crop zone maps)
# crops: rice, wheat, maize, soybean, cotton, groundnut, pigeonpeas,
#        mungbean, chickpea, mustard, lentil, barley, jowar, bajra, ragi

DISTRICT_CROP_CALENDAR = {
    # Punjab / Haryana — strong wheat-rice rotation
    "PB001": {"kharif": ["rice", "maize", "cotton"],            "rabi": ["wheat", "barley", "mustard"]},
    "PB002": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard"]},
    "PB003": {"kharif": ["cotton", "rice"],                     "rabi": ["wheat", "chickpea"]},
    "PB004": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard"]},
    "PB005": {"kharif": ["rice", "cotton", "maize"],            "rabi": ["wheat", "chickpea", "mustard"]},
    "HR001": {"kharif": ["bajra", "cotton", "jowar"],           "rabi": ["wheat", "mustard", "barley"]},
    "HR002": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard"]},
    "HR003": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard", "barley"]},
    # UP — wheat-rice-maize belt
    "UP001": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "chickpea", "mustard"]},
    "UP002": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "chickpea", "lentil"]},
    "UP003": {"kharif": ["rice", "maize", "jowar"],             "rabi": ["wheat", "mustard", "barley"]},
    "UP004": {"kharif": ["jowar", "bajra", "maize"],            "rabi": ["wheat", "mustard", "barley"]},
    "UP005": {"kharif": ["rice", "jowar"],                      "rabi": ["wheat", "mustard", "chickpea"]},
    "UP006": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "chickpea", "mustard"]},
    "UP007": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "chickpea", "lentil"]},
    "UP008": {"kharif": ["rice", "maize", "jowar"],             "rabi": ["wheat", "mustard", "barley"]},
    # Bihar
    "BR001": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "chickpea", "lentil", "mustard"]},
    "BR002": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard"]},
    "BR003": {"kharif": ["rice", "maize", "groundnut"],         "rabi": ["wheat", "chickpea", "mustard"]},
    # West Bengal
    "WB001": {"kharif": ["rice", "maize"],                      "rabi": ["wheat", "mustard", "lentil"]},
    "WB002": {"kharif": ["rice"],                               "rabi": ["wheat", "mustard", "lentil"]},
    "WB003": {"kharif": ["rice", "groundnut"],                  "rabi": ["wheat", "mustard"]},
    # MP — soybean heartland
    "MP001": {"kharif": ["soybean", "maize", "jowar"],          "rabi": ["wheat", "chickpea", "lentil"]},
    "MP002": {"kharif": ["soybean", "maize", "jowar"],          "rabi": ["wheat", "chickpea"]},
    "MP003": {"kharif": ["soybean", "rice", "maize"],           "rabi": ["wheat", "chickpea", "lentil"]},
    "MP004": {"kharif": ["soybean", "jowar"],                   "rabi": ["wheat", "chickpea"]},
    "MP005": {"kharif": ["soybean", "rice", "maize"],           "rabi": ["wheat", "chickpea"]},
    "MP006": {"kharif": ["soybean", "maize"],                   "rabi": ["wheat", "chickpea", "mustard"]},
    # Maharashtra — cotton-soybean belt
    "MH001": {"kharif": ["soybean", "cotton", "maize"],         "rabi": ["wheat", "chickpea"]},
    "MH002": {"kharif": ["cotton", "soybean", "jowar"],         "rabi": ["wheat", "chickpea"]},
    "MH003": {"kharif": ["jowar", "bajra", "soybean"],          "rabi": ["wheat", "chickpea", "ragi"]},
    "MH004": {"kharif": ["cotton", "jowar", "soybean"],         "rabi": ["chickpea", "wheat"]},
    "MH005": {"kharif": ["cotton", "soybean"],                  "rabi": ["wheat", "chickpea"]},
    "MH006": {"kharif": ["jowar", "cotton", "soybean"],         "rabi": ["chickpea", "wheat"]},
    "MH007": {"kharif": ["cotton", "jowar", "soybean"],         "rabi": ["chickpea", "wheat"]},
    "MH008": {"kharif": ["jowar", "cotton", "bajra"],           "rabi": ["chickpea", "wheat"]},
    # Telangana — rice-cotton-maize
    "TS001": {"kharif": ["rice", "cotton", "maize"],            "rabi": ["maize", "jowar", "chickpea"]},
    "TS002": {"kharif": ["cotton", "rice", "maize", "jowar"],   "rabi": ["jowar", "chickpea", "maize"]},
    "TS003": {"kharif": ["rice", "cotton", "maize"],            "rabi": ["jowar", "chickpea"]},
    "TS004": {"kharif": ["rice", "maize", "soybean"],           "rabi": ["maize", "chickpea"]},
    "TS005": {"kharif": ["rice", "cotton", "maize"],            "rabi": ["maize", "jowar"]},
    # Andhra Pradesh
    "AP001": {"kharif": ["rice", "cotton", "groundnut"],        "rabi": ["maize", "chickpea", "jowar"]},
    "AP002": {"kharif": ["groundnut", "cotton", "jowar"],       "rabi": ["jowar", "chickpea"]},
    "AP003": {"kharif": ["groundnut", "cotton", "jowar"],       "rabi": ["jowar", "chickpea", "maize"]},
    "AP004": {"kharif": ["rice", "cotton", "groundnut"],        "rabi": ["maize", "chickpea"]},
    "AP005": {"kharif": ["rice", "cotton", "maize"],            "rabi": ["maize", "chickpea"]},
    # Karnataka
    "KA001": {"kharif": ["jowar", "cotton", "maize"],           "rabi": ["jowar", "chickpea"]},
    "KA002": {"kharif": ["ragi", "jowar", "groundnut"],         "rabi": ["ragi", "maize", "chickpea"]},
    "KA003": {"kharif": ["cotton", "jowar"],                    "rabi": ["jowar", "chickpea"]},
    "KA004": {"kharif": ["ragi", "maize"],                      "rabi": ["ragi", "maize", "chickpea"]},
    # Rajasthan
    "RJ001": {"kharif": ["bajra", "jowar", "maize"],            "rabi": ["mustard", "wheat", "barley", "chickpea"]},
    "RJ002": {"kharif": ["bajra", "groundnut"],                 "rabi": ["mustard", "wheat", "barley"]},
    "RJ003": {"kharif": ["bajra", "maize", "jowar"],            "rabi": ["mustard", "wheat", "barley", "chickpea"]},
    "RJ004": {"kharif": ["soybean", "maize", "jowar"],          "rabi": ["wheat", "mustard", "chickpea"]},
    "RJ005": {"kharif": ["bajra", "maize"],                     "rabi": ["mustard", "wheat", "barley"]},
    # Gujarat
    "GJ001": {"kharif": ["cotton", "groundnut", "bajra"],       "rabi": ["wheat", "mustard", "chickpea"]},
    "GJ002": {"kharif": ["groundnut", "cotton", "bajra"],       "rabi": ["wheat", "mustard"]},
    "GJ003": {"kharif": ["groundnut", "cotton", "bajra"],       "rabi": ["wheat", "chickpea"]},
    # Odisha
    "OD001": {"kharif": ["rice", "maize"],                      "rabi": ["maize", "mustard", "lentil"]},
    "OD002": {"kharif": ["rice", "groundnut"],                  "rabi": ["mustard", "lentil"]},
    # Tamil Nadu
    "TN001": {"kharif": ["maize", "groundnut", "cotton"],       "rabi": ["ragi", "maize", "chickpea"]},
    "TN002": {"kharif": ["jowar", "ragi", "maize"],             "rabi": ["ragi", "maize"]},
    "TN003": {"kharif": ["jowar", "ragi", "groundnut"],         "rabi": ["ragi", "chickpea"]},
}

# Name → district_id resolver (lowercase)
NAME_TO_ID = {v["name"].lower(): k for k, v in DISTRICT_CENTROIDS.items()}
# Also add state → list of district IDs
STATE_DISTRICTS = {}
for did, meta in DISTRICT_CENTROIDS.items():
    STATE_DISTRICTS.setdefault(meta["state"].lower(), []).append(did)


def resolve_district_id(name: str) -> str | None:
    """Fuzzy-resolve a district name to its district_id."""
    name_lower = name.strip().lower()
    if name_lower in NAME_TO_ID:
        return NAME_TO_ID[name_lower]
    for key, did in NAME_TO_ID.items():
        if name_lower in key or key in name_lower:
            return did
    return None


def get_district_meta(district_id: str) -> dict | None:
    return DISTRICT_CENTROIDS.get(district_id)


def get_allowed_crops(district_id: str, season: str) -> list:
    calendar = DISTRICT_CROP_CALENDAR.get(district_id, {})
    return calendar.get(season.lower(), [])


ALL_CROPS = [
    "rice", "wheat", "maize", "soybean", "cotton",
    "groundnut", "pigeonpeas", "mungbean", "chickpea",
    "mustard", "lentil", "barley", "jowar", "bajra", "ragi"
]
