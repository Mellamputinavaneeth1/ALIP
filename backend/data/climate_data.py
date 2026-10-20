"""
ALIP — Climate Data Module
Historical climate normals for Indian regions (based on IMD 1991-2020 normals).
6 climate zones covering all major Indian agricultural regions.
No external API needed — data embedded directly.
"""

# ─── Climate Zone Profiles ───────────────────────────────────────────────────
# Each zone has seasonal averages: avg_temperature (C), avg_humidity (%), seasonal_rainfall (mm)

CLIMATE_ZONES = {
    "tropical_humid": {
        "description": "Tropical Humid — Eastern India, river deltas",
        "kharif": {"avg_temperature": 29.0, "avg_humidity": 85.0, "seasonal_rainfall": 230.0},
        "rabi":   {"avg_temperature": 22.0, "avg_humidity": 72.0, "seasonal_rainfall": 45.0},
        "zaid":   {"avg_temperature": 32.0, "avg_humidity": 68.0, "seasonal_rainfall": 35.0},
    },
    "semi_arid": {
        "description": "Semi-Arid Deccan — Telangana, Maharashtra, Karnataka",
        "kharif": {"avg_temperature": 30.0, "avg_humidity": 65.0, "seasonal_rainfall": 85.0},
        "rabi":   {"avg_temperature": 24.0, "avg_humidity": 55.0, "seasonal_rainfall": 22.0},
        "zaid":   {"avg_temperature": 34.0, "avg_humidity": 45.0, "seasonal_rainfall": 15.0},
    },
    "arid": {
        "description": "Arid — Rajasthan, western Gujarat desert zones",
        "kharif": {"avg_temperature": 33.0, "avg_humidity": 42.0, "seasonal_rainfall": 40.0},
        "rabi":   {"avg_temperature": 20.0, "avg_humidity": 38.0, "seasonal_rainfall": 10.0},
        "zaid":   {"avg_temperature": 38.0, "avg_humidity": 28.0, "seasonal_rainfall": 5.0},
    },
    "subtropical": {
        "description": "Sub-Tropical Plains — Punjab, Haryana, UP, Bihar",
        "kharif": {"avg_temperature": 28.0, "avg_humidity": 75.0, "seasonal_rainfall": 190.0},
        "rabi":   {"avg_temperature": 16.0, "avg_humidity": 62.0, "seasonal_rainfall": 55.0},
        "zaid":   {"avg_temperature": 33.0, "avg_humidity": 52.0, "seasonal_rainfall": 20.0},
    },
    "temperate_hills": {
        "description": "Temperate Hills — Himachal Pradesh, Uttarakhand, J&K, Nilgiris",
        "kharif": {"avg_temperature": 20.0, "avg_humidity": 72.0, "seasonal_rainfall": 135.0},
        "rabi":   {"avg_temperature": 8.0,  "avg_humidity": 65.0, "seasonal_rainfall": 80.0},
        "zaid":   {"avg_temperature": 18.0, "avg_humidity": 60.0, "seasonal_rainfall": 30.0},
    },
    "coastal": {
        "description": "Coastal Humid — Kerala, coastal Karnataka, Tamil Nadu coast, Goa, Andhra coast",
        "kharif": {"avg_temperature": 29.0, "avg_humidity": 90.0, "seasonal_rainfall": 260.0},
        "rabi":   {"avg_temperature": 26.0, "avg_humidity": 80.0, "seasonal_rainfall": 120.0},
        "zaid":   {"avg_temperature": 31.0, "avg_humidity": 76.0, "seasonal_rainfall": 40.0},
    },
}

# ─── Region → Climate Zone Mapping ──────────────────────────────────────────
# Covers 40+ major Indian states, cities, and regions

REGION_CLIMATE_ZONE = {
    # Telangana — semi_arid
    "telangana":     "semi_arid",
    "hyderabad":     "semi_arid",
    "warangal":      "semi_arid",
    "nizamabad":     "semi_arid",
    "karimnagar":    "semi_arid",
    "khammam":       "semi_arid",
    "nalgonda":      "semi_arid",

    # Andhra Pradesh — coastal/semi_arid
    "andhra pradesh": "semi_arid",
    "vijayawada":    "coastal",
    "visakhapatnam": "coastal",
    "guntur":        "semi_arid",
    "tirupati":      "semi_arid",
    "kurnool":       "semi_arid",

    # Punjab / Haryana — subtropical
    "punjab":        "subtropical",
    "ludhiana":      "subtropical",
    "amritsar":      "subtropical",
    "haryana":       "subtropical",
    "chandigarh":    "subtropical",
    "ambala":        "subtropical",

    # Uttar Pradesh — subtropical
    "uttar pradesh": "subtropical",
    "lucknow":       "subtropical",
    "varanasi":      "subtropical",
    "agra":          "subtropical",
    "kanpur":        "subtropical",
    "allahabad":     "subtropical",
    "prayagraj":     "subtropical",

    # Bihar — subtropical/tropical_humid
    "bihar":         "subtropical",
    "patna":         "subtropical",
    "muzaffarpur":   "tropical_humid",

    # West Bengal — tropical_humid
    "west bengal":   "tropical_humid",
    "kolkata":       "tropical_humid",
    "darjeeling":    "temperate_hills",
    "siliguri":      "tropical_humid",

    # Maharashtra — semi_arid (interior), coastal (Mumbai)
    "maharashtra":   "semi_arid",
    "pune":          "semi_arid",
    "nagpur":        "semi_arid",
    "mumbai":        "coastal",
    "nashik":        "semi_arid",
    "aurangabad":    "semi_arid",
    "kolhapur":      "semi_arid",

    # Madhya Pradesh — semi_arid
    "madhya pradesh": "semi_arid",
    "bhopal":        "semi_arid",
    "indore":        "semi_arid",
    "gwalior":       "semi_arid",
    "jabalpur":      "semi_arid",

    # Karnataka — coastal/semi_arid
    "karnataka":     "semi_arid",
    "bangalore":     "semi_arid",
    "mysore":        "semi_arid",
    "belgaum":       "semi_arid",
    "mangalore":     "coastal",
    "hubli":         "semi_arid",

    # Tamil Nadu — coastal/semi_arid
    "tamil nadu":    "semi_arid",
    "chennai":       "coastal",
    "coimbatore":    "semi_arid",
    "madurai":       "semi_arid",
    "trichy":        "semi_arid",
    "salem":         "semi_arid",
    "ooty":          "temperate_hills",

    # Kerala — coastal
    "kerala":        "coastal",
    "kochi":         "coastal",
    "thiruvananthapuram": "coastal",
    "kozhikode":     "coastal",
    "thrissur":      "coastal",
    "kannur":        "coastal",

    # Goa — coastal
    "goa":           "coastal",
    "panaji":        "coastal",

    # Gujarat — arid/semi_arid
    "gujarat":       "semi_arid",
    "ahmedabad":     "semi_arid",
    "surat":         "coastal",
    "rajkot":        "semi_arid",
    "vadodara":      "semi_arid",

    # Rajasthan — arid
    "rajasthan":     "arid",
    "jaipur":        "arid",
    "jodhpur":       "arid",
    "udaipur":       "semi_arid",
    "bikaner":       "arid",
    "kota":          "semi_arid",

    # Himachal Pradesh / Uttarakhand / J&K — temperate_hills
    "himachal pradesh": "temperate_hills",
    "shimla":        "temperate_hills",
    "dharamsala":    "temperate_hills",
    "manali":        "temperate_hills",
    "uttarakhand":   "temperate_hills",
    "dehradun":      "temperate_hills",
    "nainital":      "temperate_hills",
    "jammu":         "subtropical",
    "kashmir":       "temperate_hills",
    "srinagar":      "temperate_hills",

    # Northeastern states — tropical_humid
    "assam":         "tropical_humid",
    "guwahati":      "tropical_humid",
    "meghalaya":     "tropical_humid",
    "manipur":       "tropical_humid",
    "nagaland":      "tropical_humid",
    "tripura":       "tropical_humid",
    "mizoram":       "tropical_humid",
    "arunachal pradesh": "tropical_humid",
    "sikkim":        "temperate_hills",

    # Odisha / Jharkhand / Chhattisgarh — tropical_humid/semi_arid
    "odisha":        "tropical_humid",
    "bhubaneswar":   "tropical_humid",
    "jharkhand":     "tropical_humid",
    "ranchi":        "tropical_humid",
    "chhattisgarh":  "semi_arid",
    "raipur":        "semi_arid",

    # Default
    "default":       "semi_arid",
}

# ─── Crop → Climate Zone Mapping (for dataset augmentation) ─────────────────
# Maps each of the 22 crops to their typical growing climate zone

CROP_CLIMATE_ZONE = {
    "rice":         "tropical_humid",
    "maize":        "semi_arid",
    "chickpea":     "subtropical",
    "kidneybeans":  "subtropical",
    "pigeonpeas":   "semi_arid",
    "mothbeans":    "arid",
    "mungbean":     "semi_arid",
    "blackgram":    "semi_arid",
    "lentil":       "subtropical",
    "pomegranate":  "semi_arid",
    "banana":       "coastal",
    "mango":        "tropical_humid",
    "grapes":       "semi_arid",
    "watermelon":   "semi_arid",
    "muskmelon":    "semi_arid",
    "apple":        "temperate_hills",
    "orange":       "subtropical",
    "papaya":       "tropical_humid",
    "coconut":      "coastal",
    "cotton":       "semi_arid",
    "jute":         "tropical_humid",
    "coffee":       "temperate_hills",
}


def get_climate_zone(location: str) -> str:
    """Returns the climate zone key for a given location string."""
    loc = location.lower().strip()
    if loc in REGION_CLIMATE_ZONE:
        return REGION_CLIMATE_ZONE[loc]
    for key in REGION_CLIMATE_ZONE:
        if key in loc or loc in key:
            return REGION_CLIMATE_ZONE[key]
    return "semi_arid"  # default for India


def get_climate_normals(location: str, season: str) -> dict:
    """
    Returns historical climate normals for a given location and season.
    Returns: {avg_temperature, avg_humidity, seasonal_rainfall, climate_zone}
    """
    zone_key = get_climate_zone(location)
    season_key = season.lower().strip()
    if season_key not in ("kharif", "rabi", "zaid"):
        season_key = "kharif"

    zone_data = CLIMATE_ZONES[zone_key][season_key]
    return {
        "avg_temperature":  zone_data["avg_temperature"],
        "avg_humidity":     zone_data["avg_humidity"],
        "seasonal_rainfall": zone_data["seasonal_rainfall"],
        "climate_zone":     zone_key,
        "climate_description": CLIMATE_ZONES[zone_key]["description"],
        "season_used":      season_key,
    }
