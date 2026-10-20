"""
ALIP - Dynamic ML Demand Prediction Service
Replaces static hardcoded thresholds with
ML-predicted dynamic demand thresholds.
Algorithm: Gradient Boosting Regressor
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os, datetime
import shap

os.makedirs("data", exist_ok=True)

# ─── Encodings ─────────────────────────────────────────
CROP_MAP    = {"rice":0,"wheat":1,"maize":2,"cotton":3,
               "sugarcane":4,"groundnut":5,"soybean":6,
               "default":7}
SEASON_MAP  = {"kharif":0,"rabi":1,"zaid":2}
RTYPE_MAP   = {"urban":0,"semi-urban":1,"rural":2}
MSIZE_MAP   = {"large":2,"medium":1,"small":0}
SEASON_MONTHS = {
    1:"rabi", 2:"rabi", 3:"rabi",  4:"zaid",
    5:"zaid", 6:"kharif",7:"kharif",8:"kharif",
    9:"kharif",10:"kharif",11:"rabi",12:"rabi"
}

# ─── Training Data ─────────────────────────────────────
# Features: [crop, season, region_type,
#            population_density, market_size, month]
# Target:   demand_threshold_kg

X_TRAIN = [
    # rice - kharif - urban - high density
    [0,0,0,18480,2,7],[0,0,0,10100,2,7],
    [0,0,0,26903,2,7],[0,0,0,20667,2,8],
    [0,0,0,24252,2,8],[0,0,0,8000,2,7],
    # rice - kharif - semi-urban
    [0,0,1,3100,1,7],[0,0,1,4600,1,8],
    [0,0,1,3900,1,7],[0,0,1,2900,1,8],
    [0,0,1,2700,1,7],[0,0,1,2100,1,8],
    # rice - rabi - urban
    [0,1,0,18480,2,11],[0,1,0,10100,2,12],
    [0,1,0,6900,2,11],[0,1,0,9800,2,12],
    # rice - rabi - semi-urban
    [0,1,1,3100,1,11],[0,1,1,2700,1,12],
    # wheat - rabi - urban
    [1,1,0,8000,2,11],[1,1,0,6400,2,12],
    [1,1,0,5200,2,1],[1,1,0,6900,2,11],
    [1,1,0,3400,2,12],[1,1,0,9800,2,11],
    # wheat - rabi - semi-urban
    [1,1,1,3400,1,12],[1,1,1,2400,1,1],
    [1,1,1,3100,1,11],[1,1,1,2700,1,12],
    # maize - kharif - urban
    [2,0,0,18480,2,7],[2,0,0,4378,2,6],
    [2,0,0,7200,2,7],[2,0,0,9800,2,6],
    # maize - kharif - semi-urban
    [2,0,1,3100,1,7],[2,0,1,4600,1,8],
    [2,0,1,2900,1,7],[2,0,1,2700,1,6],
    # cotton - kharif - urban
    [3,0,0,9800,2,7],[3,0,0,18480,2,6],
    [3,0,0,7200,2,7],[3,0,0,4600,2,8],
    # cotton - kharif - semi-urban
    [3,0,1,4600,1,7],[3,0,1,2800,1,8],
    # cotton - rural
    [3,0,2,1750,0,7],[3,0,2,2100,0,8],
    # sugarcane - kharif - urban
    [4,0,0,11800,2,6],[4,0,0,10100,2,7],
    [4,0,0,7200,2,8],[4,0,0,18480,2,6],
    # groundnut - kharif
    [5,0,1,3600,1,7],[5,0,1,1750,0,6],
    [5,0,0,9800,2,7],[5,0,1,4600,1,8],
    # soybean - kharif
    [6,0,1,2700,1,7],[6,0,1,3400,1,6],
    [6,0,0,5200,2,7],[6,0,1,2100,1,8],
    # === ADDITIONAL TRAINING DATA ===
    # Rice Zaid urban
    [0, 2, 0, 8000, 2, 4], [0, 2, 0, 12000, 2, 4], [0, 2, 0, 20000, 2, 5], [0, 2, 0, 30000, 2, 5],
    [0, 2, 0, 8500, 2, 4], [0, 2, 0, 12500, 2, 4], [0, 2, 0, 20500, 2, 5], [0, 2, 0, 30500, 2, 5],
    # Rice Zaid semi-urban
    [0, 2, 1, 3000, 1, 4], [0, 2, 1, 5000, 1, 4], [0, 2, 1, 3500, 1, 5], [0, 2, 1, 4500, 1, 5],
    [0, 2, 1, 3100, 1, 4], [0, 2, 1, 5100, 1, 4], [0, 2, 1, 3600, 1, 5], [0, 2, 1, 4600, 1, 5],
    # Wheat Kharif urban (off-season)
    [1, 0, 0, 12000, 2, 6], [1, 0, 0, 20000, 2, 7], [1, 0, 0, 30000, 2, 8], [1, 0, 0, 15000, 2, 9],
    [1, 0, 0, 12500, 2, 6], [1, 0, 0, 20500, 2, 7], [1, 0, 0, 30500, 2, 8], [1, 0, 0, 15500, 2, 9],
    # Wheat Kharif semi-urban (off-season)
    [1, 0, 1, 3000, 1, 6], [1, 0, 1, 5000, 1, 7], [1, 0, 1, 4000, 1, 8], [1, 0, 1, 3500, 1, 9],
    [1, 0, 1, 3100, 1, 6], [1, 0, 1, 5100, 1, 7], [1, 0, 1, 4100, 1, 8], [1, 0, 1, 3600, 1, 9],
    # Maize Rabi urban
    [2, 1, 0, 12000, 2, 11], [2, 1, 0, 20000, 2, 12], [2, 1, 0, 30000, 2, 1], [2, 1, 0, 15000, 2, 2],
    [2, 1, 0, 12500, 2, 11], [2, 1, 0, 20500, 2, 12], [2, 1, 0, 30500, 2, 1], [2, 1, 0, 15500, 2, 2],
    # Maize Rabi semi-urban
    [2, 1, 1, 3000, 1, 11], [2, 1, 1, 5000, 1, 12], [2, 1, 1, 4500, 1, 1], [2, 1, 1, 3500, 1, 2],
    [2, 1, 1, 3100, 1, 11], [2, 1, 1, 5100, 1, 12], [2, 1, 1, 4600, 1, 1], [2, 1, 1, 3600, 1, 2],
    # Sugarcane semi-urban
    [4, 0, 1, 3000, 1, 6], [4, 0, 1, 5000, 1, 7], [4, 0, 1, 4000, 1, 8], [4, 0, 1, 3500, 1, 9],
    [4, 0, 1, 3100, 1, 6], [4, 0, 1, 5100, 1, 7], [4, 0, 1, 4100, 1, 8], [4, 0, 1, 3600, 1, 9],
    # Groundnut Rabi
    [5, 1, 0, 12000, 2, 11], [5, 1, 0, 20000, 2, 12], [5, 1, 1, 3000, 1, 11], [5, 1, 1, 5000, 1, 12],
    [5, 1, 0, 12500, 2, 11], [5, 1, 0, 20500, 2, 12], [5, 1, 1, 3100, 1, 11], [5, 1, 1, 5100, 1, 12],
    # Soybean Rabi
    [6, 1, 0, 12000, 2, 11], [6, 1, 0, 20000, 2, 12], [6, 1, 1, 3000, 1, 11], [6, 1, 1, 5000, 1, 12],
    [6, 1, 0, 12500, 2, 11], [6, 1, 0, 20500, 2, 12], [6, 1, 1, 3100, 1, 11], [6, 1, 1, 5100, 1, 12],
    # Rural regions (7 crops * 6 samples = 42 samples)
    # Rice
    [0, 0, 2, 500, 0, 7], [0, 0, 2, 1500, 0, 8], [0, 0, 2, 2500, 0, 9],
    [0, 1, 2, 1000, 0, 11], [0, 1, 2, 2000, 0, 12], [0, 1, 2, 800, 0, 1],
    # Wheat
    [1, 1, 2, 500, 0, 11], [1, 1, 2, 1500, 0, 12], [1, 1, 2, 2500, 0, 1],
    [1, 0, 2, 1000, 0, 7], [1, 0, 2, 2000, 0, 8], [1, 0, 2, 800, 0, 9],
    # Maize
    [2, 0, 2, 500, 0, 7], [2, 0, 2, 1500, 0, 8], [2, 0, 2, 2500, 0, 9],
    [2, 1, 2, 1000, 0, 11], [2, 1, 2, 2000, 0, 12], [2, 1, 2, 800, 0, 1],
    # Cotton
    [3, 0, 2, 500, 0, 7], [3, 0, 2, 1500, 0, 8], [3, 0, 2, 2500, 0, 9],
    [3, 0, 2, 1000, 0, 7], [3, 0, 2, 2000, 0, 8], [3, 0, 2, 800, 0, 9],
    # Sugarcane
    [4, 0, 2, 500, 0, 7], [4, 0, 2, 1500, 0, 8], [4, 0, 2, 2500, 0, 9],
    [4, 0, 2, 1000, 0, 7], [4, 0, 2, 2000, 0, 8], [4, 0, 2, 800, 0, 9],
    # Groundnut
    [5, 0, 2, 500, 0, 7], [5, 0, 2, 1500, 0, 8], [5, 0, 2, 2500, 0, 9],
    [5, 1, 2, 1000, 0, 11], [5, 1, 2, 2000, 0, 12], [5, 1, 2, 800, 0, 1],
    # Soybean
    [6, 0, 2, 500, 0, 7], [6, 0, 2, 1500, 0, 8], [6, 0, 2, 2500, 0, 9],
    [6, 1, 2, 1000, 0, 11], [6, 1, 2, 2000, 0, 12], [6, 1, 2, 800, 0, 1],
    # More population variations
    [0, 0, 0, 12000, 2, 7], [0, 0, 0, 30000, 2, 7], [0, 0, 0, 5000, 2, 7], [0, 0, 0, 8000, 2, 7],
    [0, 0, 1, 1500, 1, 7], [0, 0, 1, 2500, 1, 7], [0, 0, 1, 3500, 1, 7], [0, 0, 1, 4500, 1, 7],
    [1, 1, 0, 12000, 2, 11], [1, 1, 0, 30000, 2, 11], [1, 1, 0, 5000, 2, 11], [1, 1, 0, 8000, 2, 11],
    [1, 1, 1, 1500, 1, 11], [1, 1, 1, 2500, 1, 11], [1, 1, 1, 3500, 1, 11], [1, 1, 1, 4500, 1, 11],
    [2, 0, 0, 12000, 2, 7], [2, 0, 0, 30000, 2, 7], [2, 0, 0, 5000, 2, 7], [2, 0, 0, 8000, 2, 7],
    [2, 0, 1, 1500, 1, 7], [2, 0, 1, 2500, 1, 7], [2, 0, 1, 3500, 1, 7], [2, 0, 1, 4500, 1, 7],
    [4, 0, 0, 12000, 2, 7], [4, 0, 0, 30000, 2, 7], [4, 0, 0, 5000, 2, 7], [4, 0, 0, 8000, 2, 7],
]

Y_TRAIN = [
    # rice kharif urban
    820,650,1100,950,980,700,
    # rice kharif semi-urban
    380,420,360,310,290,250,
    # rice rabi urban
    580,460,420,500,
    # rice rabi semi-urban
    280,250,
    # wheat rabi urban
    720,580,490,610,460,680,
    # wheat rabi semi-urban
    320,240,290,260,
    # maize kharif urban
    560,380,430,480,
    # maize kharif semi-urban
    290,320,280,260,
    # cotton kharif urban
    480,520,460,410,
    # cotton semi-urban
    280,210,
    # cotton rural
    140,160,
    # sugarcane urban
    680,590,520,720,
    # groundnut
    310,180,420,350,
    # soybean
    260,290,380,230,
    # === ADDITIONAL TRAINING DATA ===
    # Rice Zaid urban (8)
    420, 500, 650, 800,
    430, 510, 660, 810,
    # Rice Zaid semi-urban (8)
    210, 280, 230, 260,
    215, 285, 235, 265,
    # Wheat Kharif urban (off-season) (8)
    100, 120, 150, 110,
    105, 125, 155, 115,
    # Wheat Kharif semi-urban (off-season) (8)
    60, 90, 80, 70,
    65, 95, 85, 75,
    # Maize Rabi urban (8)
    400, 480, 550, 430,
    410, 490, 560, 440,
    # Maize Rabi semi-urban (8)
    210, 290, 260, 230,
    215, 295, 265, 235,
    # Sugarcane semi-urban (8)
    250, 320, 280, 260,
    255, 325, 285, 265,
    # Groundnut Rabi (8)
    450, 530, 220, 290,
    460, 540, 225, 295,
    # Soybean Rabi (8)
    420, 500, 210, 280,
    430, 510, 215, 285,
    # Rural regions (42)
    # Rice
    110, 150, 180,
    100, 140, 120,
    # Wheat
    120, 160, 190,
    80, 110, 90,
    # Maize
    100, 140, 170,
    90, 130, 110,
    # Cotton
    90, 120, 140,
    95, 125, 145,
    # Sugarcane
    130, 170, 200,
    135, 175, 205,
    # Groundnut
    100, 130, 160,
    90, 120, 110,
    # Soybean
    110, 140, 170,
    100, 130, 120,
    # More population variations (28)
    600, 1100, 420, 500,
    280, 350, 390, 420,
    580, 950, 410, 480,
    260, 310, 350, 380,
    450, 800, 380, 440,
    240, 300, 340, 370,
    550, 900, 400, 460,
]

MODEL_PATH = "data/demand_model.pkl"

# ─── Train Model ───────────────────────────────────────
def train_demand_model():
    X = np.array(X_TRAIN)
    y = np.array(Y_TRAIN)
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    scores = cross_val_score(
        model, X, y, cv=5, scoring="r2"
    )
    print(f"5-Fold CV R²: "
          f"{scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")
    return model

def get_demand_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_demand_model()

_model = get_demand_model()

# ─── Predict Dynamic Threshold ─────────────────────────
def predict_dynamic_threshold(
    crop: str,
    region_type: str,
    population_density: int,
    market_size: str,
    month: int = None
) -> dict:
    """
    Predict dynamic demand threshold using
    Gradient Boosting Regressor.
    Replaces static hardcoded thresholds.
    """
    if month is None:
        month = datetime.datetime.now().month

    season     = SEASON_MONTHS.get(month, "kharif")
    crop_enc   = CROP_MAP.get(crop.lower(), 7)
    season_enc = SEASON_MAP.get(season, 0)
    rtype_enc  = RTYPE_MAP.get(region_type, 1)
    msize_enc  = MSIZE_MAP.get(market_size, 1)

    features   = np.array([[
        crop_enc, season_enc, rtype_enc,
        population_density, msize_enc, month
    ]])

    predicted  = max(100.0,
                     round(float(_model.predict(features)[0])))

    return {
        "crop":                   crop,
        "season":                 season,
        "month":                  month,
        "region_type":            region_type,
        "population_density":     population_density,
        "market_size":            market_size,
        "predicted_threshold_kg": predicted,
        "threshold_type":         "ML-predicted (dynamic)",
        "model":                  "Gradient Boosting Regressor"
    }
# ─── Supply Trend Analysis ─────────────────────────────
def analyze_supply_trend(
    supply_history: list,
    threshold: float
) -> dict:
    """
    Analyze supply trend over time using
    linear regression on historical data.
    Predicts next period and urgency level.
    Enhancement 3: Supply Trend Analysis
    """
    if len(supply_history) < 2:
        return {
            "trend":      "insufficient_data",
            "prediction": None,
            "urgency":    "unknown"
        }

    x = np.arange(len(supply_history), dtype=float)
    y = np.array(supply_history, dtype=float)

    # Linear regression coefficients
    coeffs    = np.polyfit(x, y, 1)
    slope     = coeffs[0]
    intercept = coeffs[1]

    # Next period prediction
    next_val  = slope * len(supply_history) + intercept
    next_val  = max(0.0, round(float(next_val), 2))

    # Trend classification
    mean_y      = float(np.mean(y))
    pct_change  = (slope / (mean_y + 1e-9)) * 100

    if pct_change > 15:
        trend   = "rapidly_increasing"
        urgency = "critical"
    elif pct_change > 5:
        trend   = "increasing"
        urgency = "high"
    elif pct_change < -15:
        trend   = "rapidly_decreasing"
        urgency = "monitor"
    elif pct_change < -5:
        trend   = "decreasing"
        urgency = "low"
    else:
        trend   = "stable"
        urgency = "normal"

    # Weeks until oversupply breach (1.5x threshold)
    breach_level      = threshold * 1.5
    weeks_to_breach   = None
    if slope > 0 and y[-1] < breach_level:
        remaining       = breach_level - y[-1]
        weeks_to_breach = max(0, int(round(remaining / slope)))

    # Trend message
    # In analyze_supply_trend, update message for critical:
    if urgency == "critical":
      if y[-1] >= breach_level:
        message = (
            f"CRITICAL: Supply already at {y[-1]:.0f} kg — "
            f"exceeds breach level of {breach_level:.0f} kg. "
            f"Immediate redistribution required!"
        )
      else:
        message = (
            f"CRITICAL: Supply rapidly increasing "
            f"({pct_change:.1f}%/period). "
            f"Oversupply breach expected in "
            f"{weeks_to_breach} weeks."
        )
    elif urgency == "high":
        message = (
            f"WARNING: Supply increasing steadily. "
            f"Monitor closely — breach in "
            f"{weeks_to_breach or 'unknown'} weeks."
        )
    elif urgency == "monitor":
        message = (
            f"MONITOR: Supply rapidly declining. "
            f"Undersupply risk developing."
        )
    elif urgency == "low":
        message = "Supply gradually declining. Normal seasonal pattern."
    else:
        message = "Supply stable within normal range."

    return {
        "history":           supply_history,
        "periods_analysed":  len(supply_history),
        "slope_per_period":  round(float(slope), 2),
        "pct_change":        round(float(pct_change), 2),
        "predicted_next_kg": next_val,
        "trend":             trend,
        "urgency":           urgency,
        "weeks_to_breach":   weeks_to_breach,
        "threshold":         threshold,
        "breach_level":      breach_level,
        "message":           message
    }
# ─── Transport Rate Configuration ──────────────────────
TRANSPORT_RATES = {
    "small":  {"label": "Small (< 3 tonnes)",  "rate_per_km": 25.0},
    "medium": {"label": "Medium (3-10 tonnes)", "rate_per_km": 35.0},
    "large":  {"label": "Large (> 10 tonnes)",  "rate_per_km": 50.0},
}
DEFAULT_TRUCK_TYPE = "medium"

# ─── Perishability Tiers ───────────────────────────────
PERISHABILITY_TIERS = {
    "tomato": 1, "sugarcane": 1, "banana": 1,
    "maize": 2, "groundnut": 2, "potato": 2, "onion": 2,
    "rice": 3, "wheat": 3, "cotton": 3, "soybean": 3,
}

# ─── Price Sensitivity Database ────────────────────────
PRICE_SENSITIVITY = {
    "rice":      {"base_price": 22,  "elasticity": -0.35},
    "wheat":     {"base_price": 21,  "elasticity": -0.30},
    "maize":     {"base_price": 18,  "elasticity": -0.40},
    "cotton":    {"base_price": 65,  "elasticity": -0.25},
    "sugarcane": {"base_price": 3,   "elasticity": -0.20},
    "groundnut": {"base_price": 55,  "elasticity": -0.45},
    "soybean":   {"base_price": 43,  "elasticity": -0.38},
    "default":   {"base_price": 25,  "elasticity": -0.35}
}

def estimate_price_impact(
    crop: str,
    current_supply_kg: float,
    threshold_kg: float,
    distance_km: float = 0.0,
    quantity_kg: float = 100.0,
    truck_type: str = "medium"
) -> dict:
    """
    Estimate price drop due to oversupply using
    price elasticity model.
    Enhancement 4: Price Impact Estimation.
    Formula: %ΔP = elasticity × %ΔQ
    """
    sensitivity = PRICE_SENSITIVITY.get(
        crop.lower(),
        PRICE_SENSITIVITY["default"]
    )
    base_price = sensitivity["base_price"]
    elasticity = sensitivity["elasticity"]

    oversupply_ratio = (
        (current_supply_kg - threshold_kg) / threshold_kg
    )

    if oversupply_ratio <= 0:
        return {
            "crop":           crop,
            "status":         "no_oversupply",
            "price_impact_pct": 0,
            "recommendation": "No price intervention needed."
        }

    # Price elasticity formula
    pct_supply_increase = oversupply_ratio * 100
    pct_price_change    = elasticity * (pct_supply_increase / 100) * 100
    estimated_price     = base_price * (1 + pct_price_change / 100)
    estimated_price     = max(0.0, round(estimated_price, 2))

    loss_per_100kg = round(
        (base_price - estimated_price) * 100, 2
    )

    urgency = (
        "critical" if pct_price_change < -30
        else "high" if pct_price_change < -15
        else "medium"
    )

    if urgency == "critical":
        rec = (
            f"URGENT: Redistribute {crop} immediately. "
            f"Estimated {abs(pct_price_change):.1f}% price drop "
            f"(₹{estimated_price}/kg from ₹{base_price}/kg). "
            f"Loss: ₹{loss_per_100kg:.0f} per 100 kg."
        )
    elif urgency == "high":
        rec = (
            f"Redistribute {crop} within 48 hours. "
            f"Estimated {abs(pct_price_change):.1f}% price drop "
            f"if supply remains unchanged."
        )
    else:
        rec = (
            f"Monitor {crop} supply. "
            f"Moderate {abs(pct_price_change):.1f}% "
            f"potential price drop detected."
        )

    # Transport cost calculation (fixed: ₹35/km for medium truck)
    rate_info = TRANSPORT_RATES.get(
        truck_type, TRANSPORT_RATES[DEFAULT_TRUCK_TYPE]
    )
    transport_total = round(rate_info["rate_per_km"] * distance_km, 2)
    transport_per_100kg = (
        round((transport_total / quantity_kg) * 100, 2)
        if quantity_kg > 0 else 0.0
    )

    # ROI calculation
    price_gain_per_100kg = loss_per_100kg  # what they lose by holding
    roi = (
        round(price_gain_per_100kg / transport_per_100kg, 2)
        if transport_per_100kg > 0 else 0.0
    )
    roi_verdict = (
        "STRONGLY RECOMMENDED" if roi > 3
        else "RECOMMENDED" if roi > 1.5
        else "MARGINAL" if roi > 1
        else "NOT RECOMMENDED"
    )

    return {
        "crop":                     crop,
        "current_supply_kg":        current_supply_kg,
        "threshold_kg":             threshold_kg,
        "oversupply_ratio":         round(oversupply_ratio, 3),
        "base_price_per_kg":        base_price,
        "estimated_price_per_kg":   estimated_price,
        "price_drop_pct":           round(abs(pct_price_change), 2),
        "estimated_loss_per_100kg": loss_per_100kg,
        "urgency":                  urgency,
        "recommendation":           rec,
        "model":                    "Price elasticity model",
        "transport": {
            "truck_type":           truck_type,
            "truck_label":          rate_info["label"],
            "rate_per_km":          rate_info["rate_per_km"],
            "distance_km":          distance_km,
            "total_cost":           transport_total,
            "cost_per_100kg":       transport_per_100kg
        },
        "roi":                      roi,
        "roi_verdict":              roi_verdict
    }

def evaluate_demand_model():
    """
    Publication-ready evaluation of the Demand Threshold model.
    Reports: R², MAE, RMSE, and MAPE across 5-Fold and LOO CV.
    
    Why these metrics?
    - R²: Overall variance explained (primary metric for regression)
    - MAE: Average kg error (interpretable in domain units)
    - RMSE: Penalizes large errors (important for demand planning)
    - MAPE: Percentage error (comparable across different crop scales)
    
    Also performs Leave-One-Out CV since dataset is small,
    which gives unbiased estimate of generalization.
    """
    X = np.array(X_TRAIN)
    y = np.array(Y_TRAIN)
    
    print("=== Publication-Ready Evaluation of Demand Threshold Model ===")
    print(f"Total Samples: {len(X)}")
    
    # Base Model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, mae_scores, rmse_scores, mape_scores = [], [], [], []
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        r2_scores.append(r2_score(y_te, preds))
        mae_scores.append(mean_absolute_error(y_te, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))
        mape_scores.append(np.mean(np.abs((y_te - preds) / y_te)) * 100)
        
    print("\n--- 5-Fold Cross Validation ---")
    print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"MAE:  {np.mean(mae_scores):.2f} kg ± {np.std(mae_scores):.2f}")
    print(f"RMSE: {np.mean(rmse_scores):.2f} kg ± {np.std(rmse_scores):.2f}")
    print(f"MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")

    # Leave-One-Out CV
    loo = LeaveOneOut()
    loo_preds, loo_actual = [], []
    
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        model.fit(X_tr, y_tr)
        loo_preds.append(model.predict(X_te)[0])
        loo_actual.append(y_te[0])
        
    loo_preds = np.array(loo_preds)
    loo_actual = np.array(loo_actual)
    
    print("\n--- Leave-One-Out Cross Validation ---")
    print(f"R²:   {r2_score(loo_actual, loo_preds):.4f}")
    print(f"MAE:  {mean_absolute_error(loo_actual, loo_preds):.2f} kg")
    print(f"RMSE: {np.sqrt(mean_squared_error(loo_actual, loo_preds)):.2f} kg")
    print(f"MAPE: {np.mean(np.abs((loo_actual - loo_preds) / loo_actual)) * 100:.2f}%")

    # Feature Importance
    model.fit(X, y)
    feature_names = ["Crop", "Season", "RegionType", "PopulationDensity", "MarketSize", "Month"]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\n--- Feature Importances ---")
    feat_imp_dict = {}
    for idx in sorted_idx:
        print(f"{feature_names[idx]:<20}: {importances[idx]:.4f}")
        feat_imp_dict[feature_names[idx]] = float(importances[idx])
        
    return {
        'kfold_r2_mean': float(np.mean(r2_scores)),
        'kfold_r2_std': float(np.std(r2_scores)),
        'kfold_mae_mean': float(np.mean(mae_scores)),
        'kfold_rmse_mean': float(np.mean(rmse_scores)),
        'kfold_mape_mean': float(np.mean(mape_scores)),
        'loo_r2': float(r2_score(loo_actual, loo_preds)),
        'loo_mae': float(mean_absolute_error(loo_actual, loo_preds)),
        'loo_rmse': float(np.sqrt(mean_squared_error(loo_actual, loo_preds))),
        'loo_mape': float(np.mean(np.abs((loo_actual - loo_preds) / loo_actual)) * 100),
        'feature_importances': feat_imp_dict,
        'sample_count': int(len(X))
    }

def noise_robustness_demand(noise_levels=None):
    """
    Tests GBR robustness to noisy population density and
    month inputs (the most variable real-world features).
    Proves the model doesn't memorize exact training values.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        
    X_base = np.array(X_TRAIN, dtype=float)
    y_base = np.array(Y_TRAIN, dtype=float)
    
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    
    print("\n=== Robustness to Noise ===")
    print("Injecting noise into Population Density (idx 3) and Month (idx 5)")
    
    loo = LeaveOneOut()
    
    for noise in noise_levels:
        loo_preds, loo_actual = [], []
        
        for train_idx, test_idx in loo.split(X_base):
            X_tr, X_te = np.copy(X_base[train_idx]), np.copy(X_base[test_idx])
            y_tr, y_te = y_base[train_idx], y_base[test_idx]
            
            if noise > 0:
                X_te[0, 3] = X_te[0, 3] * (1.0 + np.random.normal(0, noise))
                X_te[0, 5] = X_te[0, 5] * (1.0 + np.random.normal(0, noise))
            
            model.fit(X_tr, y_tr)
            loo_preds.append(model.predict(X_te)[0])
            loo_actual.append(y_te[0])
            
        r2 = r2_score(loo_actual, loo_preds)
        print(f"Noise Level {noise*100:5.1f}% -> LOO-CV R²: {r2:.4f}")

def evaluate_demand_baselines():
    X = np.array(X_TRAIN)
    y = np.array(Y_TRAIN)
    
    models = {
        "GBR": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "SVR (RBF)": SVR(kernel='rbf')
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    gbr_r2 = []
    
    # Pre-compute GBR scores for t-test
    gbr_scores = cross_val_score(models["GBR"], X, y, cv=kf, scoring="r2")
    gbr_r2 = gbr_scores.tolist()
    
    print(f"{'Model':<20} | {'R2 Mean':<10} | {'MAE Mean':<10} | {'p-value (vs GBR)':<15}")
    print("-" * 65)
    
    for name, model in models.items():
        r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
        neg_mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        mae = -neg_mae
        
        p_val_str = "N/A"
        p_val = None
        if name != "GBR":
            t_stat, p_val = stats.ttest_rel(gbr_r2, r2)
            p_val_str = f"{p_val:.4f}"
            
        print(f"{name:<20} | {r2.mean():<10.4f} | {mae.mean():<10.4f} | {p_val_str:<15}")
        
        results[name] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std()),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "p_value_vs_gbr": float(p_val) if p_val is not None else None
        }
        
    return results

def evaluate_demand_shap():
    X = np.array(X_TRAIN)
    y = np.array(Y_TRAIN)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    feature_names = ["Crop", "Season", "RegionType", "PopulationDensity", "MarketSize", "Month"]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, mean_abs_shap)}
    
    sorted_features = sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)
    
    return {
        "mean_abs_shap": shap_dict,
        "ranking": [feat for feat, _ in sorted_features]
    }