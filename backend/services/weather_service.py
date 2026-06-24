import requests
import os
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("WEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# ─── Load ML Model ─────────────────────────────────────
WEATHER_MODEL_PATH = "data/weather_model.pkl"

def get_weather_model():
    if os.path.exists(WEATHER_MODEL_PATH):
        model = joblib.load(WEATHER_MODEL_PATH)
        print("✅ Weather ML model loaded!")
        return model
    print("⚠️ Weather model not found — run train_weather_model.py")
    return None

weather_model = get_weather_model()

# ─── Advisory Messages ─────────────────────────────────
ADVISORY_MESSAGES = {
    'high_temp': {
        'message':  '🌡️ High temperature alert! Irrigate crops early morning or late evening.',
        'severity': 'high'
    },
    'low_temp': {
        'message':  '🥶 Low temperature warning! Protect crops from frost damage.',
        'severity': 'high'
    },
    'fungal_risk': {
        'message':  '🍄 High fungal disease risk! Inspect crops and apply preventive fungicide.',
        'severity': 'medium'
    },
    'high_wind': {
        'message':  '💨 Strong winds! Avoid pesticide and fertilizer spraying today.',
        'severity': 'medium'
    },
    'heavy_rain': {
        'message':  '🌧️ Heavy rainfall! Avoid spraying operations. Check field drainage.',
        'severity': 'high'
    },
    'moderate_rain': {
        'message':  '🌦️ Moderate rainfall. Reduce irrigation today to conserve water.',
        'severity': 'low'
    },
    'normal': {
        'message':  '✅ Weather conditions are suitable for standard farming operations.',
        'severity': 'low'
    },
}

# ─── ML Advisory ───────────────────────────────────────
def ml_advisory(temp, humidity, wind_speed, rainfall):
    if weather_model is None:
        return None
    try:
        features = np.array([[temp, humidity,
                               wind_speed, rainfall]])
        prediction  = weather_model.predict(features)[0]
        proba       = weather_model.predict_proba(features)[0]
        confidence  = round(max(proba) * 100, 2)
        advisory    = ADVISORY_MESSAGES.get(
            prediction, ADVISORY_MESSAGES['normal']
        )
        return {
            'category':   prediction,
            'message':    advisory['message'],
            'severity':   advisory['severity'],
            'confidence': confidence
        }
    except Exception as e:
        print(f"ML advisory error: {e}")
        return None

# ─── Fetch Weather + Generate Advisory ─────────────────
async def get_weather(location: str) -> dict:
    try:
        params = {
            "q":     location,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(BASE_URL, params=params)
        data     = response.json()

        if response.status_code != 200:
            return {
                "success": False,
                "error":   data.get("message", "Location not found")
            }

        # Extract weather parameters
        temp       = data["main"]["temp"]
        humidity   = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"] * 3.6
        rainfall   = data.get("rain", {}).get("1h", 0)
        condition  = data["weather"][0]["description"]
        pressure   = data["main"]["pressure"]
        visibility = data.get("visibility", 0) / 1000

        weather_data = {
            "location":    data["name"] + ", " + data["sys"]["country"],
            "temperature": temp,
            "feels_like":  data["main"]["feels_like"],
            "humidity":    humidity,
            "wind_speed":  round(wind_speed, 3),
            "condition":   condition,
            "rainfall":    rainfall,
            "pressure":    pressure,
            "visibility":  visibility
        }

        # ── ML-based advisory ───────────────────────────
        ml_result  = ml_advisory(temp, humidity,
                                  wind_speed, rainfall)
        advisories = []

        if ml_result:
            advisories.append({
                "source":     "ML Model",
                "category":   ml_result['category'],
                "message":    ml_result['message'],
                "severity":   ml_result['severity'],
                "confidence": ml_result['confidence']
            })

        return {
            "success":          True,
            "weather":          weather_data,
            "advisories":       advisories,
            "total_advisories": len(advisories),
            "advisory_method":  "Decision Tree Classifier"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}