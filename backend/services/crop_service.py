import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import urllib.request
from data.soil_data import get_soil_data, get_rainfall
from services.weather_service import get_weather

# ─── Paths ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../data/crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../data/label_encoder.pkl")

# ─── Crop Info ──────────────────────────────────────────────
CROP_INFO = {
    "rice": "Best grown in monsoon. Needs waterlogged conditions.",
    "maize": "Versatile crop for moderate climates. Good for Kharif season.",
    "chickpea": "Cool, dry climate crop. Best for Rabi season.",
    "kidneybeans": "Needs warm temperature and moderate rainfall.",
    "pigeonpeas": "Drought resistant. Suitable for tropical climates.",
    "mothbeans": "Highly drought tolerant. Best for arid regions.",
    "mungbean": "Warm humid conditions. Good for Kharif season.",
    "blackgram": "Tropical and subtropical climate crop.",
    "lentil": "Cool dry conditions. Best for Rabi season.",
    "pomegranate": "Semi-arid conditions with hot summers.",
    "banana": "High humidity and warmth needed year round.",
    "mango": "Tropical climate with dry winters.",
    "grapes": "Warm dry climate with cool nights.",
    "watermelon": "Warm temperature and sandy loam soil.",
    "muskmelon": "Warm dry climate. Best for Zaid season.",
    "apple": "Cold winters and mild summers required.",
    "orange": "Subtropical climate preferred.",
    "papaya": "Warm humid conditions year round.",
    "coconut": "Tropical coastal areas.",
    "cotton": "Warm temperature and moderate rainfall.",
    "jute": "Warm humid conditions.",
    "coffee": "Tropical highlands with moderate rainfall."
}

# ─── Download Dataset ───────────────────────────────────────
def download_dataset():
    if not os.path.exists(DATA_PATH):
        print("Downloading crop dataset...")
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Crop_recommendation.csv"
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Dataset downloaded!")
        except Exception as e:
            print(f"Download failed: {e}")

# ─── Train Model ────────────────────────────────────────────
def train_model():
    print("Training crop recommendation model...")
    df = pd.read_csv(DATA_PATH)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model saved!")

    return model, le

# ─── Load or Train ──────────────────────────────────────────
def get_crop_model():
    download_dataset()
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        print("Crop model loaded from disk")
    else:
        model, le = train_model()
    return model, le

crop_model, label_encoder = get_crop_model()

# ─── Recommend Crop ─────────────────────────────────────────
async def recommend_crop(location: str, season: str) -> dict:
    try:
        # Get soil data for location
        soil = get_soil_data(location)

        # Get rainfall for season
        rainfall = get_rainfall(season)

        # Use live weather when available; otherwise fall back to defaults.
        temperature = 28.0
        humidity = 65.0
        weather_source = "fallback-default"
        weather_error = None

        weather_result = await get_weather(location)
        if weather_result.get("success"):
            weather = weather_result["weather"]
            temperature = float(weather["temperature"])
            humidity = float(weather["humidity"])
            weather_source = "openweathermap"
        else:
            weather_error = weather_result.get("error")

        input_data = np.array([[
            soil["N"], soil["P"], soil["K"],
            temperature, humidity,
            soil["ph"], rainfall
        ]])

        # Get probabilities
        probabilities = crop_model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        recommendations = []
        for idx in top_indices:
            crop_name = label_encoder.inverse_transform([idx])[0]
            confidence = round(probabilities[idx] * 100, 2)
            recommendations.append({
                "crop": crop_name,
                "confidence": confidence,
                "description": CROP_INFO.get(
                    crop_name.lower(),
                    "Suitable for your conditions."
                )
            })

        return {
            "success": True,
            "location": location,
            "season": season,
            "soil_type": soil["soil_type"],
            "top_recommendation": recommendations[0]["crop"],
            "confidence": recommendations[0]["confidence"],
            "recommendations": recommendations,
            "weather_source": weather_source,
            "weather_error": weather_error,
            "weather_inputs": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall
            },
            "soil_summary": {
                "nitrogen": soil["N"],
                "phosphorous": soil["P"],
                "potassium": soil["K"],
                "ph": soil["ph"]
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
