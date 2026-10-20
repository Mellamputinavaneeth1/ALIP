import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import io
import os
import json
from google import genai
from services.weather_service import get_weather

# ─── Paths ─────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
MODEL_PATH       = os.path.join(BASE_DIR, "../../ml_models/multicrop_model_v2.pt")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "../../ml_models/multicrop_v2_class_names.json")

# ─── Confidence Thresholds ──────────────────────────────────
CONF_HIGH   = 0.80   # Show result normally
CONF_LOW    = 0.60   # Show result with "low confidence" warning
TEMPERATURE = 1.15   # Softmax temperature calibration for out-of-domain rejection
# < CONF_LOW → Reject: "Please upload a clearer image"

# ─── Load Class Names ───────────────────────────────────────
CLASS_NAMES = None
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"Loaded {len(CLASS_NAMES)} disease classes: {CLASS_NAMES}")
else:
    # Fallback while model is being trained
    CLASS_NAMES = [
        "corn_cercospora_leaf_spot",
        "corn_common_rust",
        "corn_healthy",
        "corn_northern_leaf_blight",
        "pepper_bacterial_spot",
        "pepper_healthy",
        "potato_early_blight",
        "potato_healthy",
        "potato_late_blight",
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_healthy",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_mosaic_virus",
        "tomato_septoria_leaf_spot",
        "tomato_spider_mites",
        "tomato_target_spot",
        "tomato_yellow_leaf_curl_virus",
    ]
    print("Using default class names (model not yet trained)")

# ─── Treatment Recommendations ─────────────────────────────
TREATMENTS = {
    # --- Corn ---
    "corn_healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your corn crop is healthy! Continue regular care.",
        "prevention":"Practice crop rotation. Monitor for early signs of disease."
    },
    "corn_common_rust": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb or Propiconazole fungicide every 7-14 days.",
        "prevention":"Plant resistant varieties. Avoid dense planting."
    },
    "corn_northern_leaf_blight": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply Azoxystrobin or Propiconazole fungicide at early tasseling.",
        "prevention":"Use resistant hybrids. Rotate crops. Remove infected debris."
    },
    "corn_cercospora_leaf_spot": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Strobilurin or Triazole fungicides at flag-leaf stage.",
        "prevention":"Improve air circulation. Use resistant varieties. Rotate crops."
    },

    # --- Potato ---
    "potato_healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your potato crop is healthy! Maintain regular monitoring.",
        "prevention":"Use certified disease-free seed tubers. Rotate crops."
    },
    "potato_early_blight": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide every 7-10 days.",
        "prevention":"Use certified seed potatoes. Avoid overhead irrigation. Destroy crop debris."
    },
    "potato_late_blight": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply Metalaxyl-M or Cymoxanil fungicide urgently within 24 hours.",
        "prevention":"Plant resistant varieties. Avoid excessive irrigation. Ensure good drainage."
    },

    # --- Pepper ---
    "pepper_healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your pepper crop is healthy! Continue regular care.",
        "prevention":"Maintain proper spacing and irrigation practices."
    },
    "pepper_bacterial_spot": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply copper-based bactericides (copper hydroxide or copper sulfate).",
        "prevention":"Use disease-free seeds. Avoid overhead watering. Remove infected plant parts."
    },

    # --- Tomato ---
    "tomato_healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your tomato crop is healthy! Continue regular care.",
        "prevention":"Maintain proper irrigation and inspect regularly."
    },
    "tomato_early_blight": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide every 7 days.",
        "prevention":"Avoid overhead irrigation. Remove infected leaves immediately."
    },
    "tomato_late_blight": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply Metalaxyl or Cymoxanil fungicide urgently.",
        "prevention":"Plant resistant varieties. Avoid wet and humid conditions."
    },
    "tomato_bacterial_spot": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply copper-based bactericides immediately.",
        "prevention":"Use disease-free seeds. Avoid working in wet conditions."
    },
    "tomato_leaf_mold": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide. Improve air circulation.",
        "prevention":"Ensure good ventilation in greenhouse. Avoid leaf wetness."
    },
    "tomato_septoria_leaf_spot": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Chlorothalonil or Mancozeb fungicide every 7-10 days.",
        "prevention":"Avoid wetting foliage. Remove infected leaves. Rotate crops."
    },
    "tomato_spider_mites": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply miticide (Abamectin or Spiromesifen). Use insecticidal soap spray.",
        "prevention":"Keep plants well-watered. Introduce beneficial predatory mites."
    },
    "tomato_target_spot": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Azoxystrobin or Chlorothalonil fungicide every 7-10 days.",
        "prevention":"Avoid excessive nitrogen. Ensure good air circulation."
    },
    "tomato_mosaic_virus": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "No chemical cure. Remove and destroy infected plants immediately.",
        "prevention":"Use virus-free seed. Control aphid vectors. Disinfect tools."
    },
    "tomato_yellow_leaf_curl_virus": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "No chemical cure. Remove infected plants. Control whitefly vector.",
        "prevention":"Use reflective mulch to repel whiteflies. Apply systemic insecticides for vector control."
    },
}

# ─── Display Names ──────────────────────────────────────────
DISPLAY_NAMES = {
    # Corn
    "corn_cercospora_leaf_spot":     "Corn - Cercospora Leaf Spot (Gray Leaf Spot)",
    "corn_common_rust":              "Corn - Common Rust",
    "corn_healthy":                  "Corn - Healthy",
    "corn_northern_leaf_blight":     "Corn - Northern Leaf Blight",

    # Pepper
    "pepper_bacterial_spot":         "Pepper - Bacterial Spot",
    "pepper_healthy":                "Pepper - Healthy",

    # Potato
    "potato_early_blight":           "Potato - Early Blight",
    "potato_healthy":                "Potato - Healthy",
    "potato_late_blight":            "Potato - Late Blight",

    # Tomato
    "tomato_bacterial_spot":         "Tomato - Bacterial Spot",
    "tomato_early_blight":           "Tomato - Early Blight",
    "tomato_healthy":                "Tomato - Healthy",
    "tomato_late_blight":            "Tomato - Late Blight",
    "tomato_leaf_mold":              "Tomato - Leaf Mold",
    "tomato_mosaic_virus":           "Tomato - Mosaic Virus",
    "tomato_septoria_leaf_spot":     "Tomato - Septoria Leaf Spot",
    "tomato_spider_mites":           "Tomato - Spider Mites (Two-spotted)",
    "tomato_target_spot":            "Tomato - Target Spot",
    "tomato_yellow_leaf_curl_virus": "Tomato - Yellow Leaf Curl Virus",
}

# ─── Image Preprocessing ───────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── Load EfficientNet-B0 Model ────────────────────────────
def load_disease_model():
    print("Loading multi-crop disease model (EfficientNet-B0)...")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(
                MODEL_PATH,
                map_location=torch.device("cpu"),
                weights_only=True
            )
            model.load_state_dict(state_dict)
            print(f"Multi-crop disease model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in demo mode.")
    else:
        print(f"Model not found at {MODEL_PATH}. Running in demo mode.")

    model.eval()
    return model

disease_model = load_disease_model()

# ─── Predict Disease ───────────────────────────────────────
async def predict_disease(image_bytes: bytes, location: str = None) -> dict:
    try:
        image        = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs       = disease_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0] / TEMPERATURE, dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        confidence_score  = round(confidence.item() * 100, 2)
        predicted_class   = CLASS_NAMES[predicted_idx.item()]
        crop_name         = predicted_class.split("_")[0].title()

        # ── Confidence Gate ─────────────────────────────────
        if confidence_score < CONF_LOW * 100 or predicted_class == "background_other":
            return {
                "success":  False,
                "rejected": True,
                "error":    (
                    f"Confidence too low ({confidence_score}%) or unsupported plant detected. "
                    "Please upload a clear, close-up photo of a single supported crop leaf. "
                    "Avoid backgrounds, multiple leaves, or blurry images. "
                    "Supported crops: Tomato, Potato, Corn, Pepper."
                )
            }

        confidence_level = (
            "high"    if confidence_score >= CONF_HIGH * 100
            else "low"
        )

        treatment_info = TREATMENTS.get(predicted_class, {
            "status":    "unknown",
            "severity":  "Unknown",
            "treatment": "Please consult an agricultural expert.",
            "prevention":"Monitor your crop regularly."
        })

        display_name = DISPLAY_NAMES.get(predicted_class, predicted_class.replace("_", " ").title())

        # ── Fetch weather if location provided ───────────────
        weather_data    = None
        weather_context = ""
        if location and location.strip():
            try:
                weather_result = await get_weather(location.strip())
                if weather_result.get("success"):
                    w = weather_result["weather"]
                    weather_data = w
                    weather_context = (
                        f"\n\nCurrent Weather at {w['location']}:\n"
                        f"- Temperature: {w['temperature']}°C (Feels like: {w['feels_like']}°C)\n"
                        f"- Humidity: {w['humidity']}%\n"
                        f"- Wind Speed: {w['wind_speed']} km/h\n"
                        f"- Rainfall (last 1h): {w['rainfall']} mm\n"
                        f"- Condition: {w['condition']}\n"
                    )
            except Exception as we:
                print(f"Weather fetch error: {we}")

        # ── Gemini AI Report ─────────────────────────────────
        def _generate_fallback_report(name, confidence, info, crop_name="Unknown"):
            sev    = info.get("severity", "Medium")
            treat  = info.get("treatment", "Consult an agricultural expert.")
            prev   = info.get("prevention", "Monitor your crop regularly.")
            status = info.get("status", "diseased")

            if status == "healthy":
                return f"""---FIRST_SCREEN---

# Action Required

Your {crop_name} crop appears healthy! No immediate treatment needed.

- Continue regular monitoring
- Maintain good agricultural practices
- Check plants weekly for early signs of disease

# Recommended Medicines

No medicines needed at this time.

# Today's Action Plan

1. Continue regular watering schedule
2. Monitor leaves for any discoloration
3. Ensure proper spacing between plants
4. Apply preventive organic fungicide if desired

---SECOND_SCREEN---

# Why This Happened

- Your plant is healthy due to good growing conditions
- Proper nutrition and watering are maintaining plant health

# Prevention Checklist

- {prev}
- Water at the base of plants, not overhead
- Ensure good air circulation between plants
- Rotate crops each season
- Remove any dead or decaying plant material

# Frequently Asked Questions

**How often should I check my plants?**
Check your plants at least once a week for any signs of disease.

---THIRD_SCREEN---

# Detailed AI Report

- Crop: {crop_name}
- Status: Healthy
- Confidence: {confidence}%
- Severity: None
"""

            urgency = "24 hours" if sev == "High" else "2-3 days" if sev == "Medium" else "1 week"
            return f"""---FIRST_SCREEN---

# Action Required

- Immediate Treatment Required
- Disease Can Spread Quickly
- Start Treatment Within {urgency}

# Recommended Medicines

🥇 **{treat.split('.')[0].strip() if '.' in treat else treat}**
- Effectiveness: Recommended primary treatment
- Apply as directed on product label

🥈 **Copper-based Fungicide**
- Effectiveness: Broad-spectrum disease control

🥉 **Neem Oil Spray**
- Effectiveness: Natural disease suppressant

# Today's Action Plan

1. Remove and destroy all severely infected leaves immediately
2. Apply recommended medicine
3. Ensure proper spacing between plants for air circulation
4. Water at the base of plants — avoid wetting the leaves
5. Monitor neighboring plants for early signs of spread

---SECOND_SCREEN---

# Why This Happened

- Humid weather conditions favor disease development
- Overhead watering spreads disease spores between plants
- Poor air circulation between plants creates ideal conditions

# Prevention Checklist

- {prev}
- Practice crop rotation — do not plant same crop 2-3 seasons in a row
- Water at the base of plants early in the morning
- Remove and destroy infected plant debris promptly

# Economic Impact

- **Without Treatment**: Potential yield loss of 30-60%
- **With Treatment**: Yield loss can be minimized to 5-15%

---THIRD_SCREEN---

# Detailed AI Report

- Infection Stage: Detected
- Severity: {sev}
- Urgency Level: {'High' if sev == 'High' else 'Medium'}
- Confidence: {confidence}%

# Scientific Information

- Affected Crop: {crop_name}
- Disease: {name}
- Transmission: Air, water splash, contaminated tools, infected seeds
- Favorable Conditions: High humidity, warm temperatures, poor air circulation

# Full Treatment Procedure

- Primary Treatment: {treat}
- Prevention: {prev}
- Monitor for 10-14 days after treatment
"""

        detailed_report = ""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key != "YOUR_GEMINI_API_KEY":
                client = genai.Client(api_key=api_key)

                system_prompt = """You are an expert agricultural AI assistant helping farmers understand crop disease and take action.

Your response MUST be divided into exactly three sections with these exact headings:
---FIRST_SCREEN---
# Action Required
# Recommended Medicines
# Today's Action Plan

---SECOND_SCREEN---
# Why This Happened
# Prevention Checklist
# Economic Impact
# Frequently Asked Questions

---THIRD_SCREEN---
# Detailed AI Report
# Scientific Explanation
# Recovery Timeline
# Treatment Procedure

CRITICAL FORMATTING RULES:
1. Under `# Recommended Medicines`, you MUST provide exactly 3 choices formatted exactly like this (including the emojis):
🥇 **[1st Medicine Name]**
- [Detail 1]
- [Detail 2]
🥈 **[2nd Medicine Name]**
- [Detail 1]
- [Detail 2]
🥉 **[3rd Medicine Name]**
- [Detail 1]
- [Detail 2]

2. Under `# Recovery Timeline`, you MUST format each step exactly like this:
Treatment Step: [Step Name]
Estimated Time: [Time]
Expected Outcome: [Outcome]

3. Under `# Frequently Asked Questions`, you MUST format each question exactly like this:
**❓ [Question]**
[Answer]

Use simple farmer-friendly language. Use bullet points for Action Plan, Prevention, and Why This Happened. Be specific to the detected disease."""

                ai_prompt = (
                    f"Crop: {crop_name}\n"
                    f"Detected Disease: {display_name}\n"
                    f"Confidence: {confidence_score}%\n"
                    f"Severity: {treatment_info['severity']}"
                    f"{weather_context}\n\n{system_prompt}"
                )

                models_to_try = [
                    'gemini-3.6-flash',
                    'gemini-3.5-flash-lite',
                    'gemini-2.5-flash',
                ]
                ai_response = None
                for model_name in models_to_try:
                    try:
                        ai_response = client.models.generate_content(
                            model=model_name,
                            contents=[ai_prompt, image]
                        )
                        break
                    except Exception:
                        continue

                if ai_response and ai_response.text:
                    detailed_report = ai_response.text
                else:
                    detailed_report = _generate_fallback_report(
                        display_name, confidence_score, treatment_info, crop_name
                    )
            else:
                detailed_report = _generate_fallback_report(
                    display_name, confidence_score, treatment_info, crop_name
                )
        except Exception as ai_e:
            print(f"AI generation failed: {ai_e}")
            detailed_report = _generate_fallback_report(
                display_name, confidence_score, treatment_info, crop_name
            )

        return {
            "success":          True,
            "disease":          display_name,
            "crop":             crop_name,
            "raw_class":        predicted_class,
            "confidence":       confidence_score,
            "confidence_level": confidence_level,
            "severity":         treatment_info["severity"],
            "status":           treatment_info["status"],
            "treatment":        treatment_info["treatment"],
            "prevention":       treatment_info["prevention"],
            "detailed_report":  detailed_report,
            "weather":          weather_data,
            "model_version":    "multicrop_v2_efficientnet_b0",
        }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }
