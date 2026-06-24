import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import json
import gdown
from google import genai

# ─── Paths ─────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
MODEL_PATH       = os.path.join(BASE_DIR, "../../ml_models/model.pt")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "../../ml_models/class_names.json")

# ─── Google Drive IDs ───────────────────────────────────────
MODEL_GDRIVE_ID      = "1ncAsRz5y1g6QlifZASx5nsOWfApusZFc"
CLASS_NAMES_GDRIVE_ID = "1JTrlKYt1N5pc_AlbatrZjG1IdR1mrj9a"

# ─── Download Files if Not Found ───────────────────────────
def download_files_if_needed():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Download model.pt
    if not os.path.exists(MODEL_PATH):
        print("Downloading model.pt from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}",
                MODEL_PATH,
                quiet=False
            )
            print("model.pt downloaded!")
        except Exception as e:
            print(f"model.pt download failed: {e}")

    # Download class_names.json
    if not os.path.exists(CLASS_NAMES_PATH):
        print("Downloading class_names.json from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={CLASS_NAMES_GDRIVE_ID}",
                CLASS_NAMES_PATH,
                quiet=False
            )
            print("class_names.json downloaded!")
        except Exception as e:
            print(f"class_names.json download failed: {e}")

# Download on startup
download_files_if_needed()

# ─── Load Class Names ───────────────────────────────────────
# Try multiple paths
possible_class_paths = [
    CLASS_NAMES_PATH,
    r"D:\ALIP-TOTAL\ALIP\ml_models\class_names.json",
    os.path.join(BASE_DIR, "../ml_models/class_names.json")
]

CLASS_NAMES = None
for path in possible_class_paths:
    if os.path.exists(path):
        with open(path, "r") as f:
            CLASS_NAMES = json.load(f)
        print(f"Loaded {len(CLASS_NAMES)} disease classes")
        break

if CLASS_NAMES is None:
    CLASS_NAMES = [
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___healthy",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___healthy"
    ]
    print("Using default class names")

# ─── Treatment Recommendations ─────────────────────────────
TREATMENTS = {
    "Tomato___healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your tomato crop is healthy! Continue regular care.",
        "prevention":"Maintain proper irrigation and inspect regularly."
    },
    "Tomato___Early_blight": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide every 7 days.",
        "prevention":"Avoid overhead irrigation. Remove infected leaves immediately."
    },
    "Tomato___Late_blight": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply Metalaxyl or Cymoxanil fungicide urgently.",
        "prevention":"Plant resistant varieties. Avoid wet and humid conditions."
    },
    "Tomato___Bacterial_spot": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply copper-based bactericides immediately.",
        "prevention":"Use disease-free seeds. Avoid working in wet conditions."
    },
    "Potato___healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your potato crop is healthy! Continue regular care.",
        "prevention":"Monitor regularly for early signs of disease."
    },
    "Potato___Early_blight": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Mancozeb fungicide. Remove infected leaves.",
        "prevention":"Rotate crops. Avoid excess nitrogen fertilization."
    },
    "Potato___Late_blight": {
        "status":    "diseased",
        "severity":  "High",
        "treatment": "Apply Metalaxyl-based fungicide immediately. Spreads fast!",
        "prevention":"Plant certified disease-free seed tubers. Ensure good drainage."
    },
    "Corn_(maize)___healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your corn crop is healthy! Continue regular care.",
        "prevention":"Maintain proper spacing and fertilization schedule."
    },
    "Corn_(maize)___Common_rust_": {
        "status":    "diseased",
        "severity":  "Medium",
        "treatment": "Apply Propiconazole or Azoxystrobin fungicide.",
        "prevention":"Plant rust-resistant varieties. Monitor humidity levels."
    },
    "Pepper,_bell___healthy": {
        "status":    "healthy",
        "severity":  "None",
        "treatment": "Your pepper crop is healthy! Continue regular care.",
        "prevention":"Ensure proper drainage and avoid overwatering."
    }
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

# ─── Load Model ────────────────────────────────────────────
def load_disease_model():
    print("Loading disease model...")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel, len(CLASS_NAMES)
    )

    # Try multiple paths
    possible_paths = [
        MODEL_PATH,
        r"D:\ALIP-TOTAL\ALIP\ml_models\model.pt",
        os.path.join(BASE_DIR, "../../ml_models/model.pt")
    ]

    model_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                state_dict = torch.load(
                    path,
                    map_location=torch.device("cpu")
                )
                model.load_state_dict(state_dict)
                print(f"Disease model loaded from: {path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Error loading from {path}: {e}")

    if not model_loaded:
        print("Running in demo mode - model not loaded")

    model.eval()
    return model

# Load model at startup
disease_model = load_disease_model()

# ─── Predict Disease ───────────────────────────────────────
async def predict_disease(image_bytes: bytes) -> dict:
    try:
        image        = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs       = disease_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = round(confidence.item() * 100, 2)

        treatment_info = TREATMENTS.get(predicted_class, {
            "status":    "unknown",
            "severity":  "Unknown",
            "treatment": "Please consult an agricultural expert.",
            "prevention":"Monitor your crop regularly."
        })

        display_name = predicted_class.replace(
            "___", " - "
        ).replace("_", " ")

        detailed_report = ""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key != "YOUR_GEMINI_API_KEY":
                client = genai.Client(api_key=api_key)
                
                system_prompt = """You are an expert agricultural AI assistant. Your goal is to help farmers quickly understand their crop disease and take action.

Your response MUST be divided into exactly three sections using these EXACT markers:

---FIRST_SCREEN---
---SECOND_SCREEN---
---THIRD_SCREEN---

Generate ALL three sections. Here is the structure:

---FIRST_SCREEN---

# 🚨 Action Required

• Immediate Treatment Required
• Disease Can Spread Quickly
• Start Treatment Within [recommended timeframe based on severity]

# 💊 Recommended Medicines

🥇 **[Best Medicine Name]**
- Effectiveness: ⭐⭐⭐⭐⭐

🥈 **[Second Medicine Name]**
- Effectiveness: ⭐⭐⭐⭐⭐

🥉 **[Third Medicine Name]**
- Effectiveness: ⭐⭐⭐⭐

# ✅ Today's Action Plan

1️⃣ [First urgent step]

2️⃣ [Second step]

3️⃣ [Third step]

4️⃣ [Fourth step]

5️⃣ [Fifth step - monitoring]

Maximum 5 steps. No paragraphs. Short actionable sentences only.

---SECOND_SCREEN---

# ☁ Why This Happened

Generate simple bullet points with icons. Maximum 5 causes.

Example format:
- ☁ High humidity
- 🌧 Recent rainfall
- 💧 Excess leaf wetness
- 🌱 Poor air circulation
- 🌾 Dense planting

# ✅ Prevent Next Time

Generate a checklist. Maximum 6 points.

Example format:
- ✔ Use disease-resistant seeds
- ✔ Avoid watering leaves
- ✔ Maintain proper spacing
- ✔ Remove infected debris
- ✔ Inspect plants weekly

# 🛒 Recommended Products

🥇 **[Product Name]**
- Active Ingredient: [ingredient]
- Best For: [usage]
- Effectiveness: ⭐⭐⭐⭐⭐

🥈 **[Product Name]**
- Active Ingredient: [ingredient]
- Best For: [usage]
- Effectiveness: ⭐⭐⭐⭐⭐

🥉 **[Product Name]**
- Active Ingredient: [ingredient]
- Best For: [usage]
- Effectiveness: ⭐⭐⭐⭐

# 💰 Economic Impact

**Without Treatment:**
- 💰 Yield Loss: [XX]%
- ⚠ Risk: High

**With Treatment:**
- 💰 Yield Loss: [XX]%
- ✅ Risk: Low

[One short sentence summary, e.g. "Timely treatment can save most of the crop."]

# ❓ Frequently Asked Questions

**❓ What is this disease?**
- [Short bullet answer]

**❓ Why did this happen?**
- [Short bullet answer]

**❓ Can it spread to nearby plants?**
- [Short bullet answer]

**❓ Which medicine is best?**
- [Short bullet answer]

**❓ When should I spray?**
- [Short bullet answer]

**❓ How often should I spray?**
- [Short bullet answer]

**❓ Will my yield be affected?**
- [Short bullet answer]

**❓ Is it safe after rain?**
- [Short bullet answer]

**❓ Can I use organic treatment?**
- [Short bullet answer]

**❓ What should I do next season?**
- [Short bullet answer]

Maximum 5 bullets per answer. Keep answers short, practical, farmer-friendly.

---THIRD_SCREEN---

# 🔬 Detailed AI Report

- Infection Stage: [Early / Mid / Advanced]
- Damage Percentage: [XX]%
- Disease Spread Risk: [XX]%
- Urgency Level: [Low / Medium / High / Critical]
- Affected Plant Parts: [list]

# 🧬 Scientific Explanation

- Scientific Name: [name]
- Pathogen Type: [Fungal / Bacterial / Viral]
- Disease Lifecycle: [brief explanation]
- Conditions Favoring Disease: [brief explanation]

# 📅 Disease Spread Forecast (7-Day)

| Day | Risk Level | Reason |
|-----|-----------|--------|
| Day 1 | [High/Medium/Low] | [short reason] |
| Day 2 | [High/Medium/Low] | [short reason] |
| Day 3 | [High/Medium/Low] | [short reason] |
| Day 4 | [High/Medium/Low] | [short reason] |
| Day 5 | [High/Medium/Low] | [short reason] |
| Day 6 | [High/Medium/Low] | [short reason] |
| Day 7 | [High/Medium/Low] | [short reason] |

# 🔍 Advanced Analysis

- Leaf Damage Analysis: [description]
- Color Analysis: [description]
- Pattern Analysis: [description]
- Confidence Breakdown: [description]
- Similar Disease Comparison: [list similar diseases and why this one was chosen]

# 📋 Full Treatment Procedure

- Mixing Ratio: [specific ratio]
- Spray Method: [method]
- Best Time To Spray: [time]
- Repeat Interval: [interval]
- Safety Precautions: [list]

STRICT RULES:
- Use simple farmer-friendly language everywhere.
- Use icons and emojis for visual clarity.
- Use bullet points, NOT paragraphs.
- Keep the FIRST SCREEN short and actionable.
- Do NOT use technical tables on the first screen.
- Every recommendation must be specific to the detected crop and disease.
- Do NOT generate generic advice.
- The three section markers (---FIRST_SCREEN---, ---SECOND_SCREEN---, ---THIRD_SCREEN---) MUST appear exactly as shown."""
                
                ai_prompt = f"Detected Crop and Disease: {display_name}\nConfidence Score: {confidence_score}%\nInitial Severity Assessment: {treatment_info['severity']}\n\n{system_prompt}"
                try:
                    ai_response = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=[ai_prompt, image]
                    )
                except Exception as e:
                    print(f"gemini-2.0-flash failed ({e}). Falling back to gemini-2.0-flash-lite...")
                    ai_response = client.models.generate_content(
                        model='gemini-2.0-flash-lite',
                        contents=[ai_prompt, image]
                    )
                detailed_report = ai_response.text
            else:
                detailed_report = "AI Report generation is currently unavailable. Please configure the GEMINI_API_KEY in the backend settings."
        except Exception as ai_e:
            print(f"AI generation failed: {ai_e}")
            detailed_report = f"Could not generate AI detailed report due to an error: {ai_e}"

        return {
            "success":    True,
            "disease":    display_name,
            "raw_class":  predicted_class,
            "confidence": confidence_score,
            "severity":   treatment_info["severity"],
            "status":     treatment_info["status"],
            "treatment":  treatment_info["treatment"],
            "prevention": treatment_info["prevention"],
            "detailed_report": detailed_report
        }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }
