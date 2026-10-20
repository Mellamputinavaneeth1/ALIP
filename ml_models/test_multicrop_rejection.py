"""
ALIP - Multi-Crop Disease Classifier v1 Rejection Testing Script
Tests model behavior on unsupported crops: Apple, Grape, Soybean, Strawberry.
Verifies that the confidence threshold gate (reject if max confidence < 0.60)
properly rejects images from crops outside the supported 4 crops.
Target: >= 18/20 (90%) rejected.
"""

import os
import glob
import json
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

random.seed(42)

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v1.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_class_names.json"
PV_ROOT      = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"

UNSUPPORTED_TEST_SOURCES = {
    "Apple (healthy)":          os.path.join(PV_ROOT, "Apple___healthy"),
    "Grape (black rot)":        os.path.join(PV_ROOT, "Grape___Black_rot"),
    "Soybean (healthy)":        os.path.join(PV_ROOT, "Soybean___healthy"),
    "Strawberry (leaf scorch)": os.path.join(PV_ROOT, "Strawberry___Leaf_scorch"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[REJECTION TEST] Device: {DEVICE}")

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

TEMPERATURE = 1.15  # Matches production temperature calibration in disease_service.py

print(f"Testing unsupported crops rejection (threshold < 60% confidence, temperature={TEMPERATURE})...\n")

total_tested = 0
total_rejected = 0
results = []

for crop_desc, folder in UNSUPPORTED_TEST_SOURCES.items():
    if not os.path.exists(folder):
        print(f"Warning: Directory not found: {folder}")
        continue
    all_imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_imgs)
    selected = all_imgs[:5]

    for img_path in selected:
        total_tested += 1
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(inp)
            probs = torch.softmax(outputs[0] / TEMPERATURE, dim=0)
            max_prob, pred_idx = torch.max(probs, dim=0)

        conf = max_prob.item() * 100
        pred_class = CLASS_NAMES[pred_idx.item()]
        is_rejected = conf < 60.0

        if is_rejected:
            total_rejected += 1
            status_str = "REJECTED (Correct)"
        else:
            status_str = f"ACCEPTED as {pred_class} (False Positive)"

        filename = os.path.basename(img_path)
        print(f"[{crop_desc:24s}] {filename[:25]:25s} -> Conf: {conf:5.1f}% | {status_str}")
        results.append({
            "crop": crop_desc,
            "filename": filename,
            "confidence": conf,
            "pred_class": pred_class,
            "rejected": is_rejected
        })

rejection_rate = (total_rejected / total_tested) * 100 if total_tested > 0 else 0
passed = total_rejected >= 18 and total_tested == 20

print("\n" + "=" * 60)
print(f"REJECTION TEST RESULTS: {total_rejected}/{total_tested} rejected ({rejection_rate:.1f}%)")
print(f"Target: >= 18/20 (>= 90.0%)")
print(f"Status: {'PASS' if passed else 'FAIL'}")
print("=" * 60)
