"""
ALIP - test_multicrop_v2_rejection.py
Tests the 20-class model with background_other against 30 unseen images of:
Apple, Grape, Soybean, Strawberry.
Verifies if they are either rejected by confidence or classified as background_other.
"""

import os, json, torch, torch.nn as nn, random
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

random.seed(42)
MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v2.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_class_names.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 1.15
CONF_THRESH = 60.0

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = efficientnet_b0(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_f, len(CLASS_NAMES)))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(p):
    img = Image.open(p).convert("RGB")
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out[0] / TEMPERATURE, 0)
        conf, idx = torch.max(probs, 0)
    return CLASS_NAMES[idx.item()], conf.item() * 100

PV = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
OOS_TEST = {
    "Apple___Apple_scab": "Apple scab",
    "Apple___healthy": "Apple healthy",
    "Grape___Black_rot": "Grape black rot",
    "Grape___healthy": "Grape healthy",
    "Soybean___healthy": "Soybean",
    "Strawberry___Leaf_scorch": "Strawberry",
}

print("=" * 75)
print("OUT-OF-SCOPE REJECTION TEST (multicrop_model_v2 with background_other)")
print("=" * 75)

safely_handled = 0
false_positives = 0
total = 0

for folder, label in OOS_TEST.items():
    folder_path = os.path.join(PV, folder)
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    sample = random.sample(imgs, 5)
    for img in sample:
        total += 1
        cls, conf = predict(os.path.join(folder_path, img))
        # An image is SAFELY HANDLED if it is identified as background_other OR low confidence (< 60%)
        if cls == "background_other" or conf < CONF_THRESH:
            safely_handled += 1
            reason = "Classified as background_other" if cls == "background_other" else "Low confidence"
            print(f"  [REJECTED/SAFE] {label:20s} -> {cls:28s} ({conf:5.1f}%) | {reason}")
        else:
            false_positives += 1
            print(f"  [FALSE POSITIVE] {label:20s} -> {cls:28s} ({conf:5.1f}%) | Dangerous!")

print("\n" + "=" * 75)
print(f"SUMMARY: Safely Handled (Rejected): {safely_handled}/{total} = {safely_handled/total*100:.1f}%")
print(f"Dangerous False Positives: {false_positives}/{total} = {false_positives/total*100:.1f}%")
print("=" * 75)
