"""
ALIP - test_v2_field_images.py
Tests multicrop_model_v2.pt on real field images in real_field_images/
"""

import os, json, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v2.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_class_names.json"
REAL_DIR     = r"D:\ALIP-TOTAL\ALIP\ml_models\real_field_images"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE  = 1.15
CONF_THRESH  = 60.0

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

print("=" * 75)
print("REAL FIELD IMAGES TEST (multicrop_model_v2)")
print("=" * 75)

real_imgs = sorted([f for f in os.listdir(REAL_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
for fname in real_imgs:
    path = os.path.join(REAL_DIR, fname)
    cls, conf = predict(path)
    tag = "PASS" if conf >= CONF_THRESH and cls != "background_other" else "REJECT"
    print(f"  [{tag:6s}] {fname:28s} -> {cls:30s} ({conf:5.1f}%)")

print("=" * 75)
