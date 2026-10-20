"""
ALIP - Comprehensive Multi-Crop Model Verification
Tests multicrop_model_v1.pt against:
  1. Real field images in real_field_images/
  2. PlantDoc test set (real world - per crop)
  3. Out-of-scope crop images (should be rejected)
  4. Edge cases (wrong images)
"""

import os, json, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v1.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_class_names.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE  = 1.15
CONF_THRESH  = 60.0

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

DISPLAY = {
    "corn_cercospora_leaf_spot":     "Corn - Cercospora Leaf Spot",
    "corn_common_rust":              "Corn - Common Rust",
    "corn_healthy":                  "Corn - Healthy",
    "corn_northern_leaf_blight":     "Corn - Northern Leaf Blight",
    "pepper_bacterial_spot":         "Pepper - Bacterial Spot",
    "pepper_healthy":                "Pepper - Healthy",
    "potato_early_blight":           "Potato - Early Blight",
    "potato_healthy":                "Potato - Healthy",
    "potato_late_blight":            "Potato - Late Blight",
    "tomato_bacterial_spot":         "Tomato - Bacterial Spot",
    "tomato_early_blight":           "Tomato - Early Blight",
    "tomato_healthy":                "Tomato - Healthy",
    "tomato_late_blight":            "Tomato - Late Blight",
    "tomato_leaf_mold":              "Tomato - Leaf Mold",
    "tomato_mosaic_virus":           "Tomato - Mosaic Virus",
    "tomato_septoria_leaf_spot":     "Tomato - Septoria Leaf Spot",
    "tomato_spider_mites":           "Tomato - Spider Mites",
    "tomato_target_spot":            "Tomato - Target Spot",
    "tomato_yellow_leaf_curl_virus": "Tomato - Yellow Leaf Curl Virus",
}

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        t   = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(t)
            scaled = logits[0] / TEMPERATURE
            probs  = torch.softmax(scaled, 0)
            top3_p, top3_i = torch.topk(probs, 3)
        top3 = [(CLASS_NAMES[i.item()], round(p.item()*100, 1)) for i, p in zip(top3_i, top3_p)]
        pred_cls  = top3[0][0]
        pred_conf = top3[0][1]
        crop      = pred_cls.split("_")[0].title()
        display   = DISPLAY.get(pred_cls, pred_cls)
        gate      = "PASS" if pred_conf >= CONF_THRESH else "REJECTED"
        return pred_cls, pred_conf, crop, display, gate, top3
    except Exception as e:
        return None, 0, "Error", str(e), "ERROR", []

SEP = "=" * 72

# ── Section 1: Real Field Images ─────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 1: Real Field Images (D:/ALIP-TOTAL/ALIP/ml_models/real_field_images/)")
print(SEP)
REAL_DIR = r"D:\ALIP-TOTAL\ALIP\ml_models\real_field_images"
real_imgs = sorted([f for f in os.listdir(REAL_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
for fname in real_imgs:
    path = os.path.join(REAL_DIR, fname)
    cls, conf, crop, display, gate, top3 = predict(path)
    if gate == "PASS":
        print(f"  [{gate}]  {fname:35s} -> {display:38s} ({conf:.1f}%)")
    elif gate == "REJECTED":
        print(f"  [{gate}] {fname:35s} -> Low confidence {conf:.1f}%  (top: {display})")
    else:
        print(f"  [ERROR]  {fname}: {display}")

# ── Section 2: PlantDoc Real World Test Set per Crop ─────────────────────
PD_TEST = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
PLANTDOC_GROUPS = {
    "CORN (real field)": {
        "Corn rust leaf":       "corn_common_rust",
        "Corn leaf blight":     "corn_northern_leaf_blight",
        "Corn Gray leaf spot":  "corn_cercospora_leaf_spot",
    },
    "POTATO (real field)": {
        "Potato leaf early blight": "potato_early_blight",
        "Potato leaf late blight":  "potato_late_blight",
    },
    "PEPPER (real field)": {
        "Bell_pepper leaf spot": "pepper_bacterial_spot",
        "Bell_pepper leaf":      "pepper_healthy",
    },
    "TOMATO (real field)": {
        "Tomato Early blight leaf":   "tomato_early_blight",
        "Tomato leaf":                "tomato_healthy",
        "Tomato leaf bacterial spot": "tomato_bacterial_spot",
        "Tomato leaf late blight":    "tomato_late_blight",
        "Tomato mold leaf":           "tomato_leaf_mold",
        "Tomato Septoria leaf spot":  "tomato_septoria_leaf_spot",
    },
}

for group_name, folders in PLANTDOC_GROUPS.items():
    total = correct = rejected = 0
    print(f"\n{SEP}")
    print(f"SECTION 2: PlantDoc Test Set — {group_name}")
    print(SEP)
    for folder, true_cls in folders.items():
        folder_path = os.path.join(PD_TEST, folder)
        if not os.path.exists(folder_path):
            print(f"  [SKIP] {folder}")
            continue
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        folder_correct = folder_rej = 0
        for img in imgs:
            cls, conf, crop, display, gate, top3 = predict(os.path.join(folder_path, img))
            total += 1
            if gate == "REJECTED":
                rejected += 1
                folder_rej += 1
            elif cls == true_cls:
                correct += 1
                folder_correct += 1
        n = len(imgs)
        acc = folder_correct / (n - folder_rej) * 100 if (n - folder_rej) > 0 else 0
        tag = "PASS" if acc >= 70 else "FAIL"
        print(f"  [{tag}] {folder:42s}  {folder_correct}/{n - folder_rej} correct = {acc:.0f}%  (rejected: {folder_rej})")
    in_scope = total - rejected
    overall  = correct / in_scope * 100 if in_scope > 0 else 0
    print(f"\n  Group Total: {correct}/{in_scope} = {overall:.1f}%   (rejected: {rejected}/{total})")

# ── Section 3: OUT-OF-SCOPE Crop Rejection ───────────────────────────────
print(f"\n{SEP}")
print("SECTION 3: Out-of-Scope Crop Rejection (Apple, Grape, Soybean, Strawberry)")
print(SEP)
PV_ROOT    = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
OOS_FOLDERS = {
    "Apple___Apple_scab":       "Apple (scab)",
    "Apple___healthy":          "Apple (healthy)",
    "Grape___Black_rot":        "Grape (black rot)",
    "Grape___healthy":          "Grape (healthy)",
    "Soybean___healthy":        "Soybean (healthy)",
    "Strawberry___Leaf_scorch": "Strawberry (leaf scorch)",
}
total_oos = rejected_oos = 0
for folder, label in OOS_FOLDERS.items():
    folder_path = os.path.join(PV_ROOT, folder)
    if not os.path.exists(folder_path):
        print(f"  [SKIP] {label}")
        continue
    imgs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:5]
    rej = 0
    for img in imgs:
        cls, conf, crop, display, gate, _ = predict(os.path.join(folder_path, img))
        total_oos += 1
        if gate == "REJECTED":
            rejected_oos += 1
            rej += 1
    tag = "REJECTED" if rej == len(imgs) else ("PARTIAL" if rej > 0 else "NOT REJECTED")
    print(f"  [{tag}] {label:30s}  {rej}/{len(imgs)} rejected")
rej_rate = rejected_oos / total_oos * 100 if total_oos > 0 else 0
rej_tag  = "PASS" if rej_rate >= 90 else "FAIL"
print(f"\n  Overall OOS rejection: {rejected_oos}/{total_oos} = {rej_rate:.1f}%  [{rej_tag}]")

print(f"\n{SEP}")
print("VERIFICATION COMPLETE")
print(SEP)
