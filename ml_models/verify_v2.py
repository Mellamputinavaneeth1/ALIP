import os, json, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v2.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

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
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        prob = torch.softmax(out[0], 0)
        conf, idx = torch.max(prob, 0)
    return CLASS_NAMES[idx.item()], conf.item()

PLANTDOC_MAP = {
    "Tomato Early blight leaf":   "tomato_early_blight",
    "Tomato leaf":                "tomato_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato leaf late blight":    "tomato_late_blight",
    "Tomato mold leaf":           "tomato_leaf_mold",
    "Tomato leaf mosaic virus":   None,
    "Tomato leaf yellow virus":   None,
    "Tomato Septoria leaf spot":  None,
}

root = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
total = correct = low_conf = 0
per_class = {}

print("=== tomato_v2.pt vs PlantDoc REAL FIELD images ===\n")
for folder, true_cls in PLANTDOC_MAP.items():
    path = os.path.join(root, folder)
    if not os.path.exists(path):
        continue
    imgs = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if true_cls is None:
        rej = sum(1 for f in imgs if predict(os.path.join(path, f))[1] < 0.60)
        print(f"  [OUT-OF-SCOPE] {folder:42s} {len(imgs):2d} imgs | Rejected(<60%): {rej}/{len(imgs)}")
        continue
    if true_cls not in per_class:
        per_class[true_cls] = {"c": 0, "n": 0, "lc": 0}
    for f in imgs:
        pred, conf = predict(os.path.join(path, f))
        total += 1
        per_class[true_cls]["n"] += 1
        if conf < 0.60:
            low_conf += 1
            per_class[true_cls]["lc"] += 1
        if pred == true_cls:
            correct += 1
            per_class[true_cls]["c"] += 1

print()
for cls, s in per_class.items():
    acc = s["c"] / s["n"] * 100 if s["n"] > 0 else 0
    tag = "PASS" if acc >= 70 else "FAIL"
    c = s["c"]; n = s["n"]; lc = s["lc"]
    print(f"  [{tag}] {cls:35s} {c}/{n} = {acc:.1f}%  (low-conf: {lc})")

overall = correct / total * 100 if total > 0 else 0
print(f"\n  OVERALL (in-scope) : {correct}/{total} = {overall:.1f}%")
print(f"  Low-conf rejections: {low_conf}/{total} = {low_conf/total*100:.1f}%")
print(f"\n  v1 PlantDoc accuracy : 40.5%")
print(f"  v2 PlantDoc accuracy : {overall:.1f}%")
improvement = overall - 40.5
print(f"  Improvement          : +{improvement:.1f} percentage points")
