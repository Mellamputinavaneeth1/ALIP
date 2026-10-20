"""
ALIP - Real-World Verification (Fixed) + Mixed dataset retrain prep
"""

import os, json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v1.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def run_single_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        t   = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out  = model(t)
            prob = torch.softmax(out[0], dim=0)
            conf, idx = torch.max(prob, 0)
        return CLASS_NAMES[idx.item()], conf.item()
    except Exception as e:
        return None, 0.0

def evaluate_folder_set(name, folder_map, root):
    print(f"\n{'='*60}")
    print(f"SOURCE: {name}")
    print(f"{'='*60}")
    total = correct = out_of_scope_total = 0
    low_conf = 0
    per_class = {}

    for folder_name, true_class in folder_map.items():
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            print(f"  [SKIP] {folder_name} not found")
            continue
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not images:
            continue

        if true_class is None:
            rejected = sum(1 for f in images
                           if run_single_image(os.path.join(folder_path,f))[1] < 0.60)
            print(f"  [OUT-OF-SCOPE] {folder_name:42s} {len(images):3d} images | Rejected: {rejected}/{len(images)}")
            out_of_scope_total += len(images)
            continue

        if true_class not in per_class:
            per_class[true_class] = {"correct": 0, "total": 0, "low_conf": 0}

        for img_file in images:
            pred_class, confidence = run_single_image(os.path.join(folder_path, img_file))
            if pred_class is None:
                continue
            total += 1
            per_class[true_class]["total"] += 1
            if confidence < 0.60:
                low_conf += 1
                per_class[true_class]["low_conf"] += 1
            if pred_class == true_class:
                correct += 1
                per_class[true_class]["correct"] += 1

    print(f"\n  Per-class results:")
    for cls, stats in per_class.items():
        n   = stats["total"]
        c   = stats["correct"]
        lc  = stats["low_conf"]
        acc = c / n * 100 if n > 0 else 0
        tag = "PASS" if acc >= 70 else "FAIL"
        print(f"  [{tag}] {cls:35s}  {c:3d}/{n:3d} = {acc:5.1f}%  (low-conf rejected: {lc})")

    if total > 0:
        overall = correct / total * 100
        print(f"\n  OVERALL: {correct}/{total} = {overall:.1f}%")
        print(f"  Low-conf rejections: {low_conf}/{total} = {low_conf/total*100:.1f}%")
    else:
        overall = 0.0
        print(f"\n  No matching images found.")
    if out_of_scope_total:
        print(f"  Out-of-scope images: {out_of_scope_total}")
    return overall

# ── 1. PlantVillage test split ──────────────────────────────
pv_test_ds = datasets.ImageFolder(
    r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_dataset\test", transform)
pv_dl = DataLoader(pv_test_ds, batch_size=64, shuffle=False, num_workers=0)
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labs in pv_dl:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        _, p = torch.max(out, 1)
        all_preds.extend(p.cpu().numpy())
        all_labels.extend(labs.numpy())
pv_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
print(f"\n{'='*60}")
print(f"SOURCE: PlantVillage Test Split (same distribution - lab images)")
print(f"{'='*60}")
print(f"  Accuracy: {pv_acc:.2f}% on {len(all_labels)} images")

# ── 2. PlantDoc real field images ───────────────────────────
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
acc2 = evaluate_folder_set(
    "PlantDoc (REAL FIELD - completely different distribution)",
    PLANTDOC_MAP,
    r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
)

# ── 3. plantdoc_field (correct folder names) ────────────────
FIELD_MAP = {
    "Tomato_bacterial_spot": "tomato_bacterial_spot",
    "Tomato_early_blight":   "tomato_early_blight",
    "Tomato_healthy":        "tomato_healthy",
    "Tomato_late_blight":    "tomato_late_blight",
}
acc3 = evaluate_folder_set(
    "plantdoc_field (additional real field images)",
    FIELD_MAP,
    r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field"
)

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FINAL VERDICT")
print(f"{'='*60}")
print(f"  PlantVillage lab images (train distribution) : {pv_acc:.1f}%")
print(f"  PlantDoc real field images (unseen sources)  : {acc2:.1f}%")
print(f"  plantdoc_field                               : {acc3:.1f}%")
gap = pv_acc - acc2
print(f"\n  Lab->Field accuracy DROP: {gap:.1f} percentage points")
if gap > 30:
    print("  CONCLUSION: Significant overfitting to PlantVillage lab images.")
    print("  SOLUTION: Retrain with mixed PlantVillage + PlantDoc/field images.")
elif gap > 15:
    print("  CONCLUSION: Moderate domain gap. Needs field image augmentation.")
else:
    print("  CONCLUSION: Good generalization. Model is field-ready.")
