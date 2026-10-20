"""
ALIP - Multi-Crop Disease Classifier v1 Evaluation Script
Evaluates multicrop_model_v1.pt on the multicrop_dataset/test split.
Generates per-class recall, weighted F1, confidence distribution,
and saves the confusion matrix heatmap to multicrop_v1_confusion_matrix.png.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR    = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_dataset"
MODEL_PATH  = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v1.pt"
CLASSES_PATH= r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_class_names.json"
CM_OUT_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v1_confusion_matrix.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[EVAL] Device: {DEVICE}")

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
print(f"[EVAL] Loaded {NUM_CLASSES} classes: {CLASS_NAMES}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_transform)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"[EVAL] Test set images: {len(test_ds)}")

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
print(f"[EVAL] Model loaded successfully from {MODEL_PATH}")

all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for imgs, labs in test_dl:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labs.numpy())
        all_probs.extend(conf.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT (Test Split)")
print("=" * 70)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

wf1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\nWeighted F1 Score: {wf1:.4f}  ({'PASS' if wf1 >= 0.82 else 'FAIL'} - Target: >= 0.82)")

print("\n" + "=" * 70)
print("PER-CLASS RECALL EVALUATION")
print("=" * 70)
all_recall_passed = True
for c in range(NUM_CLASSES):
    cls_name = CLASS_NAMES[c]
    mask = all_labels == c
    total_samples = mask.sum()
    recall = (all_preds[mask] == c).sum() / total_samples if total_samples > 0 else 0
    thresh = 0.65 if cls_name == "potato_healthy" else 0.75
    passed = recall >= thresh
    if not passed:
        all_recall_passed = False
    status = "PASS" if passed else "FAIL"
    print(f"  {cls_name:32s} (N={total_samples:3d}): Recall = {recall*100:5.1f}%  (Target >= {thresh*100:.0f}%) -> {status}")

print(f"\nAll Per-Class Recall Passed: {'YES' if all_recall_passed else 'NO'}")

# Confidence Distribution
n = len(all_probs)
high = (all_probs >= 0.80).sum()
low  = ((all_probs >= 0.60) & (all_probs < 0.80)).sum()
rej  = (all_probs < 0.60).sum()
print("\n" + "=" * 70)
print("CONFIDENCE DISTRIBUTION (Test Set)")
print("=" * 70)
print(f"  High Confidence (>= 80%): {high:4d} / {n} ({high/n*100:5.1f}%)")
print(f"  Low Confidence  (60-80%): {low:4d} / {n} ({low/n*100:5.1f}%)")
print(f"  Rejected        (< 60%) : {rej:4d} / {n} ({rej/n*100:5.1f}%)")

# Confusion Matrix Plot
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Multi-Crop Disease Classifier v1 - Confusion Matrix (Test Split)", fontsize=14)
plt.ylabel("True Class", fontsize=12)
plt.xlabel("Predicted Class", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(CM_OUT_PATH, dpi=150)
plt.close()
print(f"\nConfusion matrix plot saved to: {CM_OUT_PATH}")
