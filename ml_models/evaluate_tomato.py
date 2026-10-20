"""
Evaluate the saved tomato_v1.pt on the test set.
Reports: weighted F1, per-class recall, confusion matrix, confidence distribution.
"""
import os, json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH  = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v1.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json"
DATA_DIR    = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_dataset\test"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(DATA_DIR, val_transform)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Test set: {len(test_ds)} images | Classes: {CLASS_NAMES}")

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
print("Model loaded. Running evaluation...")

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_dl:
        images = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, 1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.max(1).values.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

wf1 = f1_score(all_labels, all_preds, average="weighted")
per_class_recall = []
print("PER-CLASS RECALL:")
for c in range(NUM_CLASSES):
    mask   = all_labels == c
    recall = (all_preds[mask] == c).sum() / mask.sum() if mask.sum() > 0 else 0
    per_class_recall.append(recall)
    status = "PASS" if recall >= 0.85 else "FAIL"
    print(f"  [{status}] {CLASS_NAMES[c]:30s}: {recall*100:.1f}%")

print(f"\nWeighted F1 : {wf1:.4f}  {'PASS' if wf1>=0.90 else 'FAIL'} (need >=0.90)")
print(f"Min recall  : {min(per_class_recall)*100:.1f}%  {'PASS' if min(per_class_recall)>=0.85 else 'FAIL'} (need >=0.85)")

high  = (all_probs >= 0.80).sum()
low   = ((all_probs >= 0.60) & (all_probs < 0.80)).sum()
rej   = (all_probs < 0.60).sum()
n     = len(all_probs)
print(f"\nConfidence distribution ({n} test images):")
print(f"  High  (>=80%): {high}/{n} = {high/n*100:.1f}%")
print(f"  Low  (60-80%): {low}/{n}  = {low/n*100:.1f}%")
print(f"  Reject (<60%): {rej}/{n}  = {rej/n*100:.1f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix - Tomato v1 (EfficientNet-B0)")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
out = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_confusion_matrix.png"
plt.savefig(out, dpi=150)
print(f"\nConfusion matrix saved -> {out}")
print("\nDONE.")
