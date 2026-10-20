"""
ALIP - evaluate_multicrop_v2.py
Evaluates multicrop_model_v2.pt on the 20-class test set.
Generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix heatmap
  - Overall accuracy and weighted F1
"""

import os, json, torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v2.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_class_names.json"
DATA_DIR     = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_dataset\test"
OUTPUT_CM    = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_confusion_matrix.png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(DATA_DIR, test_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

model = efficientnet_b0(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_f, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc = accuracy_score(all_targets, all_preds) * 100
f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

print("=" * 75)
print(f"MULTICROP MODEL V2 EVALUATION (20 Classes)")
print(f"Overall Test Accuracy: {acc:.2f}% | Weighted F1: {f1:.4f}")
print("=" * 75)

report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4)
print(report)

# Confusion Matrix Plot
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Multi-Crop V2 Confusion Matrix (Acc: {acc:.2f}%, F1: {f1:.4f})", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=150)
print(f"\nConfusion matrix saved to: {OUTPUT_CM}")
