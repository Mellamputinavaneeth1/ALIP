"""
ALIP - Multi-Crop Disease Classifier v1 (Unified 19-Class Model)
Trains on combined PlantVillage + PlantDoc dataset for Tomato, Potato, Corn, Pepper.
Phase A: Warm-up classifier head (5 epochs, batch 32, lr 1e-3)
Phase B: Full fine-tune with AMP (up to 30 epochs, batch 16, lr 1e-4)
"""

import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    "data_dir":     r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_dataset",
    "model_out":    r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v1.pt",
    "classes_out":  r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_class_names.json",
    "curves_out":   r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_training_curves.png",
    "batch_size_a": 32,
    "batch_size_b": 16,
    "epochs":       35,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     8,
    "num_workers":  0,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"[GPU] {CONFIG['device']} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Strong augmentation simulating field conditions
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "val"),   val_transform)
test_ds  = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "test"),  val_transform)

CLASS_NAMES = train_ds.classes
NUM_CLASSES  = len(CLASS_NAMES)
print(f"[DATA] Classes ({NUM_CLASSES}): {CLASS_NAMES}", flush=True)
print(f"[DATA] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}", flush=True)

# Save class names JSON
with open(CONFIG["classes_out"], "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)
print(f"[DATA] Saved class names to {CONFIG['classes_out']}", flush=True)

# Weighted sampler for class imbalance
class_counts   = np.array([sum(1 for _, l in train_ds.samples if l == c) for c in range(NUM_CLASSES)])
print(f"[DATA] Class counts in train: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)

class_weights  = 1.0 / np.maximum(class_counts, 1)
sample_weights = [class_weights[l] for _, l in train_ds.samples]
sampler_a      = WeightedRandomSampler(sample_weights, len(sample_weights))
sampler_b      = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dl_a = DataLoader(train_ds, batch_size=CONFIG["batch_size_a"],
                        sampler=sampler_a, num_workers=0, pin_memory=True)
train_dl_b = DataLoader(train_ds, batch_size=CONFIG["batch_size_b"],
                        sampler=sampler_b, num_workers=0, pin_memory=True)
val_dl     = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_dl    = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Model architecture
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(CONFIG["device"])
print(f"[MODEL] EfficientNet-B0 initialized with {NUM_CLASSES} outputs (dropout p=0.4).", flush=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
use_cuda  = CONFIG["device"] == "cuda"
scaler    = GradScaler("cuda", enabled=use_cuda)

optimizer_head = optim.AdamW(model.classifier.parameters(),
                             lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler_head = optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=5, eta_min=1e-5)

def train_one_epoch(loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(CONFIG["device"])
        labels = labels.to(CONFIG["device"])
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_cuda):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total

def evaluate_val(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            with autocast("cuda", enabled=use_cuda):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    val_loss = total_loss / total
    val_acc  = correct / total
    val_f1   = f1_score(labels_list, preds_list, average="weighted")
    return val_loss, val_acc, val_f1

best_val_f1    = 0.0
best_wts       = copy.deepcopy(model.state_dict())
patience_count = 0
history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
optimizer      = optimizer_head
scheduler      = scheduler_head
active_dl      = train_dl_a

print("\n" + "=" * 60, flush=True)
print("PHASE A: Warm-up (Classifier Only) - Epochs 1-5", flush=True)
print("=" * 60, flush=True)

for epoch in range(1, CONFIG["epochs"] + 1):
    if epoch == 6:
        print("\n" + "=" * 60, flush=True)
        print("PHASE B: Full Fine-Tune (AMP, batch=16) - Epochs 6+", flush=True)
        print("=" * 60, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(),
                                lr=CONFIG["lr"] / 10,
                                weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"] - 5, eta_min=1e-6)
        active_dl = train_dl_b

    t0 = time.time()
    train_loss, train_acc = train_one_epoch(active_dl, optimizer=optimizer)
    val_loss, val_acc, val_f1 = evaluate_val(val_dl)
    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
          f"Loss:{train_loss:.4f} | TrainAcc:{train_acc*100:.1f}% | "
          f"ValLoss:{val_loss:.4f} | ValAcc:{val_acc*100:.1f}% | ValF1:{val_f1:.4f} | {elapsed:.0f}s", flush=True)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_wts    = copy.deepcopy(model.state_dict())
        torch.save(best_wts, CONFIG["model_out"])
        print(f"  --> Best model saved (Val F1={val_f1:.4f})", flush=True)
        patience_count = 0
    else:
        patience_count += 1
        print(f"  (patience {patience_count}/{CONFIG['patience']})", flush=True)
        if patience_count >= CONFIG["patience"]:
            print(f"\n[EARLY STOP] No improvement for {CONFIG['patience']} epochs.", flush=True)
            break

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.plot(history["val_f1"], label="Val F1 (Weighted)")
plt.title("Accuracy and F1 Curves")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.savefig(CONFIG["curves_out"], dpi=150)
plt.close()
print(f"[CURVES] Saved training curves to {CONFIG['curves_out']}", flush=True)

# Final evaluation on Test Split
print("\n" + "=" * 60, flush=True)
print("FINAL TEST SET EVALUATION", flush=True)
print("=" * 60, flush=True)
model.load_state_dict(best_wts)
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labs in test_dl:
        imgs = imgs.to(CONFIG["device"])
        with autocast("cuda", enabled=use_cuda):
            out = model(imgs)
        prob = torch.softmax(out, 1)
        _, p = torch.max(prob, 1)
        all_preds.extend(p.cpu().numpy())
        all_labels.extend(labs.numpy())
        all_probs.extend(prob.max(1).values.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print("\nClassification Report (Test Set):", flush=True)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4), flush=True)
wf1 = f1_score(all_labels, all_preds, average="weighted")

print("Per-class recall:", flush=True)
for c in range(NUM_CLASSES):
    mask = all_labels == c
    recall = (all_preds[mask] == c).sum() / mask.sum() if mask.sum() > 0 else 0
    thresh = 0.65 if CLASS_NAMES[c] == "potato_healthy" else 0.75
    pass_str = "PASS" if recall >= thresh else "FAIL"
    print(f"  {CLASS_NAMES[c]:32s}: {recall*100:5.1f}%  (threshold: {thresh*100:.0f}%) -> {pass_str}", flush=True)

high = (all_probs >= 0.80).sum()
low  = ((all_probs >= 0.60) & (all_probs < 0.80)).sum()
rej  = (all_probs < 0.60).sum()
n    = len(all_probs)
print(f"\nWeighted F1 : {wf1:.4f}  ({'PASS' if wf1>=0.82 else 'FAIL'}, threshold >= 0.82)", flush=True)
print(f"Confidence distribution:", flush=True)
print(f"  High (>=80%): {high}/{n} = {high/n*100:.1f}%", flush=True)
print(f"  Low  (60-80%): {low}/{n} = {low/n*100:.1f}%", flush=True)
print(f"  Reject (<60%): {rej}/{n} = {rej/n*100:.1f}%", flush=True)

print(f"\nTraining complete. Model saved -> {CONFIG['model_out']}", flush=True)
print(f"Best Val F1: {best_val_f1:.4f}", flush=True)
