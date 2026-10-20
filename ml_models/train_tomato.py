"""
ALIP - Tomato Disease Classifier (Phase 1)
Architecture : EfficientNet-B0 (Transfer Learning from ImageNet)
Classes      : 5 Tomato disease classes
GPU          : NVIDIA RTX 3050 (CUDA)
"""

import os, json, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.cuda.amp import GradScaler, autocast   # AMP
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONFIG = {
    "data_dir":          r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_dataset",
    "model_out":         r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v1.pt",
    "classes_out":       r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json",
    "img_size":          224,
    "batch_size":        32,   # Phase A (classifier only)
    "batch_size_b":      16,   # Phase B (full fine-tune — smaller to fit 6GB VRAM)
    "epochs":            25,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "patience":          6,
    "num_workers":       0,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp":           True, # Automatic Mixed Precision (halves VRAM usage)
}

print(f"[GPU] Using: {CONFIG['device']}")
if CONFIG["device"] == "cuda":
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")

# â”€â”€â”€ Transforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),   # hide small patches
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# â”€â”€â”€ Datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_ds = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "val"),   val_transform)
test_ds  = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "test"),  val_transform)

CLASS_NAMES = train_ds.classes
NUM_CLASSES  = len(CLASS_NAMES)
print(f"[DATA] Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"[DATA] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# Save class names
with open(CONFIG["classes_out"], "w") as f:
    json.dump(CLASS_NAMES, f)
print(f"[OK] Class names saved â†’ {CONFIG['classes_out']}")

# â”€â”€â”€ Weighted Sampler (handles class imbalance) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class_counts = np.array([
    sum(1 for _, lbl in train_ds if lbl == c) for c in range(NUM_CLASSES)
])
class_weights = 1.0 / class_counts
sample_weights = [class_weights[lbl] for _, lbl in train_ds]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# â”€â”€â”€ Dataloaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                      sampler=sampler, num_workers=CONFIG["num_workers"],
                      pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                      shuffle=False, num_workers=CONFIG["num_workers"])
test_dl  = DataLoader(test_ds, batch_size=CONFIG["batch_size"],
                      shuffle=False, num_workers=CONFIG["num_workers"])

# â”€â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze all base layers first
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(CONFIG["device"])
print(f"[MODEL] EfficientNet-B0 loaded. Classifier: {NUM_CLASSES} outputs.")

# â”€â”€â”€ Loss, Optimizer, Scheduler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Phase A: train only classifier head (epochs 1-5)
optimizer_head = optim.AdamW(
    model.classifier.parameters(),
    lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
)

# Phase B: fine-tune full model (epochs 6+)
# (we unfreeze all layers after warm-up)

scheduler_head = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_head, T_max=5, eta_min=1e-5
)

# --- AMP Scaler -----------------------------------------------------------
scaler = GradScaler(enabled=CONFIG["use_amp"])

# --- Training Helpers -------------------------------------------------------
def run_epoch(loader, train=True, optimizer=None):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            if train:
                optimizer.zero_grad()
                with autocast(enabled=CONFIG["use_amp"]):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(enabled=CONFIG["use_amp"]):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

# --- Training Loop ----------------------------------------------------------
best_val_f1    = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience_count = 0
history        = {"train_loss": [], "val_acc": [], "val_f1": []}

# Phase B dataloader with smaller batch size to avoid OOM
phase_b_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size_b"],
                        sampler=sampler, num_workers=CONFIG["num_workers"],
                        pin_memory=True)

print("\n" + "="*60)
print("PHASE A: Warm-up (Classifier Only) - Epochs 1-5")
print("="*60)

for epoch in range(1, CONFIG["epochs"] + 1):

    # -- Switch to full fine-tune after warm-up --
    if epoch == 6:
        print("\n" + "="*60)
        print("PHASE B: Full Fine-Tune (All Layers, AMP, batch=16) - Epochs 6+")
        print("="*60)
        torch.cuda.empty_cache()   # free any leftover VRAM before unfreezing
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"] / 10,
            weight_decay=CONFIG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"] - 5, eta_min=1e-6
        )
        active_dl = phase_b_dl   # use smaller batch
    elif epoch == 1:
        active_dl = train_dl     # full batch for Phase A
    if epoch < 6:
        optimizer = optimizer_head
        scheduler = scheduler_head
        active_dl = train_dl

    t0 = time.time()
    train_loss, train_acc = run_epoch(active_dl, train=True,  optimizer=optimizer)
    val_loss,   val_acc   = run_epoch(val_dl,    train=False)

    # Compute weighted F1 on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(CONFIG["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    val_f1 = f1_score(all_labels, all_preds, average="weighted")

    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
          f"Loss: {train_loss:.4f} | "
          f"TrainAcc: {train_acc*100:.1f}% | "
          f"ValAcc: {val_acc*100:.1f}% | "
          f"ValF1: {val_f1:.4f} | "
          f"Time: {elapsed:.0f}s")

    # Save best model (by weighted F1, not just accuracy)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, CONFIG["model_out"])
        print(f"  âœ… Best model saved  (F1={val_f1:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= CONFIG["patience"]:
            print(f"\n[EARLY STOP] No improvement for {CONFIG['patience']} epochs.")
            break

# â”€â”€â”€ Final Evaluation on Test Set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)
model.load_state_dict(best_model_wts)
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_dl:
        images = images.to(CONFIG["device"])
        outputs = model(images)
        probs   = torch.softmax(outputs, 1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.max(1).values.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES, digits=4))

# PASS / FAIL check
wf1 = f1_score(all_labels, all_preds, average="weighted")
per_class_recall = []
for c in range(NUM_CLASSES):
    mask   = all_labels == c
    recall = (all_preds[mask] == c).sum() / mask.sum()
    per_class_recall.append(recall)
    print(f"  Recall [{CLASS_NAMES[c]}]: {recall*100:.1f}%")

print(f"\n  Weighted F1: {wf1:.4f}  {'âœ… PASS' if wf1>=0.90 else 'âŒ FAIL (need â‰¥ 0.90)'}")
print(f"  Min class recall: {min(per_class_recall)*100:.1f}%  "
      f"{'âœ… PASS' if min(per_class_recall)>=0.85 else 'âŒ FAIL (need â‰¥ 0.85)'}")

# Confidence distribution
high_conf = (all_probs >= 0.80).sum()
low_conf  = ((all_probs >= 0.60) & (all_probs < 0.80)).sum()
rejected  = (all_probs < 0.60).sum()
total_test = len(all_probs)
print(f"\n  Confidence distribution:")
print(f"  High (â‰¥80%): {high_conf}/{total_test} = {high_conf/total_test*100:.1f}%")
print(f"  Low  (60-80%): {low_conf}/{total_test} = {low_conf/total_test*100:.1f}%")
print(f"  Reject (<60%): {rejected}/{total_test} = {rejected/total_test*100:.1f}%")

# â”€â”€â”€ Confusion Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix â€” Tomato Disease Classifier")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
cm_path = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
print(f"\n  Confusion matrix saved â†’ {cm_path}")

# â”€â”€â”€ Training Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"]); axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(history["val_f1"],  label="Val F1");  axes[1].set_title("Validation F1"); axes[1].set_xlabel("Epoch")
plt.tight_layout()
curve_path = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_training_curves.png"
plt.savefig(curve_path, dpi=150)
print(f"  Training curves saved â†’ {curve_path}")

print(f"\n[DONE] Best model: {CONFIG['model_out']}")
print(f"[DONE] Best Val F1: {best_val_f1:.4f}")
