"""
ALIP - Tomato Disease Classifier v2 (Mixed Dataset)
Trains on PlantVillage + PlantDoc combined dataset for real-world robustness.
Uses stronger augmentation to simulate field conditions.
"""

import os, json, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, f1_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    "data_dir":     r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_mixed_dataset",
    "model_out":    r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v2.pt",
    "classes_out":  r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json",
    "batch_size_a": 32,
    "batch_size_b": 16,
    "epochs":       30,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     7,
    "num_workers":  0,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"[GPU] {CONFIG['device']} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Stronger augmentation to simulate field conditions (blurry, rotated, partial, shadowed)
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
print(f"[DATA] Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"[DATA] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

with open(CONFIG["classes_out"], "w") as f:
    json.dump(CLASS_NAMES, f)

# Weighted sampler for class imbalance
class_counts   = np.array([sum(1 for _,l in train_ds if l==c) for c in range(NUM_CLASSES)])
class_weights  = 1.0 / class_counts
sample_weights = [class_weights[l] for _,l in train_ds]
sampler_a      = WeightedRandomSampler(sample_weights, len(sample_weights))
sampler_b      = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dl_a = DataLoader(train_ds, batch_size=CONFIG["batch_size_a"],
                        sampler=sampler_a, num_workers=0, pin_memory=True)
train_dl_b = DataLoader(train_ds, batch_size=CONFIG["batch_size_b"],
                        sampler=sampler_b, num_workers=0, pin_memory=True)
val_dl     = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_dl    = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(CONFIG["device"])
print(f"[MODEL] EfficientNet-B0 ready with {NUM_CLASSES} outputs.")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler    = GradScaler("cuda", enabled=True)

optimizer_head = optim.AdamW(model.classifier.parameters(),
                             lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler_head = optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=5, eta_min=1e-5)

def run_epoch(loader, train=True, optimizer=None):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            if train:
                optimizer.zero_grad()
                with autocast("cuda", enabled=True):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast("cuda", enabled=True):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
    return total_loss / total, correct / total

best_val_f1    = 0.0
best_wts       = copy.deepcopy(model.state_dict())
patience_count = 0
history        = {"train_loss": [], "val_f1": []}
optimizer      = optimizer_head
scheduler      = scheduler_head
active_dl      = train_dl_a

print("\n" + "="*60)
print("PHASE A: Warm-up (Classifier Only) - Epochs 1-5")
print("="*60)

for epoch in range(1, CONFIG["epochs"]+1):
    if epoch == 6:
        print("\n" + "="*60)
        print("PHASE B: Full Fine-Tune (AMP, batch=16) - Epochs 6+")
        print("="*60)
        torch.cuda.empty_cache()
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(),
                                lr=CONFIG["lr"]/10,
                                weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"]-5, eta_min=1e-6)
        active_dl = train_dl_b

    t0 = time.time()
    train_loss, train_acc = run_epoch(active_dl, train=True, optimizer=optimizer)
    _,          val_acc   = run_epoch(val_dl,    train=False)

    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labs in val_dl:
            imgs = imgs.to(CONFIG["device"])
            with autocast("cuda"):
                out = model(imgs)
            _, p = torch.max(out, 1)
            preds_list.extend(p.cpu().numpy())
            labels_list.extend(labs.numpy())
    val_f1 = f1_score(labels_list, preds_list, average="weighted")
    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_f1"].append(val_f1)

    print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
          f"Loss:{train_loss:.4f} | TrainAcc:{train_acc*100:.1f}% | "
          f"ValAcc:{val_acc*100:.1f}% | ValF1:{val_f1:.4f} | {elapsed:.0f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_wts    = copy.deepcopy(model.state_dict())
        torch.save(best_wts, CONFIG["model_out"])
        print(f"  Best model saved (F1={val_f1:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= CONFIG["patience"]:
            print(f"\n[EARLY STOP] No improvement for {CONFIG['patience']} epochs.")
            break

# Final evaluation
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)
model.load_state_dict(best_wts)
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labs in test_dl:
        imgs  = imgs.to(CONFIG["device"])
        with autocast("cuda"):
            out = model(imgs)
        prob  = torch.softmax(out, 1)
        _, p  = torch.max(prob, 1)
        all_preds.extend(p.cpu().numpy())
        all_labels.extend(labs.numpy())
        all_probs.extend(prob.max(1).values.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
wf1 = f1_score(all_labels, all_preds, average="weighted")

print("Per-class recall:")
for c in range(NUM_CLASSES):
    mask   = all_labels == c
    recall = (all_preds[mask] == c).sum() / mask.sum() if mask.sum() > 0 else 0
    print(f"  {CLASS_NAMES[c]:35s}: {recall*100:.1f}%  {'PASS' if recall>=0.85 else 'FAIL'}")

high = (all_probs >= 0.80).sum()
low  = ((all_probs >= 0.60) & (all_probs < 0.80)).sum()
rej  = (all_probs < 0.60).sum()
n    = len(all_probs)
print(f"\nWeighted F1 : {wf1:.4f}  {'PASS' if wf1>=0.85 else 'FAIL'}")
print(f"Confidence distribution:")
print(f"  High (>=80%): {high}/{n} = {high/n*100:.1f}%")
print(f"  Low  (60-80%): {low}/{n} = {low/n*100:.1f}%")
print(f"  Reject (<60%): {rej}/{n} = {rej/n*100:.1f}%")

# Save confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix - tomato_v2 (Mixed Dataset)")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v2_confusion_matrix.png", dpi=150)
print(f"\nDone. Model saved -> {CONFIG['model_out']}")
print(f"Best Val F1: {best_val_f1:.4f}")
