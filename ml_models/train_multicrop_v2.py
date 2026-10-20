"""
ALIP - train_multicrop_v2.py
Trains EfficientNet-B0 on 20-class dataset (19 crops + background_other).
Improvements over v1:
  - 20th class: background_other (OOS rejection)
  - 3x sampling weight for potato_healthy and corn_cercospora_leaf_spot
  - MixUp augmentation for potato classes
  - Same Phase A/B schedule, AMP, early stopping
"""

import os, json, time, torch, torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":     r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_dataset",
    "model_out":    r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_model_v2.pt",
    "classes_out":  r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_class_names.json",
    "curve_out":    r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_training_curves.png",
    "batch_size_a": 32,
    "batch_size_b": 16,
    "epochs":       35,
    "lr_a":         1e-3,
    "lr_b":         1e-4,
    "weight_decay": 1e-4,
    "patience":     8,
    "num_workers":  0,   # Windows must be 0
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    # 3x boost for underrepresented classes
    "boosted_classes": ["potato_healthy", "corn_cercospora_leaf_spot"],
    "boost_factor":    3.0,
    # MixUp config (applied only to potato classes)
    "mixup_alpha":    0.2,
    "mixup_classes":  ["potato_healthy", "potato_early_blight", "potato_late_blight"],
}

# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── DATASET ─────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "val"),   val_transform)
CLASS_NAMES = train_ds.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# Save class names immediately
with open(CONFIG["classes_out"], "w") as f:
    json.dump(CLASS_NAMES, f)
print(f"Saved class names to {CONFIG['classes_out']}")

# ─── WEIGHTED SAMPLER (with boost) ───────────────────────────────────────────
targets = [label for _, label in train_ds.samples]
class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
class_weights = 1.0 / class_counts

# Apply boost factor to selected classes
boosted_idx = [CLASS_NAMES.index(c) for c in CONFIG["boosted_classes"] if c in CLASS_NAMES]
for idx in boosted_idx:
    class_weights[idx] *= CONFIG["boost_factor"]
    print(f"  [BOOST {CONFIG['boost_factor']}x] {CLASS_NAMES[idx]} "
          f"(count={int(class_counts[idx])})")

sample_weights = [class_weights[t] for t in targets]

def make_sampler():
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ─── MIXUP HELPER ────────────────────────────────────────────────────────────
mixup_class_idx = set(CLASS_NAMES.index(c) for c in CONFIG["mixup_classes"] if c in CLASS_NAMES)

def mixup_batch(images, labels, alpha=0.2):
    """Apply MixUp only if both images in the pair are potato classes."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    # Only apply MixUp when both are potato classes
    mask = torch.tensor(
        [(labels[i].item() in mixup_class_idx and labels[index[i]].item() in mixup_class_idx)
         for i in range(batch_size)],
        dtype=torch.float32, device=images.device
    )
    lam_t = lam * mask + 1.0 * (1.0 - mask)  # lam for potato pairs, 1.0 for others
    lam_t = lam_t.view(-1, 1, 1, 1)
    mixed = lam_t * images + (1 - lam_t) * images[index]
    labels_a = labels
    labels_b = labels[index]
    lam_scalar = lam_t.view(-1)
    return mixed, labels_a, labels_b, lam_scalar

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()

# ─── MODEL ───────────────────────────────────────────────────────────────────
device = CONFIG["device"]
print(f"\nDevice: {device}")
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler    = GradScaler("cuda", enabled=(device == "cuda"))

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc      = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1

def make_loader(dataset, batch_size, sampler=None, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=CONFIG["num_workers"], pin_memory=True
    )

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
best_val_f1    = 0.0
best_state     = None
patience_count = 0
history        = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

print(f"\n{'='*72}")
print(f"PHASE A — Classifier warm-up ({5} epochs, batch={CONFIG['batch_size_a']})")
print(f"{'='*72}")
optimizer_a   = torch.optim.AdamW(
    model.classifier.parameters(), lr=CONFIG["lr_a"], weight_decay=CONFIG["weight_decay"]
)
scheduler_a   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=5, eta_min=1e-5)
val_loader    = make_loader(val_ds, CONFIG["batch_size_a"])

for epoch in range(1, 6):
    t0      = time.time()
    loader  = make_loader(train_ds, CONFIG["batch_size_a"], sampler=make_sampler())
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_a.zero_grad()
        with autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer_a)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    scheduler_a.step()
    val_loss, val_acc, val_f1 = evaluate(val_loader)
    train_loss = total_loss / total
    train_acc  = correct / total * 100
    elapsed    = int(time.time() - t0)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    print(f"Epoch {epoch:2d}/35 | Loss:{train_loss:.4f} | TrainAcc:{train_acc:.1f}% | "
          f"ValLoss:{val_loss:.4f} | ValAcc:{val_acc:.1f}% | ValF1:{val_f1:.4f} | {elapsed}s")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, CONFIG["model_out"])
        print(f"  --> Best model saved (Val F1={best_val_f1:.4f})")

# ─── PHASE B: Full fine-tune ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"PHASE B — Full fine-tune (up to 30 epochs, batch={CONFIG['batch_size_b']})")
print(f"{'='*72}")
for param in model.parameters():
    param.requires_grad = True
torch.cuda.empty_cache()

optimizer_b  = torch.optim.AdamW(
    model.parameters(), lr=CONFIG["lr_b"], weight_decay=CONFIG["weight_decay"]
)
scheduler_b  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=30, eta_min=1e-6)
val_loader_b = make_loader(val_ds, CONFIG["batch_size_b"])

for epoch in range(6, CONFIG["epochs"] + 1):
    t0     = time.time()
    loader = make_loader(train_ds, CONFIG["batch_size_b"], sampler=make_sampler())
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_b.zero_grad()
        # Apply MixUp
        mixed, y_a, y_b, lam = mixup_batch(images, labels, CONFIG["mixup_alpha"])
        with autocast("cuda", enabled=(device == "cuda")):
            outputs = model(mixed)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer_b)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == y_a).sum().item()
        total      += images.size(0)
    scheduler_b.step()
    val_loss, val_acc, val_f1 = evaluate(val_loader_b)
    train_loss = total_loss / total
    train_acc  = correct / total * 100
    elapsed    = int(time.time() - t0)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    print(f"Epoch {epoch:2d}/35 | Loss:{train_loss:.4f} | TrainAcc:{train_acc:.1f}% | "
          f"ValLoss:{val_loss:.4f} | ValAcc:{val_acc:.1f}% | ValF1:{val_f1:.4f} | {elapsed}s")
    if val_f1 > best_val_f1:
        best_val_f1    = val_f1
        patience_count = 0
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, CONFIG["model_out"])
        print(f"  --> Best model saved (Val F1={best_val_f1:.4f})")
    else:
        patience_count += 1
        print(f"  (patience {patience_count}/{CONFIG['patience']})")
        if patience_count >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}. Best Val F1={best_val_f1:.4f}")
            break

# ─── SAVE TRAINING CURVES ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curves — multicrop v2"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_f1"],  label="Val F1",  color="green")
    axes[1].plot(history["val_acc"], label="Val Acc %", color="blue")
    axes[1].set_title("Val F1 & Accuracy — multicrop v2"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(CONFIG["curve_out"], dpi=150)
    print(f"\nTraining curves saved to {CONFIG['curve_out']}")
except Exception as e:
    print(f"Could not save curves: {e}")

print(f"\n[DONE] multicrop_model_v2.pt saved. Best Val F1={best_val_f1:.4f}")
