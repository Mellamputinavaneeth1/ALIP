"""
ALIP - Tomato Disease Dataset Preparation
Splits PlantVillage tomato images into train / val / test (70/15/15)
using deterministic shuffling to avoid data leakage.
"""

import os
import shutil
import random
import json

random.seed(42)

# ─── Source folders (PlantVillage color images) ─────────────────────────────
SRC_ROOT = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
DST_ROOT = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_dataset"

CLASS_MAP = {
    "Tomato___healthy":           "tomato_healthy",
    "Tomato___Early_blight":      "tomato_early_blight",
    "Tomato___Late_blight":       "tomato_late_blight",
    "Tomato___Bacterial_spot":    "tomato_bacterial_spot",
    "Tomato___Leaf_Mold":         "tomato_leaf_mold",
}

# Cap per class to balance the dataset (max 1500 images per class)
MAX_PER_CLASS = 1500

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

stats = {}

for src_folder, dst_class in CLASS_MAP.items():
    src_dir = os.path.join(SRC_ROOT, src_folder)
    if not os.path.exists(src_dir):
        print(f"[MISSING] {src_dir}")
        continue

    # Get all image files
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    # Cap to MAX_PER_CLASS
    files = files[:MAX_PER_CLASS]
    n = len(files)

    # Calculate split sizes
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])
    n_test  = n - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:]
    }

    stats[dst_class] = {"total": n, "train": n_train, "val": n_val, "test": n_test}

    for split_name, split_files in splits.items():
        dst_dir = os.path.join(DST_ROOT, split_name, dst_class)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in split_files:
            shutil.copy2(
                os.path.join(src_dir, fname),
                os.path.join(dst_dir, fname)
            )

    print(f"[OK] {dst_class:35s} total={n:4d}  train={n_train:4d}  val={n_val:4d}  test={n_test:4d}")

# Save stats
with open(os.path.join(DST_ROOT, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("\n[DONE] Dataset split complete.")
print(f"Stats saved to {DST_ROOT}/dataset_stats.json")
