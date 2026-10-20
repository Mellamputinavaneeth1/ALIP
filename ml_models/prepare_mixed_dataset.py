"""
ALIP - Mixed Dataset Preparation
Combines PlantVillage (lab) + PlantDoc (real field) images into one
unified train/val/test dataset for domain-robust training.

Strategy:
 - PlantVillage: 70% of max 1200 per class (train), rest to val/test
 - PlantDoc train: goes into training set
 - PlantDoc test: reserved as real-world test set (never used in training)
"""

import os, shutil, random, json
random.seed(42)

PV_ROOT       = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
PD_TRAIN_ROOT = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\train"
PD_TEST_ROOT  = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
DST_ROOT      = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_mixed_dataset"

# Limit PlantVillage per class (capped so field images matter more relatively)
MAX_PV_PER_CLASS = 1200

CLASS_MAP = {
    # dest_class: { PlantVillage folder, PlantDoc train folder, PlantDoc test folder }
    "tomato_healthy": {
        "pv":      "Tomato___healthy",
        "pd_train":"Tomato leaf",
        "pd_test": "Tomato leaf",
    },
    "tomato_early_blight": {
        "pv":      "Tomato___Early_blight",
        "pd_train":"Tomato Early blight leaf",
        "pd_test": "Tomato Early blight leaf",
    },
    "tomato_late_blight": {
        "pv":      "Tomato___Late_blight",
        "pd_train":"Tomato leaf late blight",
        "pd_test": "Tomato leaf late blight",
    },
    "tomato_bacterial_spot": {
        "pv":      "Tomato___Bacterial_spot",
        "pd_train":"Tomato leaf bacterial spot",
        "pd_test": "Tomato leaf bacterial spot",
    },
    "tomato_leaf_mold": {
        "pv":      "Tomato___Leaf_Mold",
        "pd_train":"Tomato mold leaf",
        "pd_test": "Tomato mold leaf",
    },
}

def get_images(folder):
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.jpeg','.png'))]

stats = {}
for split in ["train","val","test"]:
    for cls in CLASS_MAP:
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

for dest_cls, sources in CLASS_MAP.items():
    # ── Collect PlantVillage images ──────────────────────────
    pv_imgs = get_images(os.path.join(PV_ROOT, sources["pv"]))
    random.shuffle(pv_imgs)
    pv_imgs = pv_imgs[:MAX_PV_PER_CLASS]

    # ── Collect PlantDoc train images (go into training) ─────
    pd_train_imgs = get_images(os.path.join(PD_TRAIN_ROOT, sources["pd_train"]))
    random.shuffle(pd_train_imgs)

    # ── Collect PlantDoc test images (go into val + test only) ─
    pd_test_imgs  = get_images(os.path.join(PD_TEST_ROOT,  sources["pd_test"]))
    random.shuffle(pd_test_imgs)

    # ── Split PV: 70/15/15 ──────────────────────────────────
    n      = len(pv_imgs)
    n_train= int(n * 0.70)
    n_val  = int(n * 0.15)
    pv_train = pv_imgs[:n_train]
    pv_val   = pv_imgs[n_train:n_train+n_val]
    pv_test  = pv_imgs[n_train+n_val:]

    # ── Split PlantDoc test: half val, half test ─────────────
    pd_half  = len(pd_test_imgs) // 2
    pd_val   = pd_test_imgs[:pd_half]
    pd_test2 = pd_test_imgs[pd_half:]

    # ── Combine ──────────────────────────────────────────────
    train_imgs = pv_train + pd_train_imgs
    val_imgs   = pv_val   + pd_val
    test_imgs  = pv_test  + pd_test2

    random.shuffle(train_imgs)

    def copy_set(img_list, split, cls):
        dst_dir = os.path.join(DST_ROOT, split, cls)
        for i, src in enumerate(img_list):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(dst_dir, f"{cls}_{split}_{i:04d}{ext}")
            shutil.copy2(src, dst)

    copy_set(train_imgs, "train", dest_cls)
    copy_set(val_imgs,   "val",   dest_cls)
    copy_set(test_imgs,  "test",  dest_cls)

    stats[dest_cls] = {
        "pv_used": len(pv_imgs),
        "plantdoc_train": len(pd_train_imgs),
        "plantdoc_test_split": len(pd_test_imgs),
        "final_train": len(train_imgs),
        "final_val":   len(val_imgs),
        "final_test":  len(test_imgs),
    }
    print(f"[OK] {dest_cls:30s}  train={len(train_imgs):4d}  val={len(val_imgs):3d}  test={len(test_imgs):3d}  "
          f"(PV={len(pv_imgs)} + PD_train={len(pd_train_imgs)} + PD_test={len(pd_test_imgs)})")

with open(os.path.join(DST_ROOT, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("\n[DONE] Mixed dataset ready.")
