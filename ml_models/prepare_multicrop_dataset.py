"""
ALIP - Multi-Crop Dataset Preparation
Combines PlantVillage (lab) + PlantDoc (real field) images into one
unified train/val/test dataset for domain-robust training across 4 crops (19 classes):
Tomato (10 classes), Potato (3 classes), Corn (4 classes), Pepper (2 classes).

Strategy:
 - PlantVillage: 70% of max 1200 per class (train), rest to val/test (15%/15%)
 - PlantDoc train: goes 100% into training set
 - PlantDoc test: split 50/50 between val and test
"""

import os
import shutil
import random
import json

random.seed(42)

PV_ROOT       = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
PD_TRAIN_ROOT = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\train"
PD_TEST_ROOT  = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
DST_ROOT      = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_dataset"

MAX_PV_PER_CLASS = 1200

CLASS_MAP = {
    # Corn (4 classes)
    "corn_cercospora_leaf_spot": {
        "pv": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "pd_train": "Corn Gray leaf spot",
        "pd_test":  "Corn Gray leaf spot"
    },
    "corn_common_rust": {
        "pv": "Corn_(maize)___Common_rust_",
        "pd_train": "Corn rust leaf",
        "pd_test":  "Corn rust leaf"
    },
    "corn_healthy": {
        "pv": "Corn_(maize)___healthy",
        "pd_train": None,
        "pd_test":  None
    },
    "corn_northern_leaf_blight": {
        "pv": "Corn_(maize)___Northern_Leaf_Blight",
        "pd_train": "Corn leaf blight",
        "pd_test":  "Corn leaf blight"
    },

    # Pepper (2 classes)
    "pepper_bacterial_spot": {
        "pv": "Pepper,_bell___Bacterial_spot",
        "pd_train": "Bell_pepper leaf spot",
        "pd_test":  "Bell_pepper leaf spot"
    },
    "pepper_healthy": {
        "pv": "Pepper,_bell___healthy",
        "pd_train": "Bell_pepper leaf",
        "pd_test":  "Bell_pepper leaf"
    },

    # Potato (3 classes)
    "potato_early_blight": {
        "pv": "Potato___Early_blight",
        "pd_train": "Potato leaf early blight",
        "pd_test":  "Potato leaf early blight"
    },
    "potato_healthy": {
        "pv": "Potato___healthy",
        "pd_train": None,
        "pd_test":  None
    },
    "potato_late_blight": {
        "pv": "Potato___Late_blight",
        "pd_train": "Potato leaf late blight",
        "pd_test":  "Potato leaf late blight"
    },

    # Tomato (10 classes)
    "tomato_bacterial_spot": {
        "pv": "Tomato___Bacterial_spot",
        "pd_train": "Tomato leaf bacterial spot",
        "pd_test":  "Tomato leaf bacterial spot"
    },
    "tomato_early_blight": {
        "pv": "Tomato___Early_blight",
        "pd_train": "Tomato Early blight leaf",
        "pd_test":  "Tomato Early blight leaf"
    },
    "tomato_healthy": {
        "pv": "Tomato___healthy",
        "pd_train": "Tomato leaf",
        "pd_test":  "Tomato leaf"
    },
    "tomato_late_blight": {
        "pv": "Tomato___Late_blight",
        "pd_train": "Tomato leaf late blight",
        "pd_test":  "Tomato leaf late blight"
    },
    "tomato_leaf_mold": {
        "pv": "Tomato___Leaf_Mold",
        "pd_train": "Tomato mold leaf",
        "pd_test":  "Tomato mold leaf"
    },
    "tomato_mosaic_virus": {
        "pv": "Tomato___Tomato_mosaic_virus",
        "pd_train": "Tomato leaf mosaic virus",
        "pd_test":  "Tomato leaf mosaic virus"
    },
    "tomato_septoria_leaf_spot": {
        "pv": "Tomato___Septoria_leaf_spot",
        "pd_train": "Tomato Septoria leaf spot",
        "pd_test":  "Tomato Septoria leaf spot"
    },
    "tomato_spider_mites": {
        "pv": "Tomato___Spider_mites Two-spotted_spider_mite",
        "pd_train": None,
        "pd_test":  None
    },
    "tomato_target_spot": {
        "pv": "Tomato___Target_Spot",
        "pd_train": None,
        "pd_test":  None
    },
    "tomato_yellow_leaf_curl_virus": {
        "pv": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "pd_train": "Tomato leaf yellow virus",
        "pd_test":  "Tomato leaf yellow virus"
    },
}

def get_images(folder):
    if not folder or not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def main():
    print(f"Preparing multi-crop dataset in: {DST_ROOT}")
    print(f"Total target classes: {len(CLASS_MAP)}")

    for split in ["train", "val", "test"]:
        for cls in sorted(CLASS_MAP.keys()):
            os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

    stats = {}

    for dest_cls in sorted(CLASS_MAP.keys()):
        sources = CLASS_MAP[dest_cls]

        # ── PlantVillage ──
        pv_folder = os.path.join(PV_ROOT, sources["pv"])
        pv_imgs = get_images(pv_folder)
        random.shuffle(pv_imgs)
        pv_imgs = pv_imgs[:MAX_PV_PER_CLASS]

        # ── PlantDoc Train ──
        pd_train_folder = os.path.join(PD_TRAIN_ROOT, sources["pd_train"]) if sources["pd_train"] else None
        pd_train_imgs = get_images(pd_train_folder)
        random.shuffle(pd_train_imgs)

        # ── PlantDoc Test ──
        pd_test_folder = os.path.join(PD_TEST_ROOT, sources["pd_test"]) if sources["pd_test"] else None
        pd_test_imgs = get_images(pd_test_folder)
        random.shuffle(pd_test_imgs)

        # ── Split PV: 70% train / 15% val / 15% test ──
        n = len(pv_imgs)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        pv_train = pv_imgs[:n_train]
        pv_val = pv_imgs[n_train:n_train + n_val]
        pv_test = pv_imgs[n_train + n_val:]

        # ── Split PlantDoc test: 50% val / 50% test ──
        pd_half = len(pd_test_imgs) // 2
        pd_val = pd_test_imgs[:pd_half]
        pd_test2 = pd_test_imgs[pd_half:]

        # ── Combine ──
        train_imgs = pv_train + pd_train_imgs
        val_imgs = pv_val + pd_val
        test_imgs = pv_test + pd_test2

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
            "plantdoc_test": len(pd_test_imgs),
            "final_train": len(train_imgs),
            "final_val": len(val_imgs),
            "final_test": len(test_imgs),
        }

        print(f"[OK] {dest_cls:30s}  train={len(train_imgs):4d}  val={len(val_imgs):3d}  test={len(test_imgs):3d}  "
              f"(PV={len(pv_imgs)} + PD_train={len(pd_train_imgs)} + PD_test={len(pd_test_imgs)})")

    stats_path = os.path.join(DST_ROOT, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    total_train = sum(s["final_train"] for s in stats.values())
    total_val = sum(s["final_val"] for s in stats.values())
    total_test = sum(s["final_test"] for s in stats.values())
    print("\n" + "=" * 70)
    print(f"[DONE] Multi-crop dataset ready at {DST_ROOT}")
    print(f"Total Train: {total_train} | Val: {total_val} | Test: {total_test} | Total: {total_train+total_val+total_test}")
    print(f"Stats written to: {stats_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
