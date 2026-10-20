"""
ALIP - prepare_multicrop_v2_dataset.py
Builds 20-class dataset with:
  - background_other class (Apple, Grape, Soybean, Strawberry images)
  - No cap on potato_healthy (use all 152)
  - All PlantDoc potato images go to training
  - Same 70/15/15 split for everything else
"""

import os, shutil, random, json
random.seed(42)

PV_ROOT       = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantVillage-Dataset-master\raw\color"
PD_TRAIN_ROOT = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\train"
PD_TEST_ROOT  = r"D:\ALIP-TOTAL\ALIP\ml_models\PlantDoc-Dataset-master\test"
DST_ROOT      = r"D:\ALIP-TOTAL\ALIP\ml_models\multicrop_v2_dataset"

MAX_PV_PER_CLASS = 1200  # cap for non-potato classes

CLASS_MAP = {
    # Corn (4)
    "corn_cercospora_leaf_spot": {
        "pv": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "pd_train": "Corn Gray leaf spot",
        "pd_test":  "Corn Gray leaf spot",
        "pv_cap":   None,  # use all 513
    },
    "corn_common_rust": {
        "pv": "Corn_(maize)___Common_rust_",
        "pd_train": "Corn rust leaf",
        "pd_test":  "Corn rust leaf",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "corn_healthy": {
        "pv": "Corn_(maize)___healthy",
        "pd_train": None,
        "pd_test":  None,
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "corn_northern_leaf_blight": {
        "pv": "Corn_(maize)___Northern_Leaf_Blight",
        "pd_train": "Corn leaf blight",
        "pd_test":  "Corn leaf blight",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    # Pepper (2)
    "pepper_bacterial_spot": {
        "pv": "Pepper,_bell___Bacterial_spot",
        "pd_train": "Bell_pepper leaf spot",
        "pd_test":  "Bell_pepper leaf spot",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "pepper_healthy": {
        "pv": "Pepper,_bell___healthy",
        "pd_train": "Bell_pepper leaf",
        "pd_test":  "Bell_pepper leaf",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    # Potato (3) — NO cap on healthy, use all available
    "potato_early_blight": {
        "pv": "Potato___Early_blight",
        "pd_train": "Potato leaf early blight",
        "pd_test":  "Potato leaf early blight",
        "pv_cap":   1000,
    },
    "potato_healthy": {
        "pv": "Potato___healthy",
        "pd_train": None,
        "pd_test":  None,
        "pv_cap":   None,  # use ALL 152
    },
    "potato_late_blight": {
        "pv": "Potato___Late_blight",
        "pd_train": "Potato leaf late blight",
        "pd_test":  "Potato leaf late blight",
        "pv_cap":   1000,
    },
    # Tomato (10)
    "tomato_bacterial_spot": {
        "pv": "Tomato___Bacterial_spot",
        "pd_train": "Tomato leaf bacterial spot",
        "pd_test":  "Tomato leaf bacterial spot",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_early_blight": {
        "pv": "Tomato___Early_blight",
        "pd_train": "Tomato Early blight leaf",
        "pd_test":  "Tomato Early blight leaf",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_healthy": {
        "pv": "Tomato___healthy",
        "pd_train": "Tomato leaf",
        "pd_test":  "Tomato leaf",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_late_blight": {
        "pv": "Tomato___Late_blight",
        "pd_train": "Tomato leaf late blight",
        "pd_test":  "Tomato leaf late blight",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_leaf_mold": {
        "pv": "Tomato___Leaf_Mold",
        "pd_train": "Tomato mold leaf",
        "pd_test":  "Tomato mold leaf",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_mosaic_virus": {
        "pv": "Tomato___Tomato_mosaic_virus",
        "pd_train": "Tomato leaf mosaic virus",
        "pd_test":  "Tomato leaf mosaic virus",
        "pv_cap":   None,  # only 373
    },
    "tomato_septoria_leaf_spot": {
        "pv": "Tomato___Septoria_leaf_spot",
        "pd_train": "Tomato Septoria leaf spot",
        "pd_test":  "Tomato Septoria leaf spot",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_spider_mites": {
        "pv": "Tomato___Spider_mites Two-spotted_spider_mite",
        "pd_train": None,
        "pd_test":  None,
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_target_spot": {
        "pv": "Tomato___Target_Spot",
        "pd_train": None,
        "pd_test":  None,
        "pv_cap":   MAX_PV_PER_CLASS,
    },
    "tomato_yellow_leaf_curl_virus": {
        "pv": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "pd_train": "Tomato leaf yellow virus",
        "pd_test":  "Tomato leaf yellow virus",
        "pv_cap":   MAX_PV_PER_CLASS,
    },
}

# 20th class: background_other
# Pull from unsupported crops in PlantVillage (Apple, Grape, Soybean, Strawberry)
BACKGROUND_SOURCES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Soybean___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Peach___Bacterial_spot",
]
BACKGROUND_TARGET = 600  # ~420 train / 90 val / 90 test


def get_images(folder):
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create all output directories
for split in ["train", "val", "test"]:
    for cls in list(CLASS_MAP.keys()) + ["background_other"]:
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

stats = {}

# ── Process 19 main classes ──────────────────────────────────────────────
for dest_cls, sources in CLASS_MAP.items():
    cap   = sources.get("pv_cap", MAX_PV_PER_CLASS)

    pv_imgs = get_images(os.path.join(PV_ROOT, sources["pv"]))
    random.shuffle(pv_imgs)
    if cap:
        pv_imgs = pv_imgs[:cap]

    pd_train_all = []
    if sources.get("pd_train"):
        pd_train_all = get_images(os.path.join(PD_TRAIN_ROOT, sources["pd_train"]))
        random.shuffle(pd_train_all)

    pd_test_all = []
    if sources.get("pd_test"):
        pd_test_all = get_images(os.path.join(PD_TEST_ROOT, sources["pd_test"]))
        random.shuffle(pd_test_all)

    n       = len(pv_imgs)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    pv_train = pv_imgs[:n_train]
    pv_val   = pv_imgs[n_train:n_train + n_val]
    pv_test  = pv_imgs[n_train + n_val:]

    pd_half = len(pd_test_all) // 2
    pd_val  = pd_test_all[:pd_half]
    pd_test = pd_test_all[pd_half:]

    train_imgs = pv_train + pd_train_all
    val_imgs   = pv_val   + pd_val
    test_imgs  = pv_test  + pd_test

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
        "pv_used": len(pv_imgs), "pd_train": len(pd_train_all),
        "pd_test_split": len(pd_test_all),
        "final_train": len(train_imgs), "final_val": len(val_imgs),
        "final_test": len(test_imgs),
    }
    print(f"[OK] {dest_cls:35s} train={len(train_imgs):4d}  val={len(val_imgs):3d}  test={len(test_imgs):3d}")

# ── Process background_other (20th class) ────────────────────────────────
print("\nBuilding background_other class...")
all_bg_imgs = []
for src_folder in BACKGROUND_SOURCES:
    path = os.path.join(PV_ROOT, src_folder)
    imgs = get_images(path)
    if imgs:
        random.shuffle(imgs)
        # take proportional share
        share = BACKGROUND_TARGET // len(BACKGROUND_SOURCES)
        all_bg_imgs.extend(imgs[:share])
        print(f"  {src_folder}: took {min(len(imgs), share)} / {len(imgs)}")

random.shuffle(all_bg_imgs)
all_bg_imgs = all_bg_imgs[:BACKGROUND_TARGET]

n      = len(all_bg_imgs)
n_tr   = int(n * 0.70)
n_val  = int(n * 0.15)
bg_tr  = all_bg_imgs[:n_tr]
bg_val = all_bg_imgs[n_tr:n_tr + n_val]
bg_te  = all_bg_imgs[n_tr + n_val:]

for i, src in enumerate(bg_tr):
    ext = os.path.splitext(src)[1]
    shutil.copy2(src, os.path.join(DST_ROOT, "train", "background_other", f"bg_train_{i:04d}{ext}"))
for i, src in enumerate(bg_val):
    ext = os.path.splitext(src)[1]
    shutil.copy2(src, os.path.join(DST_ROOT, "val", "background_other", f"bg_val_{i:04d}{ext}"))
for i, src in enumerate(bg_te):
    ext = os.path.splitext(src)[1]
    shutil.copy2(src, os.path.join(DST_ROOT, "test", "background_other", f"bg_test_{i:04d}{ext}"))

stats["background_other"] = {
    "total": n, "final_train": len(bg_tr),
    "final_val": len(bg_val), "final_test": len(bg_te)
}
print(f"[OK] background_other               train={len(bg_tr):4d}  val={len(bg_val):3d}  test={len(bg_te):3d}")

# Verify all 20 class dirs exist
for split in ["train", "val", "test"]:
    classes = sorted(os.listdir(os.path.join(DST_ROOT, split)))
    print(f"\n{split}: {len(classes)} classes -> {classes}")

with open(os.path.join(DST_ROOT, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("\n[DONE] multicrop_v2_dataset ready.")
