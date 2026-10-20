"""
Quick test: what does the tomato model predict when given
completely unrelated images?
"""
import os, json, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

MODEL_PATH   = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_v2.pt"
CLASSES_PATH = r"D:\ALIP-TOTAL\ALIP\ml_models\tomato_class_names.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out  = model(t)
        prob = torch.softmax(out[0], 0)
        top_probs, top_idx = torch.topk(prob, 3)
    results = [(CLASS_NAMES[i.item()], p.item()*100) for i, p in zip(top_idx, top_probs)]
    max_conf = top_probs[0].item() * 100
    gate = "PASSED (shown to user)" if max_conf >= 60 else "REJECTED (conf<60%)"
    return results, gate

# The images are in train/test subfolders
TEST_CASES = [
    (r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field\Tomato_early_blight\train",  "Tomato Early Blight (CORRECT crop)"),
    (r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field\Tomato_healthy\train",       "Tomato Healthy (CORRECT crop)"),
    (r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field\Potato_early_blight\train",  "Potato leaf (WRONG crop - not tomato)"),
    (r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field\Corn_common_rust\train",     "Corn leaf (WRONG crop - not tomato)"),
    (r"D:\ALIP-TOTAL\ALIP\ml_models\plantdoc_field\Pepper_healthy\train",       "Pepper leaf (WRONG crop - not tomato)"),
]

print("=" * 70)
print("MODEL BEHAVIOUR WITH CORRECT vs WRONG CROP IMAGES")
print("=" * 70)

for folder, label in TEST_CASES:
    if not os.path.exists(folder):
        print(f"\n[SKIP] {label} - not found")
        continue
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        print(f"\n[SKIP] {label} - no images")
        continue

    print(f"\n[{label}]")
    passed = rejected = 0
    for img_path in imgs[:5]:
        preds, gate = predict(img_path)
        top_pred, top_conf = preds[0]
        if "PASSED" in gate:
            passed += 1
        else:
            rejected += 1
        print(f"  -> {top_pred:33s} {top_conf:5.1f}%  | {gate}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If WRONG crop images PASS through (conf >= 60%), the model will")
print("give a WRONG diagnosis to the farmer. This is the safety problem.")
