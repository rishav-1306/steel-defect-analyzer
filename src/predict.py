import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import SteelCNN

# ── Config ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70  # 70% — reject predictions below this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (must match training class order)
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# ── Load model ───────────────────────────────────────────────────────────
model = SteelCNN().to(device)
model.load_state_dict(torch.load("../models/steel_cnn.pth", map_location=device, weights_only=True))
model.eval()

# ── Image transform ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(img_path):
    """
    Predict the defect class for a given image path.
    Returns (class_name, confidence, is_reliable).
    If confidence < CONFIDENCE_THRESHOLD, is_reliable is False.
    """
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred_class = classes[predicted.item()]
    conf = confidence.item()
    is_reliable = conf >= CONFIDENCE_THRESHOLD

    return pred_class, conf, is_reliable


# ── Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change this to test any image
    img_path = "../data/NEU-DET/train/images/scratches/scratches_1.jpg"

    label, conf, reliable = predict_image(img_path)

    print(f"\n{'='*50}")
    print(f"  Image: {img_path}")
    print(f"{'='*50}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf*100:.2f}%")
    print(f"  Threshold  : {CONFIDENCE_THRESHOLD*100:.0f}%")

    if reliable:
        print(f"  Status     : ✅ Reliable prediction")
    else:
        print(f"  Status     : ⛔ REJECTED — Not a recognized steel defect")
        print(f"               The image may not be a steel surface or is too ambiguous.")

    print(f"{'='*50}\n")
