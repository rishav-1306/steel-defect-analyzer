import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import SteelCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (IMPORTANT)
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Load model
model = SteelCNN().to(device)
model.load_state_dict(torch.load("../models/steel_cnn.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], confidence.item()

# 🔥 TEST IMAGE PATH (CHANGE THIS)
img_path = "../data/NEU-DET/train/images/scratches/scratches_1.jpg"

label, conf = predict_image(img_path)

print(f"Prediction: {label}")
print(f"Confidence: {conf*100:.2f}%")