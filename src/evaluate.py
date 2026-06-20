import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset import SteelDataset
from model import SteelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Loading validation dataset...")
val_dataset = SteelDataset("../data/NEU-DET/validation", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

model = SteelCNN().to(device)
model.load_state_dict(torch.load("../models/steel_cnn.pth", map_location=device, weights_only=True))
model.eval()

correct = 0
total = 0
class_correct = {c: 0 for c in classes}
class_total = {c: 0 for c in classes}

print(f"\nEvaluating on {len(val_dataset)} validation images...")

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            label_idx = labels[i].item()
            cls_name = classes[label_idx]
            class_total[cls_name] += 1
            if predicted[i].item() == label_idx:
                class_correct[cls_name] += 1

overall_accuracy = 100.0 * correct / total

results = {
    "overall_accuracy": round(overall_accuracy, 2),
    "total_images": total,
    "correct_predictions": correct,
    "class_accuracy": {}
}

print(f"\n{'='*50}")
print(f"  OVERALL ACCURACY: {overall_accuracy:.2f}%")
print(f"  Total: {total} | Correct: {correct}")
print(f"{'='*50}\n")
print("Per-class accuracy:")
for cls in classes:
    if class_total[cls] > 0:
        acc = 100.0 * class_correct[cls] / class_total[cls]
        results["class_accuracy"][cls] = round(acc, 2)
        print(f"  {cls:<20}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

# Save results to JSON for dashboard use
with open("../outputs/accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to outputs/accuracy_results.json")
