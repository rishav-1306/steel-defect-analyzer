from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import SteelDataset
from model import SteelCNN

print("🚀 TRAINING STARTED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = SteelDataset("../data/NEU-DET/train", transform=transform)
print("Total images:", len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SteelCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    loop = tqdm(train_loader, leave=True)

    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 🔥 Update progress bar
        loop.set_postfix(
            loss=loss.item(),
            avg_loss=running_loss/(batch_idx+1)
        )

# Save model
torch.save(model.state_dict(), "../models/steel_cnn.pth")
print("\n✅ Model saved successfully!")