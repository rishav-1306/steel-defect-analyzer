import os
from PIL import Image
from torch.utils.data import Dataset

class SteelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        images_path = os.path.join(root_dir, "images")

        classes = sorted(os.listdir(images_path))

        for label, cls in enumerate(classes):
            cls_path = os.path.join(images_path, cls)

            if not os.path.isdir(cls_path):
                continue

            for file in os.listdir(cls_path):
                img_path = os.path.join(cls_path, file)

                try:
                    with Image.open(img_path) as img:
                        img.verify()

                    self.images.append(img_path)
                    self.labels.append(label)

                except:
                    continue

        print("Classes:", classes)
        print("Total images loaded:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label