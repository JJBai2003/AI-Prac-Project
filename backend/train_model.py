import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from datasets import load_dataset
from PIL import Image
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def transform_image(image):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image)

def transform_mask(mask):
    mask = transforms.Resize((128, 128))(mask)
    return torch.tensor(np.array(mask), dtype=torch.long)

class SegmentationDataset(Dataset):
    def __init__(self, hf_dataset, image_transform=None, mask_transform=None):
        self.dataset = hf_dataset
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        mask = item['label']

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

def main():
    dataset = load_dataset("EduardoPacheco/FoodSeg103")

    print("\n=== Dataset Structure ===")
    print("Train features:", dataset['train'].features)
    print("Sample item keys:", dataset['train'][0].keys())

    train_dataset = SegmentationDataset(
        hf_dataset=dataset['train'],
        image_transform=transform_image,
        mask_transform=transform_mask
    )
    val_dataset = SegmentationDataset(
        hf_dataset=dataset['validation'],
        image_transform=transform_image,
        mask_transform=transform_mask
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = models.segmentation.deeplabv3_resnet50(
        weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=True
    )

    model.classifier = DeepLabHead(2048, 104)  # Adjust num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'food_segmentation_model.pth')
    print("\n Training completed successfully!")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
