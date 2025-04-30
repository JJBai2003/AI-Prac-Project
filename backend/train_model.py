
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
import numpy as np
import ssl

# Disabled SSL verification 
ssl._create_default_https_context = ssl._create_unverified_context


def transform_image(image):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image)

def transform_mask(mask):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])(mask).squeeze().long()

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))

        assert len(self.image_filenames) == len(self.mask_filenames), "Mismatch between images and masks."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 'L' mode = single-channel grayscale

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Custom collate function to filter out None values
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

def main():
    # Load dataset
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    
    # Print dataset structure for verification
    print("\n=== Dataset Structure ===")
    print("Train features:", dataset['train'].features)
    sample_item = dataset['train'][0]
    print("Sample item keys:", sample_item.keys())
    
    # Initialize datasets
    train_dataset = SegmentationDataset(dataset['train'])
    val_dataset = SegmentationDataset(dataset['validation'])

    # Create DataLoaders with error handling
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

    # Model setup
    model = models.segmentation.deeplabv3_resnet50(
        weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    model.classifier[4] = nn.Conv2d(256, 103, kernel_size=(1, 1))  # 103 classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
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
            
            # progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {running_loss/len(train_loader):.4f}")

   
    torch.save(model.state_dict(), 'food_segmentation_model.pth')
    print("\n✅ Training completed successfully!")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()