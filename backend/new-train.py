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
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR


ssl._create_default_https_context = ssl._create_unverified_context


UNIQUE_LABELS = list(range(104))  
label_mapping = {label: idx for idx, label in enumerate(UNIQUE_LABELS)}
num_classes = len(label_mapping)


def transform_image(image):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image)


def transform_mask(mask):
    mask = mask.convert("L")
    mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = np.array(mask)

    remapped = np.full_like(mask, fill_value=255)  
    for original, new in label_mapping.items():
        remapped[mask == original] = new

    return torch.tensor(remapped, dtype=torch.long)


class SegmentationDataset(Dataset):
    def __init__(self, hf_dataset, image_transform=None, mask_transform=None):
        self.dataset = hf_dataset
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image']
            mask = item['label']

            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            return image, mask
        except Exception as e:
            print(f"Skipping item {idx} due to error: {e}")
            return None

# === Collate to skip bad samples ===
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

# === Save and load checkpoint ===
def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Resumed training from epoch {epoch}")
    return model, optimizer, epoch


def main():
    # Load dataset
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    print("Train keys:", dataset['train'].features)
    print("Sample keys:", dataset['train'][0].keys())

   
    train_dataset = SegmentationDataset(dataset['train'], transform_image, transform_mask)
    val_dataset = SegmentationDataset(dataset['validation'], transform_image, transform_mask)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Model: DeepLabV3 + ResNet101
    model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=True
    )
    model.classifier = DeepLabHead(2048, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

 
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()


    checkpoint_path = 'checkpoint.pth'
    start_epoch = 0
    try:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")


    num_epochs = 20
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        print(f"✅ Epoch {epoch+1}/{num_epochs} completed | Avg Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, avg_loss)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                outputs = model(val_images)['out']
                loss = criterion(outputs, val_masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

        # === Early stopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_loss, filename='best_model.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Stopping at epoch {epoch + 1}.")
            break


    torch.save(model.state_dict(), 'deeplabv3_Foodseg103.pth')
    print("🎉 Training completed and model saved!")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
