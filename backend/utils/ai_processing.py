import torch
from torchvision import transforms
from PIL import Image
import os
from ai_models.model import load_model

class ImageProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.getenv("MODEL_PATH")
        self.classes = self._load_classes()
        self.model = load_model(self.model_path, self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_classes(self):
        return ['egg', 'milk', 'flour', 'sugar', 'butter',
                'chicken', 'beef', 'tomato', 'onion', 'garlic']

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.sigmoid(outputs)
                detected = [self.classes[i] for i, p in enumerate(probs[0]) if p > 0.5]
            
            return detected
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return []