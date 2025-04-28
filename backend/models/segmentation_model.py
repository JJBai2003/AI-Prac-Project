import torch
from your_model_def import YourFoodSegmentationModel  # make sure this file defines your model class

model = YourFoodSegmentationModel()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
model.eval()
