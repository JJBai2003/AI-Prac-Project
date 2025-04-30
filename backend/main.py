from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask_cors import CORS  # Allow cross-origin requests



# class ConvNet(nn.Module):
#     def __init__(self, num_classes= 4):
#         super(ConvNet, self).__init__()
#         # TODO: define the network
#         self.conv1 = torch.nn.Conv2d(3,4,3,stride=2,padding=1)
#         self.conv2 = torch.nn.Conv2d(4,16,3,stride=2,padding=1)
#         self.conv3 = torch.nn.Conv2d(16,32,3,stride=2,padding=1)
#         self.activation = torch.nn.ReLU()
#         self.fc1 = torch.nn.Linear(2048,1024)
#         self.fc2 = torch.nn.Linear(1024,num_classes)
#         ## END TODO

#     def forward(self, x):
#         # TODO: create a convnet forward pass
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = self.conv3(x)
#         x = self.activation(x)
#         x = x.flatten(start_dim=1, end_dim=-1)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         ## END TODO

#         return x
    
# Load model
def load_model():
    model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)  
    model.classifier[4] = nn.Conv2d(256, 103, kernel_size=(1, 1))  # 103 classes
    model.load_state_dict(torch.load('food_segmentation_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_id_to_name = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients"
}


app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can communicate

@app.route('/api/predict', methods=['POST'])  # Match frontend endpoint
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = transform(img).unsqueeze(0) 
        print(img.shape)

        with torch.no_grad():
            output = model(img)['out']  # shape [1, num_classes, H, W]
            probs = torch.softmax(output.squeeze(0), dim=0)  # now shape [num_classes, H, W]

            # Compute average confidence per class
            avg_probs = probs.mean(dim=(1, 2))  # [num_classes]
            print(avg_probs)

            # Set a threshold
            threshold = 0.001  

            predicted_classes = (avg_probs > threshold).nonzero(as_tuple=True)[0]


        class_ids = predicted_classes.tolist()
        detected_ingredients = [class_id_to_name.get(class_id, f"unknown_{class_id}") for class_id in class_ids]

        # Create dummy recipes
        dummy_recipes = [
            {"title": f"{ingredient.title()} Surprise", "ingredients": [ingredient]} 
            for ingredient in detected_ingredients
        ]

        return jsonify({
            'recipes': dummy_recipes,
            'ingredients': detected_ingredients
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Frontend is calling port 5001
