from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask_cors import CORS
from pymongo import MongoClient
import os
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model for food segmentation
def load_model():
    model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 104, kernel_size=(1, 1))
    model.load_state_dict(torch.load('food_segmentation_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load GPT2 model for recipe generation
def load_gpt2_model():
    model_path = './checkpoint-675'
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Instantiate models
model = load_model()
gpt2_model, gpt2_tokenizer, gpt2_device = load_gpt2_model()

# Image transformation for segmentation model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ingredient class-id mapping (to be used with segmentation model output)
class_id_to_name = {
    0: "background", 1: "candy", 2: "egg tart", 3: "french fries", 4: "chocolate", 5: "biscuit",
    6: "popcorn", 7: "pudding", 8: "ice cream", 9: "cheese butter", 10: "cake", 11: "wine",
    12: "milkshake", 13: "coffee", 14: "juice", 15: "milk", 16: "tea", 17: "almond", 18: "red beans",
    19: "cashew", 20: "dried cranberries", 21: "soy", 22: "walnut", 23: "peanut", 24: "egg",
    25: "apple", 26: "date", 27: "apricot", 28: "avocado", 29: "banana", 30: "strawberry",
    31: "cherry", 32: "blueberry", 33: "raspberry", 34: "mango", 35: "olives", 36: "peach",
    37: "lemon", 38: "pear", 39: "fig", 40: "pineapple", 41: "grape", 42: "kiwi", 43: "melon",
    44: "orange", 45: "watermelon", 46: "steak", 47: "pork", 48: "chicken duck", 49: "sausage",
    50: "fried meat", 51: "lamb", 52: "sauce", 53: "crab", 54: "fish", 55: "shellfish", 56: "shrimp",
    57: "soup", 58: "bread", 59: "corn", 60: "hamburg", 61: "pizza", 62: "hanamaki baozi",
    63: "wonton dumplings", 64: "pasta", 65: "noodles", 66: "rice", 67: "pie", 68: "tofu",
    69: "eggplant", 70: "potato", 71: "garlic", 72: "cauliflower", 73: "tomato", 74: "kelp",
    75: "seaweed", 76: "spring onion", 77: "rape", 78: "ginger", 79: "okra", 80: "lettuce",
    81: "pumpkin", 82: "cucumber", 83: "white radish", 84: "carrot", 85: "asparagus",
    86: "bamboo shoots", 87: "broccoli", 88: "celery stick", 89: "cilantro mint",
    90: "snow peas", 91: "cabbage", 92: "bean sprouts", 93: "onion", 94: "pepper",
    95: "green beans", 96: "French beans", 97: "king oyster mushroom", 98: "shiitake",
    99: "enoki mushroom", 100: "oyster mushroom", 101: "white button mushroom",
    102: "salad", 103: "other ingredients"
}

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = client["food_db"]
recipes_collection = db["recipes"]

# Flask app setup
app = Flask(__name__)
CORS(app)

# Function to generate a recipe from ingredients list using GPT2
def generate_recipe(ingredients_list, max_length=300):
    """
    Generate a recipe given a list of ingredients.
    Returns the generated text as a string.
    """
    prompt = f"Ingredients: {', '.join(ingredients_list)}\nRecipe Name:"
    inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(gpt2_device)

    output = gpt2_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        eos_token_id=gpt2_tokenizer.eos_token_id
    )

    generated_text = gpt2_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return generated_text

# Endpoint for uploading image and getting ingredients and recipes
@app.route('/api/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        all_ingredients = []

        # Process the uploaded images and extract ingredients
        for file in files:
            file_bytes = file.stream.read()
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)['out']
                probs = torch.softmax(output.squeeze(0), dim=0)
                avg_probs = probs.mean(dim=(1, 2))

                threshold = 0.02
                predicted_classes = (avg_probs > threshold).nonzero(as_tuple=True)[0]
                class_ids = predicted_classes.tolist()

                ingredients = [
                    class_id_to_name.get(cid, f"unknown_{cid}")
                    for cid in class_ids
                    if class_id_to_name.get(cid, f"unknown_{cid}") != "background"
                ]
                all_ingredients.append(ingredients)

        # Flatten ingredients list and remove duplicates
        flat_ingredients = sorted(set(i for sub in all_ingredients for i in sub))
        filtered_ingredients = [
    item for item in flat_ingredients if item.lower() != "other ingredients"
]

        subset = random.sample(flat_ingredients, min(len(flat_ingredients), 3))
        generated_recipe = generate_recipe(subset)
        print(subset)
        return jsonify({
                "ingredients": subset,
                "generated_recipe": generated_recipe.strip()
            })

     
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
