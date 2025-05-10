from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from PIL import Image
import io
from flask_cors import CORS
# from pymongo import MongoClient
import os
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def load_model():
    model = deeplabv3_resnet101(weights=None, aux_loss=True)
    model.classifier = DeepLabHead(2048, 104)
    model.load_state_dict(torch.load('deeplabv3_Foodseg103.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


def load_gpt2_model():
    model_path = './checkpoint-2250'
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model = load_model()
gpt2_model, gpt2_tokenizer, gpt2_device = load_gpt2_model()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class_id_to_name = {    0: "background", 1: "candy", 2: "egg tart", 3: "french fries", 4: "chocolate", 5: "biscuit",
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
    102: "salad", 103: "other ingredients" }  # ← Paste your full 104-class dictionary here

# === MongoDB (optional) ===
# client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
# db = client["food_db"]
# recipes_collection = db["recipes"]

app = Flask(__name__)
CORS(app)


def generate_recipe(ingredients_list, max_length=512):
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

    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)


@app.route('/api/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        all_ingredients = []

        for file in files:
            file_bytes = file.stream.read()
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)['out']
                pred_classes = torch.argmax(output.squeeze(0), dim=0)
                class_ids = torch.unique(pred_classes).tolist()

                # Filter out small/noisy areas and background (0)
                for cid in class_ids:
                    if cid == 0:  # background
                        continue
                    pixel_area = (pred_classes == cid).sum().item()
                    if pixel_area > 100:  # ignore small regions
                        name = class_id_to_name.get(cid)
                        if name and name != "other ingredients":
                            all_ingredients.append(name)

        filtered_ingredients = sorted(set(all_ingredients))
        subset_size = min(len(filtered_ingredients), 3)
        subset = random.sample(filtered_ingredients, subset_size) if subset_size > 0 else []
        generated_recipe = generate_recipe(subset)

        return jsonify({
            "ingredients": filtered_ingredients,
            "subset_used": subset,
            "generated_recipe": generated_recipe.strip() if generated_recipe else "No recipe generated"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
