from main import transform, model, class_id_to_name  
from PIL import Image
import torch
from sklearn.metrics import *

validation_data = [
    {"image": "./images/protein_others.jpeg", "ground_truth": ["apple", "blueberry", "almond", "lemon", "steak", "ginger", "sauce"]},
    {"image": "./images/noodle.jpg", "ground_truth": ["noodles"]},
    {"image": "./images/veggie.png", "ground_truth": ["lettuce", "grape", "tomato", "cucumber", "broccoli", "pepper", "orange"]},
    {"image": "./images/candy.png", "ground_truth": ["candy", "egg tart", "french fries", "chocolate"]},
    {"image": "./images/veggie2.jpg",
    "ground_truth": [
        "carrot", "broccoli", "cauliflower", "tomato", "pepper", "cucumber", "garlic", "potato",
        "eggplant", "lettuce", "onion"
    ]},
    {"image": "./images/fruit.png",  
    "ground_truth": [
        "orange", "lemon","banana", "mango", "grape", "blueberry", "apple",
        "peach",
    ]}, 
    # image generated from gpt to get the other food items for testing
    {  "image": "./images/food.png",
    "ground_truth": [
        "biscuit", "popcorn", "sausage", "sauce", "avocado", "pizza", "lettuce", "asparagus",
        "white radish", "pear", "kiwi", "olive", "strawberry", "peach", "almond", "cashew",
        "walnut", "pasta", "coffee", "cake", "bread", "cherry"
    ]
},
]

def detect_ingredients(image_path):
    """
    Detect ingredients in an image using the segmentation model.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)['out']
            pred_classes = torch.argmax(output.squeeze(0), dim=0)
            class_ids = torch.unique(pred_classes).tolist()

            detected_ingredients = []
            for cid in class_ids:
                if cid == 0:  
                    continue
                pixel_area = (pred_classes == cid).sum().item()
                if pixel_area > 100:  
                    name = class_id_to_name.get(cid)
                    if name and name != "other ingredients":
                        detected_ingredients.append(name)

            return sorted(set(detected_ingredients))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def evaluate_model(validation_data):
    """
    Evaluate the ingredient detection model using precision, recall, and F1-score.
    """
    all_ground_truth = []
    all_predictions = []

    for data in validation_data:
        image_path = data["image"]
        ground_truth = data["ground_truth"]

        predicted_ingredients = detect_ingredients(image_path)

        all_ground_truth.append(set(ground_truth))
        all_predictions.append(set(predicted_ingredients))

        print(f"Image: {image_path}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted: {predicted_ingredients}")
        print("-" * 50)

    flat_ground_truth = [item for sublist in all_ground_truth for item in sublist]
    flat_predictions = [item for sublist in all_predictions for item in sublist]

    # Calculate Precision, Recall, and F1 Score
    true_positives = len(set(flat_ground_truth) & set(flat_predictions))
    false_positives = len(set(flat_predictions) - set(flat_ground_truth))
    false_negatives = len(set(flat_ground_truth) - set(flat_predictions))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    evaluate_model(validation_data)