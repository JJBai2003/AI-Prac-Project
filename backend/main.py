
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io


model = torch.load('food_detection_model.pth')
model.eval()  

# same transformation used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read image
        img = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        img = transform(img).unsqueeze(0)  
        
   
        with torch.no_grad():  
            output = model(img)
            _, predicted_class = torch.max(output, 1)

      
        predicted_class = predicted_class.item()
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
