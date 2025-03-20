from flask import Flask, request, jsonify
import torch
from model.architecture import PneumoniaClassifier
from utils.config import DEVICE, NUM_CLASSES
from data.preprocessing import val_test_transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the optimized model
model = PneumoniaClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('model/optimized_pneumonia_classifier.pth', map_location=DEVICE))
model.eval()

# Define prediction function
def predict_image(image_file):
    # Open and preprocess the image
    image = Image.open(image_file).convert('RGB')
    image_tensor = val_test_transforms(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        confidence = probs[0][predicted].item()
        label = "Pneumonia" if predicted.item() == 1 else "Normal"
    
    return label, confidence

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    try:
        label, confidence = predict_image(image_file)
        return jsonify({
            'prediction': label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)