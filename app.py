from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from cnntrain import CNNModel
import os

app = Flask(__name__)

model = torch.load('cnn_model.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Define preprocessing for uploaded images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


@app.route('/')
def upload_page():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = 'Non Flooded' if predicted.item() == 1 else 'Flooded'

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
