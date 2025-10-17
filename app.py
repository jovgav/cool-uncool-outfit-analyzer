from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import json
from model import CoolUncoolCNN, CoolUncoolTrainer
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model"""
    global model
    model_path = "cool_uncool_model.pth"
    
    if not os.path.exists(model_path):
        return False
    
    try:
        model = CoolUncoolCNN(num_classes=2)
        trainer = CoolUncoolTrainer(model, device)
        trainer.load_model(model_path)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_image(image):
    """Make prediction on an image"""
    if model is None:
        return None, 0.0
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, 0.0

@app.route('/')
def index():
    """Main page"""
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train a model first.'}), 400
    
    try:
        # Read image
        image = Image.open(file.stream)
        
        # Make prediction
        predicted_class, confidence = predict_image(image)
        
        if predicted_class is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        # Prepare result
        result = {
            'prediction': 'cool' if predicted_class == 1 else 'uncool',
            'confidence': round(confidence * 100, 1)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    """Check if model is loaded"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint to train the model"""
    try:
        # This would integrate with your training script
        # For now, return a placeholder response
        return jsonify({
            'status': 'success',
            'message': 'Training started. This will take several minutes.',
            'redirect': url_for('index')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    model_loaded = load_model()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ No trained model found. Please train a model first.")
    
    # Run the app
    print("🚀 Starting web server...")
    print("🌐 Open your browser and go to: http://localhost:3000")
    app.run(debug=True, host='127.0.0.1', port=3000)
