import pandas as pd
import numpy as np
import torch
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Add after app initialization

# Define paths to model files in the models/ subdirectory
MODEL_DIR = 'models'
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.pth')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Load pre-trained models and preprocessing objects
with open(RF_MODEL_PATH, 'rb') as f:
    rf_model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Define LSTM model class (must match training architecture)
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, num_classes=22):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load LSTM model
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
lstm_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model.to(device)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'model']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Extract features and model type
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'], 
                             data['humidity'], data['ph'], data['rainfall']]], dtype=float)
        model_type = data['model'].lower()
        
        # Validate model type
        if model_type not in ['rf', 'lstm']:
            return jsonify({'error': 'Invalid model type. Use "rf" or "lstm"'}), 400
        
        # Preprocess input
        features_scaled = scaler.transform(features)
        
        # Make prediction
        if model_type == 'rf':
            # Random Forest prediction
            pred_proba = rf_model.predict_proba(features_scaled)[0]
            pred_class = rf_model.predict(features_scaled)[0]
        else:
            # LSTM prediction
            features_tensor = torch.FloatTensor(features_scaled).reshape(1, 1, -1).to(device)
            with torch.no_grad():
                output = lstm_model(features_tensor)
                pred_proba = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Decode prediction
        predicted_crop = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred_proba))
        
        # Return response
        return jsonify({
            'prediction': predicted_crop,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)