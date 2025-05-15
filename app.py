from flask import Flask, render_template, request
import pandas as pd
import joblib
import torch
import numpy as np
from torch import nn
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load data and models
df = pd.read_csv("Crop_recommendation.csv")
rf_model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Load LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

lstm_model = LSTMClassifier(1, 64, 1, len(label_encoder.classes_))
lstm_model.load_state_dict(torch.load("model/lstm_model.pth"))
lstm_model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lstm_model.to(device)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Check if visualizations exist to avoid regenerating
def generate_visualizations():
    if not all(os.path.exists(f'static/{img}') for img in ['histogram.png', 'crop_dist.png', 'heatmap.png']):
        # Create and save histogram
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(df.drop('label', axis=1).columns):
            axes[i].hist(df[col], bins=20, color='#4CAF50', edgecolor='black')
            axes[i].set_title(col, fontsize=10)
        plt.tight_layout()
        plt.savefig('static/histogram.png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure

        # Crop distribution
        fig = plt.figure(figsize=(12, 6))
        sns.countplot(y='label', data=df, order=df['label'].value_counts().index, palette='viridis')
        plt.title("Crop Distribution", fontsize=14, pad=10)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Crop", fontsize=12)
        plt.tight_layout()
        plt.savefig('static/crop_dist.png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure

        # Heatmap
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10})
        plt.title("Feature Correlation Heatmap", fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig('static/heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/visualize')
def visualize():
    generate_visualizations()  # Generate visualizations only if they don't exist
    return render_template("visualize.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        model_type = data['model']

        features = [N, P, K, temperature, humidity, ph, rainfall]
        scaled = scaler.transform([features])

        if model_type == "rf":
            pred = rf_model.predict(scaled)[0]
            crop = label_encoder.inverse_transform([pred])[0]
        else:
            tensor_input = torch.tensor(scaled.reshape(1, 7, 1), dtype=torch.float32).to(device)
            with torch.no_grad():
                output = lstm_model(tensor_input)
                _, pred = torch.max(output, 1)
                crop = label_encoder.inverse_transform([pred.cpu().item()])[0]

        return {"recommended_crop": crop}
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True)