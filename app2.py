import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import torch
import numpy as np
from torch import nn

# Load data and models
df = pd.read_csv("Crop_recommendation.csv")
rf_model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Load LSTM model structure
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Load LSTM model
lstm_model = LSTMClassifier(1, 64, 1, len(label_encoder.classes_))
lstm_model.load_state_dict(torch.load("model/lstm_model.pth"))
lstm_model.eval()
lstm_model.to("cuda" if torch.cuda.is_available() else "cpu")

# App Title
st.title("🌾 Crop Recommendation System")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose View", ["Data Overview", "Model Performance", "Predict Crop"])

# Section: Data Overview
if option == "Data Overview":
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("📌 Feature Distributions")
    fig, ax = plt.subplots(figsize=(14, 8))
    df.drop('label', axis=1).hist(ax=ax)
    st.pyplot(fig)

    st.subheader("🌱 Crop Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='label', data=df, order=df['label'].value_counts().index, ax=ax)
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Section: Model Performance
elif option == "Model Performance":
    st.subheader("🎯 Random Forest Accuracy")
    st.text("See model training output in your Jupyter Notebook or CLI.")

    st.subheader("🎯 LSTM Accuracy")
    st.text("See model training output in your Jupyter Notebook or CLI.")

    st.info("You can display metrics or plots here after saving them during training.")

# Section: Prediction
else:
    st.subheader("🔍 Enter Input Data for Crop Recommendation")

    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.number_input("Potassium (K)", 0, 200, 40)
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 200.0)
    model_choice = st.selectbox("Choose Model", ["Random Forest", "LSTM (GPU)"])

    if st.button("🚀 Recommend Crop"):
        input_data = [N, P, K, temp, humidity, ph, rainfall]
        scaled = scaler.transform([input_data])

        if model_choice == "Random Forest":
            pred = rf_model.predict(scaled)[0]
            crop = label_encoder.inverse_transform([pred])[0]
        else:
            input_tensor = torch.tensor(scaled.reshape(1, 7, 1), dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                out = lstm_model(input_tensor)
                _, pred = torch.max(out, 1)
                crop = label_encoder.inverse_transform([pred.cpu().item()])[0]

        st.success(f"✅ Recommended Crop: **{crop}**")
