# 🌾 Weather-Based Crop Recommendation System

## 🚀 Description

This project helps farmers and agricultural advisors recommend the **most suitable crop** based on environmental and soil parameters. It uses **machine learning (Random Forest)** and **deep learning (LSTM with CUDA)** models to predict the optimal crop to grow. The app includes a **Flask-based web interface** and **interactive visualizations**.

---

## 📌 Features

- 🌱 User inputs: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall  
- 🧠 Two model choices: Random Forest or GPU-enabled LSTM  
- 📊 Visualizations: Histograms, crop class distribution, correlation heatmap  
- 🖥️ UI: Beautiful, responsive web interface using Flask  
- 🌐 One-click deployment on [Render.com](https://render.com)

---

## 📁 Project Structure

```
crop-recommendation-app/
├── app.py                   # Flask backend
├── model/                   # Trained models & scalers
│   ├── rf_model.pkl
│   ├── lstm_model.pth
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── static/
│   └── style.css            # CSS styling
├── templates/
│   ├── index.html           # Home page form
│   └── visualize.html       # Visualization dashboard
├── Crop_recommendation.csv  # Source dataset
├── requirements.txt         # Python dependencies
└── render.yaml              # Deployment config
```

---

## 💻 Installation & Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/crop-recommendation-app.git
cd crop-recommendation-app

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Go to `http://127.0.0.1:5000` in your browser.

---

## 🌐 Deployment on Render

1. Push your code to GitHub  
2. Go to [https://render.com](https://render.com)  
3. Click **New Web Service** → Connect your GitHub repo  
4. In Render settings:  
   - **Start Command**: `python app.py`  
   - **Build Command**: *(leave blank)*  
   - **Environment**: Python 3  
5. Wait for deployment & access your live app via the public URL provided

---

## 🔍 Model Info

- **Random Forest**: Sklearn-based, high accuracy, fast inference  
- **LSTM**: PyTorch-based deep learning model, designed for sequence learning  
- **Data Scaling**: `StandardScaler`  
- **Label Encoding**: `LabelEncoder` for crop classes

---

## 📊 Visualizations

- Feature histograms  
- Crop class distribution  
- Correlation heatmap  

> Access them at `/visualize` in the web app

---

## 🛠 Requirements

- Python 3.8+  
- Flask  
- pandas, seaborn, matplotlib  
- scikit-learn  
- torch  
- joblib

---

## 📸 Screenshots

You can include screenshots or screen recordings of:
- The home form
- The prediction output
- The visualizations page

---

## 👨‍💻 Author

Built by **[Your Name]**  
Guided by: **[Your Faculty / Mentor]**  
Deployed via: [Render.com](https://render.com)

---

## 📜 License

This project is for educational use. For production deployment, please review dependencies and security guidelines.
