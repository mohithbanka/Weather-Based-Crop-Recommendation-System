# Weather-Based Crop Recommendation System

## Overview
This project develops a machine learning-powered system to recommend optimal crops for farmers based on environmental and soil conditions. It analyzes seven key features—Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall—to predict the most suitable crop from 22 common varieties (e.g., rice, maize, apple, banana). The system uses Random Forest and LSTM (Long Short-Term Memory) models, leveraging GPU acceleration for training and a Flask-based RESTful API for real-time predictions. The goal is to promote sustainable agriculture by maximizing yield and minimizing resource waste.

## Dataset
- **Source**: The dataset is based on the "Crop Recommendation" dataset, augmented from ~2,200 to 10,000 records using Gaussian noise (2% variance) to simulate real-world variations while maintaining class balance.
- **Features**: 7 numerical features:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature (°C)
  - Humidity (%)
  - pH
  - Rainfall (mm)
- **Target**: Categorical label (22 crop classes, e.g., rice, maize, coffee).
- **Balance**: ~455 samples per class post-augmentation.
- **File**: `augmented_Crop_recommendation.csv`

## Models
Two models are trained for crop prediction:
- **Random Forest (RF)**: A Scikit-learn ensemble classifier with 100 trees, robust for tabular data.
- **LSTM**: A PyTorch-based recurrent neural network with 2 layers and 64 hidden units, designed to capture sequential patterns in environmental data.
- **Preprocessing**:
  - Features scaled using `StandardScaler`.
  - Labels encoded using `LabelEncoder`.
  - Train-test split: 80/20 (4,000 train, 1,000 test samples), stratified for class balance.
- **Performance**:
  - Random Forest: ~95% training accuracy, ~92% test accuracy.
  - LSTM: ~93% training accuracy, ~90% test accuracy.

## Project Structure
```
crop_recommendation/
├── data.ipynb                 # Jupyter notebook for data preprocessing and model training
├── app.py                    # Flask API for model deployment
├── augmented_Crop_recommendation.csv  # Augmented dataset
├── rf_model.pkl              # Trained Random Forest model
├── lstm_model.pth            # Trained LSTM model state dictionary
├── scaler.pkl                # StandardScaler object
├── label_encoder.pkl         # LabelEncoder object
├── README.md                 # Project documentation
```

## Requirements
- Python 3.11+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `torch`, `flask`, `matplotlib`, `seaborn`
- Hardware: GPU recommended (e.g., NVIDIA GTX 1650) for LSTM training.

Install dependencies:
```bash
pip install pandas numpy scikit-learn torch flask matplotlib seaborn
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mohithbanka/Weather-Based-Crop-Recommendation-System.git
   cd Weather-Based-Crop-Recommendation-System
   ```

2. **Prepare the Dataset**:
   - Ensure `augmented_Crop_recommendation.csv` is in the project directory.
   - If not available, generate it using the data augmentation script provided in the repository (or follow the notebook).

3. **Train Models**:
   - Open `data.ipynb` in Jupyter Notebook.
   - Run all cells to:
     - Load and preprocess the dataset.
     - Train Random Forest and LSTM models.
     - Save models (`rf_model.pkl`, `lstm_model.pth`) and preprocessing objects (`scaler.pkl`, `label_encoder.pkl`).
   - Expected output:
     ```
     RF Training Accuracy: ~0.95
     RF Test Accuracy: ~0.92
     LSTM Training Accuracy: ~0.93
     LSTM Test Accuracy: ~0.90
     ```

4. **Verify Saved Files**:
   - Ensure `rf_model.pkl`, `lstm_model.pth`, `scaler.pkl`, and `label_encoder.pkl` are in the project directory.

## Deployment
The system is deployed as a Flask API for real-time predictions.

1. **Run the Flask Server**:
   ```bash
   python app.py
   ```
   - The server runs on `http://127.0.0.1:5000`.
   - Output:
     ```
     * Serving Flask app 'app'
     * Debug mode: on
     * Running on http://0.0.0.0:5000
     ```

2. **Test the API**:
   - Send a POST request to `http://127.0.0.1:5000/predict` with JSON input:
     ```bash
     curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"N":90,"P":40,"K":40,"temperature":24.5,"humidity":80,"ph":6.5,"rainfall":200,"model":"lstm"}'
     ```
   - Or use Python:
     ```python
     import requests
     url = 'http://127.0.0.1:5000/predict'
     data = {
         "N": 90, "P": 40, "K": 40, "temperature": 24.5,
         "humidity": 80, "ph": 6.5, "rainfall": 200, "model": "lstm"
     }
     response = requests.post(url, json=data)
     print(response.json())
     ```
   - Expected output: `{"prediction":"rice","confidence":0.92}` (example).

3. **Troubleshooting**:
   - **ConnectionError**: Ensure the server is running before sending requests. Check if port 5000 is free:
     ```bash
     netstat -aon | findstr :5000
     ```
     If occupied, change the port in `app.py` (e.g., `app.run(host='0.0.0.0', port=5001)`).
   - **Model Loading Errors**: Verify the `LSTMModel` class in `app.py` matches the training architecture (input_size=7, hidden_size=64, num_layers=2, num_classes=22).
   - **Firewall**: Temporarily disable:
     ```bash
     netsh advfirewall set allprofiles state off
     ```
     Re-enable after testing: `netsh advfirewall set allprofiles state on`.

## Production Deployment
- **Ngrok**: For external access:
  ```bash
  ngrok http 5000
  ```
  Use the provided HTTPS URL (e.g., `https://abc123.ngrok.io/predict`).
- **Gunicorn**: For production:
  ```bash
  pip install gunicorn
  gunicorn -w 4 app:app
  ```
- **Docker**: Build and run:
  ```bash
  docker build -t crop-recommendation .
  docker run -p 5000:5000 crop-recommendation
  ```

## Results
- **Random Forest**: High accuracy due to robustness with tabular data.
  - Training Accuracy: ~95%
  - Test Accuracy: ~92%
- **LSTM**: Effective for capturing sequential patterns.
  - Training Accuracy: ~93%
  - Test Accuracy: ~90%
- **Visualizations**: Class distribution plots and confusion matrices generated in `data.ipynb` confirm balanced classes and high precision.

## Future Enhancements
- Add SHAP for feature importance explainability.
- Implement multi-crop ranking for alternative recommendations.
- Integrate real-time weather APIs for live data.
- Simulate climate change impacts on crop suitability.

## Contributing
Contributions are welcome! Please submit issues or pull requests on the GitHub repository: [Weather-Based-Crop-Recommendation-System](https://github.com/mohithbanka/Weather-Based-Crop-Recommendation-System).

## License
This project is licensed under the MIT License.