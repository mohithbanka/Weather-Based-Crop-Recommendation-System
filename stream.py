import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="wide")

# Title and description
st.title("🌱 Weather-Based Crop Recommendation System")
st.markdown(
    """
    Enter soil and weather parameters to get personalized crop recommendations. 
    The system uses machine learning models (Random Forest or LSTM) trained on augmented data for 22 crop types.
    """
)

# Sidebar for inputs
st.sidebar.header("Input Parameters")
N = st.sidebar.slider("Nitrogen (N) Level", 0.0, 200.0, 90.0)
P = st.sidebar.slider("Phosphorus (P) Level", 0.0, 200.0, 40.0)
K = st.sidebar.slider("Potassium (K) Level", 0.0, 200.0, 40.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 24.5)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 200.0)

# Model selection
model_type = st.sidebar.selectbox("Select Model", ["rf", "lstm"])

# API URL (update with your deployed backend URL, e.g., Render URL)
API_URL = "http://127.0.0.1:5000/predict"  # Change to your Render backend URL after deployment

# Prediction button
if st.sidebar.button("Predict Crop", type="primary"):
    # Prepare data
    data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "model": model_type
    }
    
    try:
        # Call API
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Display results
        st.success("Prediction Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommended Crop", result["prediction"])
        with col2:
            st.metric("Confidence Score", f"{result['confidence']:.2%}")
        
        # Additional info
        st.info(f"**Model Used:** {model_type.upper()}")
        st.caption("This recommendation is based on soil nutrients and weather conditions. Consult local experts for final decisions.")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Ensure the backend Flask server is running and the API URL is correct.")
    except KeyError as e:
        st.error(f"Unexpected API response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    Built with Streamlit | Models: Random Forest & LSTM | Dataset: Augmented Crop Recommendation
    """
)