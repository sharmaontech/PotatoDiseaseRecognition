import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/1.keras")


model = load_model()

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("Plant Disease Recognition System")
    st.image("home_page.jpeg", use_container_width=True)

    st.markdown("""
    ---
    ### Welcome
    Welcome to the **Plant Disease Recognition System**, a smart tool built to detect **potato leaf diseases** using deep learning.

    ---
    ### Features
    - **Automated Detection** using advanced convolutional neural networks  
    - **Instant Results** with high confidence levels  
    - **Simple Interface** for effortless usability in labs and farms

    ---
    ### How It Works
    1. Go to the **Disease Recognition** tab  
    2. Upload a clear image of a potato leaf  
    3. View disease prediction and confidence instantly

    ---
    Whether you're a **farmer**, **agriculture student**, or **researcher**, this system is designed to support faster and more accurate diagnosis of crop health.
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")

    st.markdown("""
    ---
    ### Objective
    To deliver a reliable, real-time system for identifying **potato leaf diseases** using image classification techniques.

    ---
    ### Model Overview
    - **Input Size:** 256x256 RGB images  
    - **Architecture:** Convolutional Neural Network (CNN)  
    - **Dataset:** Thousands of labeled potato leaf images across three categories  
    - **Classes:**  
        - Early Blight  
        - Late Blight  
        - Healthy

    ---
    ### Why This Matters
    - **Manual diagnosis is slow and prone to error**  
    - Early and accurate detection prevents severe crop damage  
    - Offers value in both **research settings** and **field applications**

    This project demonstrates the power of combining deep learning with agricultural needs to improve efficiency and crop yield.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display image and prediction side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            # Preprocess and predict
            img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Assign color to prediction
            color_map = {
                "Healthy": "white",
                "Early Blight": "white",
                "Late Blight": "white"
            }
            prediction_color = color_map.get(predicted_class, "white")

            st.markdown(
                f"""
                <div style='
                    background-color: #2C2C36;
                    padding: 1.5rem;
                    border-radius: 10px;
                    border-left: 5px solid #2C2C36;
                    font-size: 1.3rem;
                    font-weight: 600;
                    color: white;
                '>
                    Prediction: <span style="color: {prediction_color};">{predicted_class}</span><br><br>
                    Confidence: <span style="color: white;">{confidence:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )