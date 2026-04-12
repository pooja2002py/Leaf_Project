import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -------------------------------
# Load Models (FROM GITHUB FILES)
# -------------------------------
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(__file__)

    mobilenet_path = os.path.join(BASE_DIR, "MobileNetV2_model.h5")
    resnet_path = os.path.join(BASE_DIR, "ResNet_model.h5")
    cnn_path = os.path.join(BASE_DIR, "leaf_model.h5")

    mobilenet_model = tf.keras.models.load_model(mobilenet_path, compile=False)
    resnet_model = tf.keras.models.load_model(resnet_path, compile=False)
    cnn_model = tf.keras.models.load_model(cnn_path, compile=False)

    return mobilenet_model, resnet_model, cnn_model


# Load models
mobilenet_model, resnet_model, cnn_model = load_models()

# -------------------------------
# Class Names
# -------------------------------
class_names = ["Healthy", "boron", "kalium", "mg", "nitrogen"]

# -------------------------------
# UI
# -------------------------------
st.title("🌿 Palm Oil Leaf Nutrient Stress Detection")

model_choice = st.selectbox(
    "Select Model",
    ["MobileNetV2", "ResNet50", "CNN"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Select model
    if model_choice == "MobileNetV2":
        model = mobilenet_model
    elif model_choice == "ResNet50":
        model = resnet_model
    else:
        model = cnn_model

    # Resize based on model input
    height, width = model.input_shape[1], model.input_shape[2]
    image = image.resize((height, width))

    # Preprocess
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
