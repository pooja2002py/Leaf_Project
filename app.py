import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load Models (NO DOWNLOAD)
# -------------------------------
@st.cache_resource
def load_models():
    mobilenet_model = tf.keras.models.load_model("MobileNetV2.h5", compile=False)
    cnn_model = tf.keras.models.load_model("CNN.h5", compile=False)
    return mobilenet_model, cnn_model

mobilenet_model, cnn_model = load_models()

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
    ["MobileNetV2", "CNN"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Select model
    model = mobilenet_model if model_choice == "MobileNetV2" else cnn_model

    # Auto resize
    height, width = model.input_shape[1], model.input_shape[2]
    image = image.resize((height, width))

    # Preprocess
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)

    st.success(f"Prediction: {class_names[np.argmax(prediction)]}")
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")

    st.subheader("Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
