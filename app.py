import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# -------------------------------
# Download Models (CORRECT WAY)
# -------------------------------
def download_model(file_id, output_name):
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(output_name):
        st.write(f"Downloading {output_name}...")
        gdown.download(url, output_name, quiet=False, fuzzy=True)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    mobilenet_id = "1jTroVKuF-e_Sb5AT5jj-W6wHIBejmsjz"
    resnet_id = "1fmA2GlwgevN8OjguASYb0pn2EZIFSnkw"
    cnn_id = "1_Dk072uJoSYXbMvft53nUZ97ExKVNm0W"

    download_model(mobilenet_id, "MobileNetV2.h5")
    download_model(resnet_id, "ResNet.h5")
    download_model(cnn_id, "CNN.h5")

    mobilenet_model = tf.keras.models.load_model("MobileNetV2.h5", compile=False)
    resnet_model = tf.keras.models.load_model("ResNet.h5", compile=False)
    cnn_model = tf.keras.models.load_model("CNN.h5", compile=False)

    return mobilenet_model, resnet_model, cnn_model


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

    if model_choice == "MobileNetV2":
        model = mobilenet_model
    elif model_choice == "ResNet50":
        model = resnet_model
    else:
        model = cnn_model

    height, width = model.input_shape[1], model.input_shape[2]
    image = image.resize((width, height))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
