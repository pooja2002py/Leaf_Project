import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# -------------------------------
# DOWNLOAD MODEL (FIXED)
# -------------------------------
def download_model(file_id, output_name):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Download if file missing OR corrupted
    if not os.path.exists(output_name) or os.path.getsize(output_name) < 1000000:
        st.warning(f"Downloading {output_name}...")
        gdown.download(url, output_name, quiet=False)

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    # 🔥 PUT YOUR FILE IDs HERE
    mobilenet_id = "1jTroVKuF-e_Sb5AT5jj-W6wHIBejmsjz"
    cnn_id = "1JYRGz34z72QOn3B1cSb5_4Gu5KCTVRRM"

    # Download
    download_model(mobilenet_id, "MobileNetV2.h5")
    download_model(cnn_id, "CNN.h5")

    # Load safely
    try:
        mobilenet_model = tf.keras.models.load_model("MobileNetV2.h5", compile=False)
        cnn_model = tf.keras.models.load_model("CNN.h5", compile=False)
    except Exception as e:
        st.error("❌ Model loading failed. File may be corrupted.")
        st.stop()

    return mobilenet_model, cnn_model

# -------------------------------
# LOAD MODELS
# -------------------------------
mobilenet_model, cnn_model = load_models()

# -------------------------------
# CLASS NAMES
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

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Select model
    if model_choice == "MobileNetV2":
        model = mobilenet_model
    else:
        model = cnn_model

    # 🔥 AUTO INPUT SIZE
    try:
        height, width = model.input_shape[1], model.input_shape[2]
    except:
        height, width = 224, 224

    # Resize
    image = image.resize((height, width))

    # Preprocess
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    try:
        prediction = model.predict(image)
    except Exception as e:
        st.error("❌ Prediction failed. Input shape mismatch.")
        st.stop()

    # Output
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # Probabilities
    st.subheader("Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
