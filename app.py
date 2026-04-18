import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -------------------------------
# Load Selected Model Only (Efficient)
# -------------------------------
@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "MobileNetV2":
            return tf.keras.models.load_model("MobileNetV2.h5", compile=False)
        elif model_name == "ResNet50":
            return tf.keras.models.load_model("ResNet.h5", compile=False)
        else:
            return tf.keras.models.load_model("CNN.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

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

# Load only selected model
model = load_model(model_choice)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None and model is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # Get input shape
        height, width = model.input_shape[1], model.input_shape[2]

        # ✅ Correct resize (width, height)
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

    except Exception as e:
        st.error(f"Prediction failed: {e}")
