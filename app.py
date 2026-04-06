import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------
# Load model (correct)
# -------------------
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNetV2":
        return tf.keras.models.load_model("MobileNetV2_model.h5")
    elif model_name == "ResNet50":
        return tf.keras.models.load_model("ResNet_model.h5")
    else:
        return tf.keras.models.load_model("leaf_model.h5")

# -------------------
# UI
# -------------------
st.title("🌿 Palm Oil Leaf Nutrient Stress Detection")

model_choice = st.selectbox(
    "Select Model",
    ["MobileNetV2", "ResNet50", "CNN"]
)

model = load_model(model_choice)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# -------------------
# Prediction
# -------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # ✅ set correct size
    if model_choice in ["MobileNetV2", "ResNet50"]:
        size = (224, 224)
    else:
        size = (150, 150)

    # ✅ preprocessing
    image = image.resize(size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # ✅ predict
    prediction = model.predict(image)

    class_names = ["Healthy", "boron", "kalium", "mg", "nitrogen"]

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # -------------------
    # Output
    # -------------------
    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
