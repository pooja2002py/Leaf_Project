import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -----------------------
# STEP 1: TITLE
# -----------------------
st.title("🌿 Palm Oil Leaf Nutrient Stress Detection")

# -----------------------
# STEP 2: SELECT MODEL
# -----------------------
model_choice = st.selectbox(
    "Select Model",
    ["MobileNetV2", "ResNet50", "CNN"]
)

# -----------------------
# STEP 3: UPLOAD IMAGE
# -----------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# -----------------------
# STEP 4: IF IMAGE UPLOADED
# -----------------------
if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # -----------------------
    # STEP 5: SELECT MODEL PATH + SIZE
    # -----------------------
    if model_choice == "MobileNetV2":
        model_path = "MobileNetV2_model.h5"
        size = (224, 224)

    elif model_choice == "ResNet50":
        model_path = "ResNet_model.h5"
        size = (224, 224)

    else:
        model_path = "leaf_model.h5"
        size = (150, 150)

    # -----------------------
    # STEP 6: CHECK FILE EXISTS
    # -----------------------
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    # -----------------------
    # STEP 7: LOAD MODEL
    # -----------------------
    model = tf.keras.models.load_model(model_path)

    # -----------------------
    # STEP 8: PREPROCESS IMAGE
    # -----------------------
    image = image.resize(size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # -----------------------
    # STEP 9: PREDICT
    # -----------------------
    prediction = model.predict(image)

    class_names = ["Healthy", "boron", "kalium", "mg", "nitrogen"]

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # -----------------------
    # STEP 10: OUTPUT
    # -----------------------
    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
