import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load Models
mobilenet_model = tf.keras.models.load_model("MobileNetV2_model.h5")
resnet_model = tf.keras.models.load_model("ResNet_model.h5")
cnn_model = tf.keras.models.load_model("leaf_model.h5")

# Class Names
class_names = ["Healthy", "boron", "kalium", "mg", "nitrogen"]

# UI design
st.title("🌿Palm Oil Leaf Nutrient Stress Detection")

model_choice = st.selectbox(
    "Select Model",
    ["MobileNetV2", "ResNet50", "CNN"]
)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# Prediction
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Select model (FIXED)
    if model_choice == "MobileNetV2":
        model = mobilenet_model
        size = (224, 224)
    elif model_choice == "ResNet50":
        model = resnet_model
        size = (224, 224)
    else:
        model = cnn_model
        size = (150, 150)   

    # Preprocess
    image = Image.open(uploaded_file).convert('RGB')              # ensure 3 channels
    image = image.resize((224, 224))
    image=np.array(image)
    image=image/255.0
    image=np.expand_dims(image,axis=0)
    print("FINAL SHAPE:",image,image.shape)
    prediction=model.predict(image) 

    # Prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # 🔥 ADD PROBABILITIES
    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")