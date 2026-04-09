# Leaf_Project
# 🌿 Palm Oil Leaf Nutrient Stress Detection using CNN

---

## 📌 1. Project Title

**Palm Oil Leaf Nutrient Stress Detection using Convolutional Neural Network (CNN)**

---

## 📖 2. Introduction

This project focuses on detecting nutrient stress in palm oil leaves using a Convolutional Neural Network (CNN). The model is trained to classify different nutrient deficiencies such as **boron, nitrogen, kalium, magnesium**, and also identify **healthy leaves**. This system helps in early detection of plant health issues and supports better agricultural management.

---

## 🎯 3. Objective

* To build a CNN model for palm oil leaf classification
* To detect nutrient stress conditions in leaves
* To improve model performance using balanced dataset
* To evaluate model using confusion matrix and classification report

---

## 📊 4. Dataset Details

* **Total Images:** ~14,000+
* **Classes:**

  * boron
  * Healthy
  * kalium
  * mg
  * nitrogen

### 📂 Dataset Split:

* Training: 80%
* Validation: 10%
* Testing: 10%

---

## ⚙️ 5. Data Preprocessing

* Resized images to **224 × 224**
* Applied **data augmentation**:

  * Rotation
  * Zoom
  * Horizontal flip
* Balanced dataset using augmentation techniques
* Normalized pixel values (rescale = 1./255)

---

## 🧠 6. Model Architecture (CNN)

The CNN model consists of:

* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout Layer (0.5)
* Output Layer (Softmax with 5 classes)

---

## ⚡ 7. Training Details

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 32
* **Epochs:** 25
* **Callback:** EarlyStopping (to prevent overfitting)

---

## 📈 8. Results

* Achieved **~85–90% accuracy** on test dataset
* Evaluation methods:

  * Confusion Matrix
  * Classification Report (Precision, Recall, F1-score)

---

## 📊 9. Visualizations

* Training vs Validation Accuracy Graph
* Training vs Validation Loss Graph
* Confusion Matrix Heatmap

---

## ⚠️ 10. Challenges Faced

* Class imbalance in dataset (initial stage)
* Similar visual symptoms between nutrient deficiencies:

  * kalium vs magnesium
  * nitrogen vs boron
* Model confusion due to overlapping patterns

---

## ✅ 11. Conclusion

The CNN model successfully detects nutrient stress in palm oil leaves with good accuracy. Dataset balancing and data augmentation improved performance significantly. This system can assist in early diagnosis of plant health issues.

---

## 🚀 12. Future Work

* Use **Transfer Learning (MobileNetV2 / ResNet50)**
* Improve dataset quality and labeling
* Deploy as a **web application using Streamlit**
* Enable real-time leaf detection using camera

---

## 📁 13. Project Structure

```
project/
├── train/
├── val/
├── test/
├── model.h5
├── code.py
└── README.md
```

---

✨ **End of Documentation**
