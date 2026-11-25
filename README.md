https://github.com/ARROWZZ-11/PALMO-AI.git
# 🫁 PULMO AI – Pneumonia Detection Using CNN

PULMO AI is a web-based image classification application that uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project is developed using Python, TensorFlow, Keras, and Streamlit for academic and educational purposes.

## 🚀 Live Demo

**Live Application:** https://pulmo-ai.streamlit.app

## ✨ Features

- Upload chest X-ray images.
- Classify images as NORMAL or PNEUMONIA.
- Display prediction confidence and class probabilities.
- User-friendly and responsive interface.
- CNN-based image classification.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Scikit-learn

## 🧠 Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Image Size: 180 × 180 pixels
- Image Format: Grayscale
- Output Layer: Sigmoid
- Classification: Binary Classification
- Classes: NORMAL and PNEUMONIA

## 📂 Project Structure

```text
PULMO-AI/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── pneumonia_cnn_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## 💻 Run Locally

**1. Clone the repository**

```bash
https://github.com/ARROWZZ-11/PALMO-AI.git
```

**2. Open the project folder**

```bash
cd PULMO-AI
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

## 📊 Model Performance

The model achieved **82.53% test accuracy** in the previously conducted evaluation.

Performance may vary depending on the input image and classification threshold.

## ⚠️ Disclaimer

PULMO AI is an academic and educational project. It is not a medically validated diagnostic system and must not be used as a substitute for professional medical advice or diagnosis.

## 👩‍💻 Project Purpose

Developed as an academic Machine Learning and Deep Learning project to explore CNN-based medical image classification.
