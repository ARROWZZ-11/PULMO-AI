
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PULMO AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. PROJECT SETTINGS
# ==========================================

MODEL_PATH = Path("pneumonia_cnn_model.keras")
CLASS_PATH = Path("class_names.json")

IMAGE_SIZE = (180, 180)

EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]

THRESHOLD = 0.5


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f4f8ff 0%,
        #edf4ff 50%,
        #f8fbff 100%
    );
    color: #172b4d;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #102a43,
        #183b56
    );
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.hero {
    background: linear-gradient(
        120deg,
        #123b63,
        #087eaa
    );
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(18, 59, 99, 0.15);
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e1f2ff;
    font-size: 16px;
}

.section-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e1eaf5;
    box-shadow: 0 4px 16px rgba(20, 55, 90, 0.06);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #dce8f5;
    text-align: center;
    box-shadow: 0 4px 14px rgba(20, 55, 90, 0.05);
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #087eaa;
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    margin-top: 5px;
}

.result-normal {
    background: #e8f8ef;
    border: 1px solid #a8e2bd;
    border-radius: 15px;
    padding: 22px;
    color: #166534;
}

.result-pneumonia {
    background: #fff1f0;
    border: 1px solid #ffc5c1;
    border-radius: 15px;
    padding: 22px;
    color: #a12622;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #087eaa,
        #125b91
    );
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #064d73;
    color: white;
}

.footer {
    text-align: center;
    padding: 25px 5px;
    color: #718096;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "pneumonia_cnn_model.keras not found!"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            "class_names.json not found!"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected class order in class_names.json"
        )

    if model.input_shape != (None, 180, 180, 1):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    if model.output_shape[-1] != 1:
        raise ValueError(
            f"Unexpected model output shape: {model.output_shape}"
        )

    return model, class_names


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("L")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model already contains Rescaling(1/255).
    # Therefore, do not divide by 255 here.

    return image_array


# ==========================================
# 6. PREDICTION
# ==========================================

def predict_image(model, image):

    image_array = preprocess_image(image)

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # DEBUG INFORMATION
    print("\n========== PULMO AI DEBUG ==========")

    print(
        "DEBUG - Image shape:",
        image_array.shape
    )

    print(
        "DEBUG - Pixel range:",
        image_array.min(),
        image_array.max()
    )

    print(
        "DEBUG - Raw prediction:",
        prediction
    )

    print("====================================\n")

    pneumonia_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    pneumonia_probability = float(
        np.clip(pneumonia_probability, 0.0, 1.0)
    )

    normal_probability = (
        1.0 - pneumonia_probability
    )

    if pneumonia_probability >= THRESHOLD:

        predicted_class = "PNEUMONIA"
        confidence = pneumonia_probability

    else:

        predicted_class = "NORMAL"
        confidence = normal_probability

    probabilities = {
        "NORMAL": normal_probability,
        "PNEUMONIA": pneumonia_probability
    }

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>🫁 PULMO AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "CNN-Based Pneumonia Detection"
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "X-Ray Analysis",
            "About the Model"
        ]
    )

    st.divider()

    st.caption(
        "Academic Machine Learning Project"
    )


# ==========================================
# 8. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to PULMO AI 🫁</h1>
        <p>
        An academic deep learning project that uses
        Convolutional Neural Networks (CNN) to classify
        chest X-ray images as NORMAL or PNEUMONIA.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    st.markdown("""
    <div class="section-card">
        <h3>About PULMO AI</h3>
        <p>
        PULMO AI is a CNN-based image classification
        application developed using Python and TensorFlow.
        It analyzes uploaded chest X-ray images and produces
        a model prediction.
        </p>
        <p>
        The application is designed for academic
        demonstration and learning purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">CNN</div>
            <div class="metric-label">Deep Learning Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">180×180</div>
            <div class="metric-label">Input Image Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2 Classes</div>
            <div class="metric-label">NORMAL / PNEUMONIA</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.info(
        "Navigate to X-Ray Analysis to test an image."
    )


# ==========================================
# 9. X-RAY ANALYSIS PAGE
# ==========================================

elif page == "X-Ray Analysis":

    st.markdown("""
    <div class="hero">
        <h1>X-Ray Analysis</h1>
        <p>
        Upload a chest X-ray image to obtain
        a CNN model prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:

        model, class_names = load_model()

    except Exception as error:

        st.error(
            f"Unable to load the CNN model: {error}"
        )

        st.stop()

    st.markdown(
        "### Upload Chest X-Ray Image"
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file)

            col1, col2 = st.columns(
                [1, 1],
                gap="large"
            )

            with col1:

                st.markdown(
                    '<div class="section-card">'
                    '<h4>Uploaded X-Ray</h4>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    caption="Uploaded Chest X-Ray",
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    '<div class="section-card">'
                    '<h4>Image Information</h4>',
                    unsafe_allow_html=True
                )

                st.write(
                    "**File name:**",
                    uploaded_file.name
                )

                st.write(
                    "**Image format:**",
                    image.format
                )

                st.write(
                    "**Image dimensions:**",
                    f"{image.width} × {image.height}"
                )

                st.write(
                    "**Model input:**",
                    "180 × 180 Grayscale"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            if st.button(
                "🔍 Analyze X-Ray",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing X-ray image..."
                ):

                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_image(
                        model,
                        image
                    )

                st.markdown("---")

                st.markdown(
                    "## Prediction Result"
                )

                if predicted_class == "NORMAL":

                    st.markdown(f"""
                    <div class="result-normal">
                        <h2>🟢 NORMAL</h2>
                        <h3>Confidence: {confidence * 100:.2f}%</h3>
                        <p>
                        The model classified this image
                        as NORMAL.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown(f"""
                    <div class="result-pneumonia">
                        <h2>🔴 PNEUMONIA</h2>
                        <h3>Confidence: {confidence * 100:.2f}%</h3>
                        <p>
                        The model classified this image
                        as PNEUMONIA.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    "### Class Probability Scores"
                )

                for class_name in class_names:

                    probability = probabilities[class_name]

                    st.write(
                        f"**{class_name}:** "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(probability)

                st.warning(
                    "This is an AI model prediction, "
                    "not a medical diagnosis. Please consult "
                    "a qualified medical professional."
                )

        except Exception as error:

            st.error(
                f"Unable to process this image: {error}"
            )


# ==========================================
# 10. ABOUT THE MODEL PAGE
# ==========================================

elif page == "About the Model":

    st.markdown("""
    <div class="hero">
        <h1>About the CNN Model</h1>
        <p>
        Technical information about the PULMO AI
        deep learning architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Specifications")

    specifications = {
        "Model Type": "Convolutional Neural Network (CNN)",
        "Framework": "TensorFlow / Keras",
        "Input Image Size": "180 × 180 pixels",
        "Color Mode": "Grayscale",
        "Output Layer": "Single Sigmoid Neuron",
        "Loss Function": "Binary Crossentropy",
        "Optimizer": "Adam",
        "Classes": "NORMAL and PNEUMONIA"
    }

    for key, value in specifications.items():

        st.write(f"**{key}:** {value}")

    st.markdown("---")

    st.markdown("### CNN Architecture")

    st.write("""
    1. Input Layer
    2. Rescaling Layer
    3. Data Augmentation
    4. Convolutional Layer (32 filters)
    5. Convolutional Layer (64 filters)
    6. Convolutional Layer (128 filters)
    7. Convolutional Layer (256 filters)
    8. Global Average Pooling
    9. Dense Layer (128 neurons)
    10. Dropout Layer
    11. Output Layer (Sigmoid)
    """)

    st.markdown("---")

    st.markdown("### Important Information")

    st.info(
        "The model's actual performance must be "
        "reported using the latest evaluation results. "
        "No accuracy value is assumed or hardcoded here."
    )

    st.warning(
        "PULMO AI is an academic demonstration project. "
        "It is not approved or validated for clinical use."
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    <hr>
    <p>
    PULMO AI | Pneumonia Detection Using CNN
    </p>
    <p>
    Developed for academic and educational purposes.
    </p>
    <p>
    Not intended for medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)
