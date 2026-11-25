# PULMO AI
### Pneumonia Detection from Chest X-ray Images Using Convolutional Neural Network (CNN)

<p align="center">
  <strong>A Deep Learning-Based Chest X-ray Image Classification Web Application</strong>
</p>

<p align="center">
  <a href="https://pulmo-ai.streamlit.app/">Live Application</a> •
  <a href="https://github.com/ARROWZZ-11/PALMO-AI">GitHub Repository</a>
</p>

---

## 1. Project Overview

PULMO AI is a deep learning-based web application developed to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project uses a Convolutional Neural Network (CNN), a deep learning architecture widely used in image classification and computer vision applications. The trained model analyzes an uploaded chest X-ray image and generates a prediction indicating whether the image belongs to the NORMAL or PNEUMONIA category.

The application provides a simple and user-friendly interface developed using Streamlit. Users can upload chest X-ray images and view the model's prediction and corresponding probability scores.

This project demonstrates the practical application of Artificial Intelligence, Deep Learning, Image Processing, and Web Application Development in the healthcare-related domain.

## 2. Project Objectives

The main objectives of PULMO AI are:

- To develop an image classification system using a Convolutional Neural Network.
- To classify chest X-ray images into NORMAL and PNEUMONIA categories.
- To understand the practical implementation of deep learning in medical image classification.
- To perform image preprocessing before model prediction.
- To evaluate the trained model using test data and classification metrics.
- To develop an interactive web application using Streamlit.
- To demonstrate the integration of a trained machine learning model into a web application.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and loading |
| CNN | Chest X-ray image classification |
| NumPy | Numerical operations and array processing |
| Pillow (PIL) | Image loading and preprocessing |
| Scikit-learn | Model evaluation and performance metrics |
| Streamlit | Web application development |
| Visual Studio Code | Development environment |
| GitHub | Source code hosting and version control |
| Streamlit Community Cloud | Web application deployment |

## 4. Project Methodology

The PULMO AI system follows a sequence of steps, from dataset preparation to final prediction.

### 4.1 Dataset Preparation

The project uses chest X-ray images organized into two categories:

- NORMAL
- PNEUMONIA

The images are divided into training and testing datasets. The training dataset is used to train the CNN model, while the testing dataset is used to evaluate its classification performance.

### 4.2 Image Preprocessing

Before classification, the uploaded X-ray image is processed to match the model's expected input format.

The preprocessing steps include:

- Loading the uploaded image.
- Correcting image orientation when necessary.
- Converting the image into grayscale.
- Resizing the image to 180 × 180 pixels.
- Converting the image into a numerical array.
- Preparing the image in the required input shape for the CNN model.

The trained model includes a rescaling layer that normalizes pixel values before further processing.

### 4.3 CNN Model Training

A Convolutional Neural Network is used to learn visual patterns from chest X-ray images.

The CNN architecture includes:

- Input Layer
- Rescaling Layer
- Data Augmentation
- Convolutional Layers
- Batch Normalization
- ReLU Activation
- Max Pooling Layers
- Global Average Pooling
- Fully Connected Dense Layer
- Dropout Layer
- Sigmoid Output Layer

The model learns image features through convolutional operations and uses these features to distinguish between the two image categories.

After training, the model is saved in the Keras format for future evaluation and application use.

### 4.4 Model Prediction

When a user uploads an X-ray image, the application preprocesses the image and passes it to the trained CNN model.

The model produces a prediction score representing the estimated probability of the PNEUMONIA class.

The application uses a classification threshold of 0.5 to determine the predicted category.

### 4.5 Result Display

The predicted category is displayed through the Streamlit web interface.

The application displays:

- Predicted class.
- Prediction confidence.
- NORMAL class probability.
- PNEUMONIA class probability.

## 5. System Workflow

The overall workflow of PULMO AI is:

1. User opens the PULMO AI web application.
2. User uploads a chest X-ray image.
3. The application reads the uploaded image.
4. The image is converted to grayscale and resized to 180 × 180 pixels.
5. The processed image is passed to the trained CNN model.
6. The model generates a prediction score.
7. The application determines the predicted category using the classification threshold.
8. The prediction and class probabilities are displayed on the screen.

## 6. Model Evaluation and Performance

The trained CNN model was evaluated using a test dataset containing **624 chest X-ray images**.

The following evaluation metrics were used:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall:** Measures how many actual cases of a particular class were correctly identified.
- **F1-Score:** Represents the balance between precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

### 6.1 Test Results

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 82.53% |
| Total Test Images | 624 |
| NORMAL Precision | 93.71% |
| NORMAL Recall | 57.26% |
| NORMAL F1-Score | 71.09% |
| PNEUMONIA Precision | 79.21% |
| PNEUMONIA Recall | 97.69% |
| PNEUMONIA F1-Score | 87.49% |

### 6.2 Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 134 | 100 |
| PNEUMONIA | 9 | 381 |

### 6.3 Performance Interpretation

The model achieved an overall test accuracy of 82.53% on the available test dataset.

The PNEUMONIA class achieved a recall of 97.69%, while the NORMAL class achieved a recall of 57.26%.

These results indicate that the model performed differently across the two classes. In particular, some NORMAL images were incorrectly classified as PNEUMONIA.

Therefore, evaluating class-wise precision, recall, and F1-score is important for understanding the model's performance rather than relying only on overall accuracy.

These results represent the performance of the evaluated model on the available test dataset and do not indicate clinical diagnostic accuracy.

## 7. Application Features

The main features of PULMO AI include:

- Simple and interactive web interface.
- Upload functionality for chest X-ray images.
- Support for JPG, JPEG, and PNG image formats.
- Automatic image preprocessing.
- CNN-based image classification.
- Classification into NORMAL and PNEUMONIA categories.
- Display of prediction confidence.
- Display of individual class probabilities.
- Web-based access through Streamlit Community Cloud.

## 8. Project Structure

The main development project is organized as follows:

```text
PULMO-AI/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── class_names.json
├── pneumonia_cnn_model.keras
├── README.md
│
└── data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### File Descriptions

| File / Folder | Description |
|---|---|
| `app.py` | Runs the Streamlit application and handles image upload, preprocessing, prediction, and result display. |
| `train_model.py` | Contains the code for training the CNN model. |
| `evaluate_model.py` | Evaluates the trained model using test data and generates performance metrics. |
| `requirements.txt` | Contains the Python libraries required to run the project. |
| `class_names.json` | Stores the class names used by the application. |
| `pneumonia_cnn_model.keras` | Contains the saved trained CNN model. |
| `data/` | Contains the training and testing image datasets. |

**Note:** The structure above describes the local development setup. The dataset folders and additional local model files may not be included in the public GitHub repository.

## 9. Installation and Setup

Follow these steps to run PULMO AI on a local computer.

### Step 1: Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/ARROWZZ-11/PALMO-AI.git
```

### Step 2: Open the Project Folder

```bash
cd PALMO-AI
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Required Libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

### Step 6: Open the Application

After executing the command, Streamlit provides a local URL in the terminal.

Open that URL in a web browser to access the PULMO AI application.

**Important:** The trained model file (`pneumonia_cnn_model.keras`) and `class_names.json` must be available in the expected project location. The dataset is required separately for model training and evaluation.

## 10. Deployment

PULMO AI is deployed using Streamlit Community Cloud and connected to its GitHub repository.

The deployment process includes:

1. Uploading the project source code to GitHub.
2. Connecting the GitHub repository to Streamlit Community Cloud.
3. Selecting `app.py` as the main application file.
4. Installing the required dependencies.
5. Deploying the application.
6. Accessing the application through its public URL.

**Live Application:**  
https://pulmo-ai.streamlit.app

## 11. Future Enhancements

The project can be improved further through the following enhancements:

- Increasing the size and diversity of the training dataset.
- Improving classification performance for both classes.
- Experimenting with different CNN architectures.
- Applying additional data augmentation techniques.
- Using a separate validation dataset for model selection.
- Performing more extensive model evaluation.
- Improving the user interface and result visualization.
- Exploring explainable AI techniques such as Grad-CAM to visualize image regions influencing predictions.

## 12. Limitations

Although PULMO AI demonstrates the use of deep learning for chest X-ray classification, it has certain limitations:

- The model's performance depends on the quality and diversity of the dataset.
- The current model does not achieve equally strong recall for both classes.
- The model may produce incorrect predictions.
- The model has not been clinically validated.
- Performance on images from different hospitals or imaging equipment may vary.
- The prediction is based on learned image patterns and does not replace professional medical assessment.

Therefore, the application should be considered an educational and experimental project rather than a clinically approved diagnostic system.

### Important Medical Disclaimer

PULMO AI is developed strictly for academic and educational purposes. It is not intended to replace professional medical advice, clinical examination, or diagnosis by qualified healthcare professionals. Its predictions must not be used to make medical decisions.

## 13. Conclusion

PULMO AI demonstrates the practical implementation of a Convolutional Neural Network for classifying chest X-ray images into NORMAL and PNEUMONIA categories.

By integrating Python, TensorFlow, Keras, NumPy, Scikit-learn, and Streamlit, the project combines deep learning model development with an interactive web application.

The project provides practical experience in dataset preparation, image preprocessing, CNN model training, performance evaluation, web application development, and cloud deployment.

Overall, PULMO AI represents an educational application of Artificial Intelligence and Deep Learning in the field of medical image classification.

---

## Project Information

**Project Name:** PULMO AI  
**Project Domain:** Artificial Intelligence and Deep Learning  
**Project Type:** Web-Based Image Classification Application  
**Model:** Convolutional Neural Network (CNN)  
**Programming Language:** Python  
**Application Framework:** Streamlit  
**Classification Categories:** NORMAL and PNEUMONIA  

**Live Demo:** https://pulmo-ai.streamlit.app  
**Source Code:** https://github.com/ARROWZZ-11/PALMO-AI

---

*Developed as an Academic Project.*
