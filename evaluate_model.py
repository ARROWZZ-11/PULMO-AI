
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Dataset settings
TEST_DIR = "data/test"
MODEL_PATH = "pneumonia_cnn_model.keras"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

# Check model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print("Class order:", class_names)

# Actual labels
actual_labels = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
]).astype(int)

# Model predictions
prediction_scores = model.predict(test_ds, verbose=1).ravel()

predicted_labels = (
    prediction_scores >= 0.5
).astype(int)

# Accuracy
accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Confusion matrix
matrix = confusion_matrix(
    actual_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(matrix)

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)

# Threshold analysis
print("\nThreshold Analysis:")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    predictions = (
        prediction_scores >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1]
    )

    normal_recall = (
        matrix[0, 0] / matrix[0].sum()
    )

    pneumonia_recall = (
        matrix[1, 1] / matrix[1].sum()
    )

    acc = accuracy_score(
        actual_labels,
        predictions
    )

    print(f"\nThreshold: {threshold}")

    print(f"Accuracy: {acc * 100:.2f}%")

    print(
        f"NORMAL Recall: {normal_recall * 100:.2f}%"
    )

    print(
        f"PNEUMONIA Recall: {pneumonia_recall * 100:.2f}%"
    )

print("\nEvaluation completed!")
