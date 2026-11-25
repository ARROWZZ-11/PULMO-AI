import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================
# 1. PROJECT SETTINGS
# =====================================

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

MODEL_PATH = "pneumonia_cnn_model.keras"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================================
# 2. CHECK DATASET
# =====================================

for directory in [TRAIN_DIR, TEST_DIR]:

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Folder not found: {directory}"
        )

    for class_name in CLASS_NAMES:

        path = os.path.join(directory, class_name)

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing folder: {path}"
            )

# =====================================
# 3. COUNT TRAINING IMAGES
# =====================================

valid_extensions = (
    ".jpg", ".jpeg", ".png", ".bmp"
)

image_counts = {}

for class_name in CLASS_NAMES:

    class_path = os.path.join(
        TRAIN_DIR, class_name
    )

    image_counts[class_name] = sum(
        1 for filename in os.listdir(class_path)
        if filename.lower().endswith(valid_extensions)
    )

print("\nTraining image counts:")
print(image_counts)

# =====================================
# 4. CALCULATE CLASS WEIGHTS
# =====================================

total_images = sum(image_counts.values())

class_weights = {}

for index, class_name in enumerate(CLASS_NAMES):

    count = image_counts[class_name]

    if count == 0:
        raise ValueError(
            f"No images found in {class_name}"
        )

    class_weights[index] = (
        total_images / (2 * count)
    )

print("\nClass weights:")
print(class_weights)

# =====================================
# 5. LOAD DATASET
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

if train_ds.class_names != CLASS_NAMES:
    raise ValueError("Class order mismatch!")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================
# 6. DATA AUGMENTATION
# =====================================

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.05)

])

# =====================================
# 7. BUILD CNN MODEL
# =====================================

model = models.Sequential([

    layers.Input(shape=(180, 180, 1)),

    layers.Rescaling(1.0 / 255),

    data_augmentation,

    layers.Conv2D(32, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")

])

# =====================================
# 8. COMPILE MODEL
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

)

model.summary()

# =====================================
# 9. CALLBACKS
# =====================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# =====================================
# 10. TRAIN MODEL
# =====================================

print("\nStarting CNN training...\n")

print("CLASS WEIGHTS:", class_weights)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop
    ]

)

# =====================================
# 11. LOAD BEST MODEL
# =====================================

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

# =====================================
# 12. SAVE CLASS NAMES
# =====================================

with open("class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file)

# =====================================
# 13. VALIDATION EVALUATION
# =====================================

print("\nEvaluating on validation dataset...\n")

val_results = best_model.evaluate(
    val_ds,
    return_dict=True
)

for metric, value in val_results.items():
    print(f"Validation {metric}: {value:.4f}")

# =====================================
# 14. TEST EVALUATION
# =====================================

print("\nEvaluating on test dataset...\n")

test_results = best_model.evaluate(
    test_ds,
    return_dict=True
)

for metric, value in test_results.items():
    print(f"Test {metric}: {value:.4f}")

# =====================================
# 15. CONFUSION MATRIX
# =====================================

actual_labels = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
]).astype(int)

prediction_scores = best_model.predict(
    test_ds,
    verbose=1
).ravel()

predicted_labels = (
    prediction_scores >= 0.5
).astype(int)

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(matrix)

# =====================================
# 16. CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# =====================================
# 17. COMPLETION
# =====================================

print("\nCNN Training Completed!")

print("Model saved:", MODEL_PATH)
print("Class names saved: class_names.json")