"""
AI-Based Waste Classification System
Loads a Teachable Machine model (keras_model.h5) and classifies an input image
into one of the trained categories (Plastic, Paper, Metal, Organic).
"""

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "plastic_bottle.jpg.jpg"   # change this to your test image
IMG_SIZE = (224, 224)


def load_labels(path):
    """Read class names from labels.txt (format: '0 Plastic', '1 Paper', ...)"""
    with open(path, "r") as f:
        lines = f.readlines()
    # Teachable Machine labels.txt has "index classname" per line
    labels = [line.strip().split(" ", 1)[1] for line in lines]
    return labels


def preprocess_image(image_path):
    """Resize and normalize image to match Teachable Machine's training format."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    normalized_array = (image_array / 127.5) - 1  # scale to [-1, 1]

    # Model expects shape: (1, 224, 224, 3)
    input_data = np.expand_dims(normalized_array, axis=0)
    return input_data


def predict(model, labels, image_path):
    input_data = preprocess_image(image_path)
    predictions = model.predict(input_data)[0]  # shape: (num_classes,)

    top_index = np.argmax(predictions)
    class_name = labels[top_index]
    confidence = predictions[top_index] * 100

    return class_name, confidence


def main():
    model = load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    class_name, confidence = predict(model, labels, IMAGE_PATH)

    print(f"Predicted Class : {class_name}")
    print(f"Confidence      : {confidence:.2f}%")


if __name__ == "__main__":
    main()
