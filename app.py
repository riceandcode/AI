"""
AI-Based Waste Classification System — Streamlit Web App
Environmental Studies CA1 Project

Loads a Teachable Machine model (keras_model.h5) and classifies an image
into: Plastic, Paper, Metal, or Organic. Accepts either an uploaded image
or a live webcam photo, and shows eco-friendly disposal tips.
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

# Eco-tips shown after prediction, keyed by class name.
# Edit these freely to match your project's tone.
ECO_TIPS = {
    "Plastic": "Rinse and place in the Blue Bin. Avoid single-use plastics where you can!",
    "Paper": "Keep it dry and flatten before recycling. Remove any plastic tape or lamination.",
    "Metal": "Rinse cans before recycling. Metals like aluminium can be recycled endlessly without losing quality.",
    "Organic": "Compost it if possible. Organic waste in landfills produces methane, a potent greenhouse gas.",
}

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ AI-Based Waste Classification System")
st.subheader("Snap or upload a photo — let AI sort it into Plastic, Paper, Metal, or Organic")
st.markdown(
    "*An Environmental Studies project demonstrating how AI can support "
    "smarter waste segregation and recycling.*"
)
st.divider()


# ---------- MODEL LOADING (cached so it only loads once) ----------
@st.cache_resource
def load_classification_model():
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    return model, labels


model, labels = load_classification_model()


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(image: Image.Image):
    """Resize and normalize image to match the model's expected input."""
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_array = (image_array / 127.5) - 1  # scale to [-1, 1]
    return np.expand_dims(normalized_array, axis=0)


# ---------- PREDICTION ----------
def predict(image: Image.Image):
    input_data = preprocess_image(image)
    predictions = model.predict(input_data)[0]
    top_index = np.argmax(predictions)
    class_name = labels[top_index]
    confidence = predictions[top_index] * 100
    return class_name, confidence


# ---------- INPUT OPTIONS ----------
st.markdown("### Step 1: Provide an Image")
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Webcam"])

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input("Take a photo of the waste item")
    if camera_file is not None:
        input_image = Image.open(camera_file)

st.divider()

# ---------- DISPLAY + CLASSIFY ----------
st.markdown("### Step 2: Result")

if input_image is None:
    # Graceful handling when no image has been provided yet
    st.info("Upload an image or take a photo above to get a prediction.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.image(input_image, caption="Input Image", use_container_width=True)

    with col2:
        try:
            class_name, confidence = predict(input_image)

            st.metric(label="Predicted Category", value=class_name)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

            if confidence < 60:
                st.warning("Low confidence — try a clearer or closer photo for a more reliable result.")

        except Exception as e:
            st.error(f"Something went wrong while classifying the image: {e}")
            class_name = None

    # Eco-tip shown below, full width
    if input_image is not None and "class_name" in locals() and class_name in ECO_TIPS:
        st.success(f"**Disposal Tip ({class_name}):** {ECO_TIPS[class_name]}")

st.divider()
st.caption("Built for Environmental Studies CA1 — Waste Classification using Teachable Machine + TensorFlow")
