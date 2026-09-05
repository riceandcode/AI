"""
AI-Based Waste Classification System — Streamlit Web App
Environmental Studies CA1 Project
"""

import numpy as np
import streamlit as st
from PIL import Image

# Import directly from tf_keras (handles Teachable Machine H5 files natively)
from tf_keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

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


# ---------- MODEL LOADING ----------
@st.cache_resource
def load_classification_model():
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        labels = [
            line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip()
            for line in f.readlines()
        ]
    return model, labels


model, labels = load_classification_model()


# ---------- PREPROCESSING & INFERENCE ----------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_array = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized_array, axis=0)


def predict(image: Image.Image):
    input_data = preprocess_image(image)
    predictions = model.predict(input_data)[0]
    top_index = int(np.argmax(predictions))
    class_name = labels[top_index]
    confidence = float(predictions[top_index]) * 100.0
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
    st.info("Upload an image or take a photo above to get a prediction.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.image(input_image, caption="Input Image", use_container_width=True)

    class_name = None

    with col2:
        try:
            with st.spinner("Analyzing material..."):
                class_name, confidence = predict(input_image)

            st.metric(label="Predicted Category", value=class_name)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

            if confidence < 60:
                st.warning("Low confidence — try a clearer photo for a more reliable result.")

        except Exception as e:
            st.error(f"Error during classification: {e}")

    if class_name:
        matched_tip = None
        for key in ECO_TIPS:
            if key.lower() in class_name.lower():
                matched_tip = ECO_TIPS[key]
                break

        if matched_tip:
            st.success(f"**Disposal Tip ({class_name}):** {matched_tip}")
        else:
            st.info("🔵 Place recyclables in designated bins and segregate wet organic waste.")

st.divider()
st.caption("Built for Environmental Studies CA1 — Waste Classification using Teachable Machine")