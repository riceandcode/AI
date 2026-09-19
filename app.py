"""
AI-Based Waste Classification System — Streamlit Web App
Environmental Studies CA1 Project
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
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
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4fbf4 0%, #edf7ff 52%, #fdf7ec 100%);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    .hero-card {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(15, 118, 110, 0.12);
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.4rem;
    }
    .hero-pill {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: clamp(2.2rem, 4vw, 3.2rem);
        font-weight: 800;
        margin: 0 0 0.5rem;
        color: #0f172a;
    }
    .hero-subtitle {
        font-size: 1.04rem;
        color: #334155;
        margin-bottom: 0;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        box-shadow: 0 8px 18px rgba(15,23,42,0.04);
        padding: 0.8rem 0.9rem;
    }
    .stFileUploader > div,
    .stCameraInput > div,
    .stTabs [role="tablist"] {
        border-radius: 16px;
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pill">Eco AI assistant</div>
        <h1 class="hero-title">♻️ Waste Classification System</h1>
        <p class="hero-subtitle">Upload or capture a photo to sort waste into Plastic, Paper, Metal, or Organic with a quick AI-powered recommendation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("An Environmental Studies project demonstrating how AI can support smarter waste segregation and recycling.")
st.divider()


# ---------- MODEL LOADING ----------
@st.cache_resource
def load_classification_model():
    model = load_model(str(MODEL_PATH), compile=False)
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        labels = []
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue
            label = cleaned.split(" ", 1)[1].strip() if " " in cleaned else cleaned.strip()
            if label:
                labels.append(label)
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
    predictions = model.predict(input_data, verbose=0)[0]
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
    col1, col2 = st.columns([1.1, 0.9])

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