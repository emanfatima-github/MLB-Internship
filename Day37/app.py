import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="YOLO V1 vs V2",
    layout="wide"
)

st.title("YOLO Model Improvement")
st.write("Model V1 vs Model V2 Comparison")


# =========================
# Model paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

V1_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "v1_best.pt"
)

V2_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "v2_best.pt"
)


# =========================
# Check model files
# =========================

if not os.path.exists(V1_MODEL):
    st.error(f"V1 model not found: {V1_MODEL}")
    st.stop()

if not os.path.exists(V2_MODEL):
    st.error(f"V2 model not found: {V2_MODEL}")
    st.stop()


# =========================
# Load models
# =========================

@st.cache_resource
def load_models():
    model_v1 = YOLO(V1_MODEL)
    model_v2 = YOLO(V2_MODEL)

    return model_v1, model_v2


model_v1, model_v2 = load_models()


# =========================
# Upload image
# =========================

uploaded_file = st.file_uploader(
    "Upload a test image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    v1_result = model_v1.predict(
        image_array,
        conf=0.25,
        verbose=False
    )[0]

    v2_result = model_v2.predict(
        image_array,
        conf=0.25,
        verbose=False
    )[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model V1")

        st.image(
            v1_result.plot(),
            channels="BGR",
            use_container_width=True
        )

        st.write(f"Detections: {len(v1_result.boxes)}")

    with col2:
        st.subheader("Model V2")

        st.image(
            v2_result.plot(),
            channels="BGR",
            use_container_width=True
        )

        st.write(f"Detections: {len(v2_result.boxes)}")


# =========================
# Metrics
# =========================

st.divider()

st.subheader("V1 vs V2 Performance")

st.table({
    "Metric": [
        "Precision",
        "Recall",
        "mAP@50",
        "mAP@50-95"
    ],
    "V1": [
        0.7772,
        0.5114,
        0.5764,
        0.3263
    ],
    "V2": [
        0.7756,
        0.4886,
        0.5809,
        0.2766
    ]
})