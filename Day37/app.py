import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# =========================
# Page
# =========================

st.set_page_config(
    page_title="YOLO V1 vs V2",
    layout="wide"
)

st.title("YOLO Model Improvement")
st.write("Comparison of Model V1 and Model V2")


# =========================
# Model paths
# =========================

V1_MODEL = r"C:\Users\Localws\Bottle-Detection-1\best.pt"

V2_MODEL = r"C:\Users\Localws\Bottle-Detection-1\Day37_Training\V2\weights\best.pt"


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

    # Predictions
    result_v1 = model_v1.predict(
        image_array,
        conf=0.25,
        verbose=False
    )[0]

    result_v2 = model_v2.predict(
        image_array,
        conf=0.25,
        verbose=False
    )[0]


    # =========================
    # Original image
    # =========================

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )


    # =========================
    # V1 vs V2
    # =========================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Model V1")

        v1_image = result_v1.plot()

        st.image(
            v1_image,
            channels="BGR",
            use_container_width=True
        )

        v1_count = len(result_v1.boxes)

        st.write(
            f"Detections: **{v1_count}**"
        )


    with col2:

        st.subheader("Model V2")

        v2_image = result_v2.plot()

        st.image(
            v2_image,
            channels="BGR",
            use_container_width=True
        )

        v2_count = len(result_v2.boxes)

        st.write(
            f"Detections: **{v2_count}**"
        )


# =========================
# Model metrics
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