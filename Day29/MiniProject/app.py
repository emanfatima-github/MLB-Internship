import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import tempfile

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Construction Equipment Detection",
    page_icon="🚜",
    layout="centered"
)

st.title("🚜 Construction Equipment Detection")
st.write(
    "Upload an image to detect construction equipment using a trained YOLO model."
)

# -----------------------------------
# Load YOLO Model
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train",
    "weights",
    "best.pt"
)

if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt model not found.")
    st.write("Expected Path:")
    st.code(MODEL_PATH)
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    # -----------------------------------
    # Run Prediction
    # -----------------------------------
    with st.spinner("Detecting objects..."):

        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=False
        )

    result = results[0]

    # Draw bounding boxes
    plotted = result.plot()

    predicted_image = Image.fromarray(plotted[:, :, ::-1])

    st.subheader("Prediction Result")
    st.image(predicted_image, use_container_width=True)

    # -----------------------------------
    # Detected Objects
    # -----------------------------------
    st.subheader("Detected Objects")

    if len(result.boxes) == 0:
        st.info("No objects detected.")

    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            st.write(
                f"**{model.names[class_id]}** — Confidence: **{confidence:.2f}**"
            )

    # -----------------------------------
    # Download Result
    # -----------------------------------
    output_path = os.path.join(tempfile.gettempdir(), "prediction.jpg")
    predicted_image.save(output_path)

    with open(output_path, "rb") as file:
        st.download_button(
            label="⬇ Download Processed Image",
            data=file,
            file_name="prediction.jpg",
            mime="image/jpeg"
        )

st.markdown("---")
st.caption("Custom Object Detection using YOLOv8")