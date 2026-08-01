import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Construction Equipment Detection",
    layout="centered"
)

st.title(" Construction Equipment Detection")
st.write(
    "Upload an image or video to detect construction equipment using a trained YOLO model."
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "Construction_Project",
    "Experiment1",
    "weights",
    "best.pt"
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # -----------------------------
    # Image Inference
    # -----------------------------
    if suffix.lower() in [".jpg", ".jpeg", ".png"]:

        st.subheader("Uploaded Image")
        st.image(temp_path, use_container_width=True)

        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=True,
            project="runs/detect",
            name="Predictions",
            exist_ok=True
        )

        prediction_path = results[0].save_dir / os.path.basename(temp_path)

        st.subheader("Prediction Result")
        st.image(str(prediction_path), use_container_width=True)

        with open(prediction_path, "rb") as file:
            st.download_button(
                label="⬇ Download Processed Image",
                data=file,
                file_name="prediction.jpg",
                mime="image/jpeg"
            )

    # -----------------------------
    # Video Inference
    # -----------------------------
    else:

        st.subheader("Uploaded Video")
        st.video(temp_path)

        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=True,
            project="runs/detect",
            name="Predictions",
            exist_ok=True
        )

        prediction_path = results[0].save_dir / os.path.basename(temp_path)

        st.subheader("Prediction Result")
        st.video(str(prediction_path))

        with open(prediction_path, "rb") as file:
            st.download_button(
                label="⬇ Download Processed Video",
                data=file,
                file_name="prediction.mp4",
                mime="video/mp4"
            )