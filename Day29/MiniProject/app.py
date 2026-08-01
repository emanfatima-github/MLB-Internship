import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Construction Equipment Detection",
    page_icon="🚜",
    layout="centered"
)

st.title("🚜 Construction Equipment Detection")
st.write(
    "Upload an image or video to detect construction equipment using a trained YOLO model."
)

# -----------------------------
# Load Custom Model
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
# Upload File
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # ==========================================
    # IMAGE
    # ==========================================
    if extension in [".jpg", ".jpeg", ".png"]:

        st.subheader("Uploaded Image")
        st.image(temp_path, use_container_width=True)

        with st.spinner("Running Detection..."):

            results = model.predict(
                source=temp_path,
                conf=0.25,
                save=True,
                project="runs/detect",
                name="Predictions",
                exist_ok=True
            )

        output_folder = str(results[0].save_dir)

        prediction_images = [
            f for f in os.listdir(output_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if prediction_images:

            prediction_path = os.path.join(output_folder, prediction_images[-1])

            st.subheader("Prediction Result")
            st.image(prediction_path, use_container_width=True)

            with open(prediction_path, "rb") as file:
                st.download_button(
                    label="⬇ Download Processed Image",
                    data=file,
                    file_name="prediction.jpg",
                    mime="image/jpeg"
                )

    # ==========================================
    # VIDEO
    # ==========================================
    else:

        st.subheader("Uploaded Video")
        st.video(temp_path)

        with st.spinner("Running Detection..."):

            results = model.predict(
                source=temp_path,
                conf=0.25,
                save=True,
                project="runs/detect",
                name="Predictions",
                exist_ok=True
            )

        output_folder = str(results[0].save_dir)

        prediction_videos = [
            f for f in os.listdir(output_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]

        if prediction_videos:

            prediction_path = os.path.join(output_folder, prediction_videos[-1])

            st.subheader("Prediction Result")
            st.video(prediction_path)

            with open(prediction_path, "rb") as file:
                st.download_button(
                    label="⬇ Download Processed Video",
                    data=file,
                    file_name="prediction.mp4",
                    mime="video/mp4"
                )

st.markdown("---")
st.write("Developed using YOLOv8 and Streamlit")