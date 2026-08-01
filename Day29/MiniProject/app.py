import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

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
# Load Model
# -----------------------------
import os
from ultralytics import YOLO

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

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # ==========================================
    # IMAGE
    # ==========================================
    if file_extension in [".jpg", ".jpeg", ".png"]:

        st.subheader("Uploaded Image")
        st.image(temp_file.name, use_container_width=True)

        with st.spinner("Running Detection..."):

            results = model.predict(
                source=temp_file.name,
                conf=0.25
            )

        result = results[0]

        predicted_image = result.plot()

        st.subheader("Prediction Result")
        st.image(
            predicted_image,
            channels="BGR",
            use_container_width=True
        )

        output_image = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        Image.fromarray(predicted_image[:, :, ::-1]).save(output_image.name)

        with open(output_image.name, "rb") as file:
            st.download_button(
                "⬇ Download Processed Image",
                file,
                file_name="prediction.jpg",
                mime="image/jpeg"
            )

    # ==========================================
    # VIDEO
    # ==========================================
    else:

        st.subheader("Uploaded Video")
        st.video(temp_file.name)

        with st.spinner("Running Detection..."):

            results = model.predict(
                source=temp_file.name,
                conf=0.25,
                save=True,
                project="runs/detect",
                name="video_predictions",
                exist_ok=False
            )

        output_folder = str(results[0].save_dir)

        video_files = [
            f for f in os.listdir(output_folder)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

        if len(video_files) > 0:

            prediction_video = os.path.join(output_folder, video_files[0])

            st.subheader("Prediction Result")
            st.video(prediction_video)

            with open(prediction_video, "rb") as file:
                st.download_button(
                    "⬇ Download Processed Video",
                    file,
                    file_name="prediction.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Prediction video was not generated.")

st.markdown("---")
st.write("Developed using YOLOv8 and Streamlit")