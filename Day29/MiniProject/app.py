import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Construction Equipment Detection",
    page_icon="🚜",
    layout="centered"
)

st.title("🚜 Construction Equipment Detection")
st.write(
    "Upload an image or video to detect construction equipment using a custom YOLO model."
)

# ---------------------------------
# Load Model
# ---------------------------------
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
    st.error("best.pt not found.")
    st.write(MODEL_PATH)
    st.stop()

model = YOLO(MODEL_PATH)

# ---------------------------------
# Upload File
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    # ======================================
    # IMAGE
    # ======================================
    if extension in [".jpg", ".jpeg", ".png"]:

        st.subheader("Uploaded Image")
        st.image(temp_file.name, use_container_width=True)

        with st.spinner("Running Detection..."):

            results = model.predict(
                source=temp_file.name,
                conf=0.25
            )

        result = results[0]

        plotted = result.plot()

        st.subheader("Prediction Result")
        st.image(
            plotted,
            channels="BGR",
            use_container_width=True
        )

        save_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ).name

        Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)).save(save_path)

        with open(save_path, "rb") as f:
            st.download_button(
                "⬇ Download Processed Image",
                f,
                file_name="prediction.jpg",
                mime="image/jpeg"
            )

        st.subheader("Detected Objects")

        boxes = result.boxes

        if len(boxes) == 0:
            st.info("No objects detected.")

        else:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                st.write(
                    f"**{model.names[cls]}** — Confidence: **{conf:.2f}**"
                )

    # ======================================
    # VIDEO
    # ======================================
    else:

        st.subheader("Uploaded Video")
        st.video(temp_file.name)

        with st.spinner("Processing Video..."):

            output_folder = tempfile.mkdtemp()

            model.predict(
                source=temp_file.name,
                conf=0.25,
                save=True,
                project=output_folder,
                name="prediction"
            )

        prediction_folder = os.path.join(output_folder, "prediction")

        output_video = None

        for file in os.listdir(prediction_folder):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                output_video = os.path.join(prediction_folder, file)
                break

        if output_video:

            st.subheader("Prediction Result")
            st.video(output_video)

            with open(output_video, "rb") as f:
                st.download_button(
                    "⬇ Download Processed Video",
                    f,
                    file_name="prediction.mp4",
                    mime="video/mp4"
                )

        else:
            st.error("Processed video was not generated.")

st.markdown("---")
st.caption("Custom Object Detection using YOLOv8")