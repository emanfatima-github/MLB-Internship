import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# -----------------------------
# Fix Ultralytics Settings
# -----------------------------
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Smart Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Smart Object Detection Application")
st.write("Upload an image or video and detect objects using YOLOv8.")

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Select Input Type
# -----------------------------
option = st.radio(
    "Choose Input Type:",
    ["Image", "Video"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================
if option == "Image":

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Original Image")
        st.image(image, width="stretch")

        image_np = np.array(image)

        with st.spinner("Detecting objects..."):
            results = model(image_np)

        annotated_image = results[0].plot()

        output_path = "detected_image.jpg"

        cv2.imwrite(
            output_path,
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )

        st.subheader("Detected Image")
        st.image(output_path, width="stretch")

        st.subheader("Detected Objects")

        boxes = results[0].boxes

        if len(boxes) == 0:
            st.info("No objects detected.")

        else:
            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                st.write(
                    f"**{model.names[class_id]}** : {confidence:.2f}"
                )

        with open(output_path, "rb") as file:

            st.download_button(
                label="⬇ Download Processed Image",
                data=file,
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )

# ==================================================
# VIDEO DETECTION
# ==================================================
elif option == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("Original Video")
        st.video(uploaded_video)

        # Save uploaded video
        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            fps = 25

        output_video = "processed_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        progress = st.progress(0)
        status = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            frame_number += 1

            if total_frames > 0:
                progress.progress(frame_number / total_frames)

            status.text(
                f"Processing frame {frame_number} of {total_frames}"
            )

        cap.release()
        out.release()

        progress.empty()
        status.empty()

        st.success("✅ Video processed successfully!")

        st.subheader("Processed Video")

        st.video(output_video)

        with open(output_video, "rb") as file:

            st.download_button(
                label="⬇ Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )