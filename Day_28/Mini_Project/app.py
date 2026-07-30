import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Smart Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title(" Smart Object Detection Application")
st.write(
    "Upload an image or video and detect objects using YOLOv8."
)

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

        image = Image.open(uploaded_image)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Save temporary image
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Run YOLO
        results = model(temp_path)

        # Save result image
        output_path = "detected_image.jpg"
        results[0].save(filename=output_path)

        st.subheader("Detected Image")
        st.image(output_path, use_container_width=True)

        # Show detections
        st.subheader("Detected Objects")

        boxes = results[0].boxes

        if len(boxes) > 0:

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                st.write(
                    f"**{class_name}** : {confidence:.2f}"
                )

        else:
            st.write("No objects detected.")

        # Download image
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )

# ==================================================
# VIDEO DETECTION
# ==================================================
if option == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        # Save uploaded video
        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.write("Processing video...")

        # Run YOLO
        results = model.predict(
            source=temp_video.name,
            save=True,
            project="runs/detect",
            name="video_result",
            exist_ok=True
        )

        output_video = "runs/detect/video_result/" + os.path.basename(temp_video.name)

        if os.path.exists(output_video):

            st.subheader("Processed Video")

            st.video(output_video)

            with open(output_video, "rb") as file:

                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )