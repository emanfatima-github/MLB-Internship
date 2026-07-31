import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

st.set_page_config(page_title="Construction Equipment Detection", layout="centered")

st.title("🚜 Construction Equipment Detection")
st.write("Upload an image or video to detect construction equipment using a custom YOLO model.")

# Load your trained model
MODEL_PATH = "runs/detect/Construction_Project/Experiment1/weights/best.pt"

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if suffix.lower() in [".jpg", ".jpeg", ".png"]:

        st.image(temp_path, caption="Uploaded Image")

        results = model.predict(
            source=temp_path,
            save=True,
            conf=0.25
        )

        output_path = results[0].save_dir

        image_name = os.path.basename(temp_path)

        prediction = os.path.join(output_path, image_name)

        st.image(prediction, caption="Prediction")

        with open(prediction, "rb") as file:
            st.download_button(
                "Download Processed Image",
                file,
                file_name="prediction.jpg"
            )

    else:

        results = model.predict(
            source=temp_path,
            save=True,
            conf=0.25
        )

        output_path = results[0].save_dir

        video_name = os.path.basename(temp_path)

        prediction = os.path.join(output_path, video_name)

        st.video(prediction)

        with open(prediction, "rb") as file:
            st.download_button(
                "Download Processed Video",
                file,
                file_name="prediction.mp4"
            )