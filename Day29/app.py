import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Construction Equipment Detection",
    page_icon="🚜",
    layout="centered"
)

st.title(" Construction Equipment Detection System")

# -----------------------------
# Debug Information
# -----------------------------
st.subheader("Debug Information")

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

st.write("Current Working Directory:")
st.code(current_dir)

st.write("Script Directory:")
st.code(script_dir)

st.write("Files inside Script Directory:")
st.write(os.listdir(script_dir))

st.write("Model Path:")
st.code(model_path)

st.write("Model Exists:", os.path.exists(model_path))

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"Model not found!\n\nExpected location:\n{model_path}")
        st.stop()

    return YOLO(model_path)

model = load_model()

# -----------------------------
# Upload File
# -----------------------------
uploaded = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded:

    suffix = os.path.splitext(uploaded.name)[1]

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded.read())
    temp.close()

    with st.spinner("Running Detection..."):

        results = model.predict(
            source=temp.name,
            conf=0.25,
            save=True
        )

    output_dir = results[0].save_dir

    if suffix.lower() == ".mp4":

        video_files = [
            f for f in os.listdir(output_dir)
            if f.endswith(".mp4")
        ]

        if video_files:

            output_video = os.path.join(output_dir, video_files[0])

            st.success("Detection Completed!")

            st.video(output_video)

            with open(output_video, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    f,
                    file_name=video_files[0]
                )

    else:

        image_files = [
            f for f in os.listdir(output_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if image_files:

            output_image = os.path.join(output_dir, image_files[0])

            st.success("Detection Completed!")

            st.image(output_image, use_container_width=True)

            with open(output_image, "rb") as f:
                st.download_button(
                    "Download Processed Image",
                    f,
                    file_name=image_files[0]
                )