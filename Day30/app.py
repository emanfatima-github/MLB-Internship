import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Smart Object Tracking", layout="wide")

st.title(" Smart Object Tracking System")

st.write("""
Upload a video to detect and track objects.
The application assigns unique IDs, shows confidence scores,
counts unique objects, and allows downloading the processed video.
""")

# -------------------------------
# Load YOLO Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_video = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video:

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    os.makedirs("output", exist_ok=True)

    output_path = os.path.join("output", "tracked_video.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    unique_ids = set()

    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        if result.boxes.id is not None:

            ids = result.boxes.id.cpu().numpy().astype(int)

            for obj_id in ids:
                unique_ids.add(obj_id)

        annotated = result.plot()

        writer.write(annotated)

        frame_count += 1

        if total_frames > 0:
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    writer.release()

    st.success("Tracking Completed!")

    st.write(f"### Total Unique Objects Detected: {len(unique_ids)}")

    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            "Download Processed Video",
            f,
            file_name="tracked_video.mp4",
            mime="video/mp4"
        )