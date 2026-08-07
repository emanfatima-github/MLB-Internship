import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import csv
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart People Counting System",
    layout="wide"
)

st.title("👥 Smart People Counting System")
st.write("Upload an image or video to detect, track, and count people.")

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Confidence Slider
# -----------------------------
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_input.write(uploaded_file.read())
    temp_input.close()

    input_path = temp_input.name

    # ==========================================================
    # IMAGE PROCESSING
    # ==========================================================
    if suffix.lower() in [".jpg", ".jpeg", ".png"]:

        results = model.predict(
            input_path,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        people_count = 0
        csv_data = []

        for box in result.boxes:

            if int(box.cls[0]) != 0:
                continue

            people_count += 1

            conf = float(box.conf[0])

            csv_data.append([
                people_count,
                round(conf, 2)
            ])

        processed = result.plot()

        output_image = "processed_image.jpg"

        cv2.imwrite(
            output_image,
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        csv_file = "people_report.csv"

        with open(csv_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Person No",
                "Confidence"
            ])

            writer.writerows(csv_data)

        st.success(f"People Detected: {people_count}")

        st.image(
            output_image,
            caption="Processed Image",
            use_container_width=True
        )

        with open(output_image, "rb") as f:
            st.download_button(
                "Download Processed Image",
                f,
                file_name="processed_image.jpg"
            )

        with open(csv_file, "rb") as f:
            st.download_button(
                "Download CSV Report",
                f,
                file_name="people_report.csv"
            )

    # ==========================================================
    # VIDEO PROCESSING
    # ==========================================================
    else:

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Unable to open video.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video = "processed_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        progress_bar = st.progress(0)

        status = st.empty()

        csv_data = []

        frame_no = 0

        max_people = 0

        video_placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_no += 1

            progress_bar.progress(min(frame_no / total_frames, 1.0))

            status.text(f"Processing Frame {frame_no}/{total_frames}")

            results = model.track(
                frame,
                conf=confidence,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            people_count = 0

            if results[0].boxes is not None:

                for box in results[0].boxes:

                    if int(box.cls[0]) != 0:
                        continue

                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    conf = float(box.conf[0])

                    if box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        track_id = -1

                    csv_data.append([
                        frame_no,
                        track_id,
                        round(conf, 2)
                    ])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID:{track_id} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            max_people = max(max_people, people_count)

            cv2.putText(
                frame,
                f"People: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Maximum: {max_people}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            out.write(frame)

        cap.release()
        out.release()

        csv_file = "people_report.csv"

        with open(csv_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Frame",
                "Track ID",
                "Confidence"
            ])

            writer.writerows(csv_data)

        st.success("Video Processing Completed!")

        st.write(f"### Maximum People Detected: {max_people}")

        st.video(output_video)

        with open(output_video, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        with open(csv_file, "rb") as f:
            st.download_button(
                label="Download CSV Report",
                data=f,
                file_name="people_report.csv",
                mime="text/csv"
            )

        os.remove(input_path)