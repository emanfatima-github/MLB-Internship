import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart People Counting System",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Smart People Counting System")
st.write("Upload an image or video to detect and count people using YOLOv8.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # ==================================================
    # IMAGE
    # ==================================================
    if uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file)

        results = model(image)

        result = results[0]

        people_count = 0

        for box in result.boxes:
            if int(box.cls[0]) == 0:
                people_count += 1

        processed = result.plot()

        st.subheader("Original Image")
        st.image(image)

        st.subheader("Processed Image")
        st.image(processed)

        st.success(f"People Detected: {people_count}")

        output_image = "processed_image.jpg"

        cv2.imwrite(
            output_image,
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        with open(output_image, "rb") as file:
            st.download_button(
                "Download Processed Image",
                file,
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )

    # ==================================================
    # VIDEO
    # ==================================================
    else:

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = "processed_video.mp4"

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_placeholder = st.empty()

        current_placeholder = st.empty()

        maximum_placeholder = st.empty()

        maximum_people = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
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

                    confidence = float(box.conf[0])

                    track_id = -1

                    if box.id is not None:
                        track_id = int(box.id[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID:{track_id} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            maximum_people = max(maximum_people, people_count)

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
                f"Maximum: {maximum_people}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            writer.write(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame_rgb)

            current_placeholder.info(f"Current People Count: {people_count}")

            maximum_placeholder.success(f"Maximum People Count: {maximum_people}")

        cap.release()
        writer.release()

        st.success("Video Processing Completed!")

        st.video(output_video)

        with open(output_video, "rb") as file:
            st.download_button(
                "Download Processed Video",
                file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )