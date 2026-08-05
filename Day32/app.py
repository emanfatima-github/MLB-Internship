import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image

st.set_page_config(
    page_title="Smart People Counting System",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Smart People Counting System")
st.write("Upload an image or video to detect and count people.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # -----------------------------
    # IMAGE
    # -----------------------------
    if uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file)

        results = model(image)

        result = results[0]

        people = 0

        for box in result.boxes:
            if int(box.cls[0]) == 0:
                people += 1

        processed = result.plot()

        st.image(image, caption="Original Image", use_container_width=True)

        st.image(processed, caption="Processed Image", use_container_width=True)

        st.success(f"People Detected: {people}")

        output_image = "processed_image.jpg"

        cv2.imwrite(
            output_image,
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        with open(output_image, "rb") as file:
            st.download_button(
                "Download Processed Image",
                file,
                file_name="processed_image.jpg"
            )

    # -----------------------------
    # VIDEO
    # -----------------------------
    else:

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp.name)

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

        frame_window = st.empty()

        current_count = st.empty()

        max_count = st.empty()

        maximum = 0

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

            people = 0

            if results[0].boxes is not None:

                for box in results[0].boxes:

                    if int(box.cls[0]) != 0:
                        continue

                    people += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    conf = float(box.conf[0])

                    track_id = -1

                    if box.id is not None:
                        track_id = int(box.id[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID:{track_id} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )

            maximum = max(maximum, people)

            cv2.putText(
                frame,
                f"People: {people}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                f"Maximum: {maximum}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

            writer.write(frame)

            frame_window.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            current_count.info(f"Current People Count: {people}")

            max_count.success(f"Maximum People Count: {maximum}")

        cap.release()
        writer.release()

        st.success("Video Processing Completed!")

        st.video(output_video)

        with open(output_video, "rb") as file:
            st.download_button(
                "Download Processed Video",
                file,
                file_name="processed_video.mp4"
            )