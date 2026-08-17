import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

st.set_page_config(
    page_title="Bottle Detection AI",
    page_icon="🍾"
)

st.title("Bottle Detection AI")
st.write("Upload an image or short video to detect bottles.")

# Load model
model = YOLO("best.pt")

# Confidence threshold
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Select input
input_type = st.radio(
    "Choose input type:",
    ["Image", "Video"]
)

# ---------------- IMAGE ----------------

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Prediction
        result = model.predict(
            image,
            conf=confidence,
            verbose=False
        )[0]

        # Annotated image
        annotated = result.plot()

        st.subheader("Detection Result")
        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

        # Statistics
        total_detections = len(result.boxes)

        st.subheader("Detection Statistics")

        st.write(
            "Total detections:",
            total_detections
        )

        st.write(
            "Confidence threshold:",
            confidence
        )

        # Save prediction
        output_path = "prediction.jpg"

        cv2.imwrite(
            output_path,
            annotated
        )

        with open(output_path, "rb") as file:

            st.download_button(
                label="Download Prediction",
                data=file,
                file_name="prediction.jpg",
                mime="image/jpeg"
            )


# ---------------- VIDEO ----------------

else:

    uploaded_video = st.file_uploader(
        "Upload a short video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_video.read()
        )

        temp_input.close()

        output_video = "processed_video.mp4"

        cap = cv2.VideoCapture(
            temp_input.name
        )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 20

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0
        total_detections = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            result = model.predict(
                frame,
                conf=confidence,
                verbose=False
            )[0]

            annotated = result.plot()

            total_detections += len(result.boxes)

            out.write(annotated)

            frame_count += 1

        cap.release()
        out.release()

        st.success("Video processing completed!")

        st.subheader("Video Statistics")

        st.write(
            "Frames processed:",
            frame_count
        )

        st.write(
            "Total detections:",
            total_detections
        )

        st.write(
            "Confidence threshold:",
            confidence
        )

        with open(output_video, "rb") as file:

            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        os.remove(temp_input.name)