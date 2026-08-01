import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Custom YOLO Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Custom YOLO Object Detection")
st.write("Upload an image and run detection using your trained YOLO model.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Your trained custom model

model = load_model()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Run Detection"):

        with st.spinner("Running inference..."):

            results = model(image)
            result = results[0]

            # Annotated image
            annotated_image = result.plot()

            st.subheader("Prediction Result")
            st.image(annotated_image, use_container_width=True)

            # Save prediction image
            output_path = "prediction.jpg"
            cv2.imwrite(
                output_path,
                cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            )

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Prediction",
                    data=file,
                    file_name="prediction.jpg",
                    mime="image/jpeg"
                )

            # Display detected objects
            st.subheader("Detected Objects")

            if len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    st.write(
                        f"**{model.names[cls]}** : {conf:.2f}"
                    )
            else:
                st.info("No objects detected.")