import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Construction Equipment Detection")

# Load trained model
model = YOLO("construction-equipment-1/Construction_Project/Experiment1/weights/best.pt")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Run detection
    results = model.predict(image, conf=0.25)

    # Display result
    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result")