import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# ==========================
# Streamlit Page Configuration
# ==========================
st.set_page_config(
    page_title="Document & Object Segmentation Tool",
    layout="centered"
)

st.title("📄 Document & Object Segmentation Tool")
st.write(
    "Upload an image, choose a segmentation method, view the segmented result, and download it."
)

# ==========================
# Upload Image
# ==========================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# Select Segmentation Method
# ==========================
method = st.selectbox(
    "Select Segmentation Method",
    [
        "Binary Thresholding",
        "Adaptive Thresholding",
        "Otsu Thresholding"
    ]
)

# ==========================
# Process Image
# ==========================
if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply selected thresholding method
    if method == "Binary Thresholding":

        _, segmented = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

    elif method == "Adaptive Thresholding":

        segmented = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    else:

        _, segmented = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # ==========================
    # Display Images
    # ==========================
    st.subheader("Original Image")
    st.image(image)

    st.subheader("Segmented Output")

    segmented_image = Image.fromarray(segmented)

    st.image(segmented_image)

    # ==========================
    # Download Button
    # ==========================
    buffer = BytesIO()
    segmented_image.save(buffer, format="PNG")

    st.download_button(
        label="📥 Download Processed Image",
        data=buffer.getvalue(),
        file_name="segmented_image.png",
        mime="image/png"
    )