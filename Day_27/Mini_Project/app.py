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
        "Otsu Thresholding",
        "Foreground/Background Segmentation"
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

    # --------------------------
    # Binary Thresholding
    # --------------------------
    if method == "Binary Thresholding":

        _, segmented = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        st.subheader("Original Image")
        st.image(image)

        st.subheader("Segmented Output")
        segmented_image = Image.fromarray(segmented)
        st.image(segmented_image)

    # --------------------------
    # Adaptive Thresholding
    # --------------------------
    elif method == "Adaptive Thresholding":

        segmented = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        st.subheader("Original Image")
        st.image(image)

        st.subheader("Segmented Output")
        segmented_image = Image.fromarray(segmented)
        st.image(segmented_image)

    # --------------------------
    # Otsu Thresholding
    # --------------------------
    elif method == "Otsu Thresholding":

        _, segmented = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        st.subheader("Original Image")
        st.image(image)

        st.subheader("Segmented Output")
        segmented_image = Image.fromarray(segmented)
        st.image(segmented_image)

    # --------------------------
    # Foreground / Background Segmentation
    # --------------------------
    else:

        _, foreground = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        background = cv2.bitwise_not(foreground)

        st.subheader("Original Image")
        st.image(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Foreground")
            st.image(Image.fromarray(foreground))

        with col2:
            st.subheader("Background")
            st.image(Image.fromarray(background))

        segmented = foreground
        segmented_image = Image.fromarray(segmented)

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