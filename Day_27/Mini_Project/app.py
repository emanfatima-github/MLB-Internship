import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Document & Object Segmentation Tool",
    layout="centered"
)

st.title("📄 Document & Object Segmentation Tool")
st.write("Upload an image, choose a segmentation method, and download the result.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

method = st.selectbox(
    "Select Segmentation Method",
    (
        "Binary Thresholding",
        "Adaptive Thresholding",
        "Otsu Thresholding"
    )
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == "Binary Thresholding":

        _, result = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

    elif method == "Adaptive Thresholding":

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    else:

        _, result = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    st.subheader("Segmented Output")

    st.image(
        result,
        use_container_width=True
    )

    success, encoded = cv2.imencode(".png", result)

    if success:

        st.download_button(
            "Download Processed Image",
            data=encoded.tobytes(),
            file_name="segmented_image.png",
            mime="image/png"
        )