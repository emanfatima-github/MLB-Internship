import os
import torch
import cv2
import numpy as np
import streamlit as st
import easyocr

torch.set_num_threads(1)

st.set_page_config(
    page_title="Simple OCR Document Reader",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Simple OCR Document Reader")
st.write("Upload a document image to extract visible text using EasyOCR.")

# Store EasyOCR models in temporary directory
MODEL_DIR = "/tmp/easyocr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    """
    Load and cache the EasyOCR model.
    """
    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=MODEL_DIR,
        download_enabled=True
    )

# Sidebar
st.sidebar.header("Configuration")
default_filename = st.sidebar.text_input(
    "Saved File Name",
    value="extracted_text"
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a document image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load OCR model only after image upload
    with st.spinner("Loading OCR model..."):
        try:
            reader = get_ocr_reader()
        except Exception as e:
            st.error(f"Error loading OCR model:\n\n{e}")
            st.stop()

    # Convert uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Invalid image format.")
        st.stop()

    col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.subheader("Original Image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original Image")

    # OCR
    with col2:
        st.subheader("Extracted Text")

        with st.spinner("Extracting text..."):
            try:
                results = reader.readtext(image)

                extracted_lines = [item[1] for item in results]
                final_text = "\n".join(extracted_lines)

                if final_text.strip():

                    st.text_area(
                        "Output",
                        value=final_text,
                        height=300
                    )

                    st.download_button(
                        label="📥 Download Text File",
                        data=final_text.encode("utf-8"),
                        file_name=f"{default_filename}.txt",
                        mime="text/plain"
                    )

                else:
                    st.warning("No text detected.")

            except Exception as ex:
                st.error(f"OCR Error:\n\n{ex}")