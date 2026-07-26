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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "easyocr_models")

os.makedirs(MODEL_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory=MODEL_DIR,
        download_enabled=True      # <-- Changed
    )


st.sidebar.header("Configuration")

default_filename = st.sidebar.text_input(
    "Saved File Name",
    value="extracted_text"
)

uploaded_file = st.file_uploader(
    "Choose a document image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read image.")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    st.write("Loading OCR model...")

    try:
        reader = get_ocr_reader()
        st.success("OCR model loaded.")
    except Exception as e:
        st.error(e)
        st.stop()

    st.write("Starting OCR...")

    try:

        results = reader.readtext(image_rgb)

        st.success("OCR Finished")

        st.write(f"Detections Found: {len(results)}")

        extracted_text = ""

        for item in results:
            extracted_text += item[1] + "\n"

        if extracted_text.strip() == "":
            st.warning("No text detected.")

        else:

            st.text_area(
                "Extracted Text",
                extracted_text,
                height=300
            )

            st.download_button(
                "📥 Download Text",
                extracted_text,
                file_name=f"{default_filename}.txt"
            )

    except Exception as e:
        st.error(f"OCR Error:\n{e}")