import torch
import cv2
import numpy as np
import streamlit as st
import easyocr

# Limit CPU threads
torch.set_num_threads(1)

st.set_page_config(
    page_title="Simple OCR Document Reader",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Simple OCR Document Reader")
st.write("Upload a document image to extract text using EasyOCR.")

# Cache OCR model
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        detector=True,
        recognizer=True,
        verbose=False
    )

# Sidebar
st.sidebar.header("Configuration")

default_filename = st.sidebar.text_input(
    "Output File Name",
    value="extracted_text"
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.write("✅ Image uploaded")

    # Convert uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        image_rgb,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("Loading OCR model...")

    try:
        reader = get_ocr_reader()
        st.success("✅ OCR model loaded")
    except Exception as e:
        st.error(f"Error loading OCR model:\n{e}")
        st.stop()

    st.write("Running OCR...")
    st.write(f"Image Shape: {image_rgb.shape}")

    try:

        results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False,
            batch_size=1
        )

        st.success("✅ OCR Finished")
        st.write(f"Detections Found: {len(results)}")

        extracted_text = ""

        for item in results:
            extracted_text += item[1] + "\n"

        if extracted_text.strip():

            st.subheader("Extracted Text")

            st.text_area(
                "",
                extracted_text,
                height=300
            )

            st.download_button(
                "📥 Download Text",
                extracted_text,
                file_name=f"{default_filename}.txt",
                mime="text/plain"
            )

        else:
            st.warning("No text detected.")

    except Exception as e:
        st.error(f"OCR Error:\n{e}")