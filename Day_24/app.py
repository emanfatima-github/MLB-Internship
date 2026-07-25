import streamlit as st
from PIL import Image
import tempfile
import os

from ocr import extract_text

st.set_page_config(
    page_title="Simple OCR Document Reader",
    page_icon="📄",
    layout="wide"
)

st.title("Simple OCR Document Reader")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    with st.spinner("Extracting text..."):
        text = extract_text(image_path)

    os.makedirs("outputs", exist_ok=True)

    output_file = "outputs/extracted_text.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    with col2:
        st.subheader("Extracted Text")

        if text.strip() == "":
            st.warning("No text detected.")
        else:
            st.text_area(
                "",
                text,
                height=400
            )

            st.download_button(
                "Download TXT",
                text,
                file_name="extracted_text.txt"
            )

    os.remove(image_path)