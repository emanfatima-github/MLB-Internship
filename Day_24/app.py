import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR
)

st.title("OCR Document Reader")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.equalizeHist(gray)

    result = reader.readtext(gray)

    extracted_text = ""

    for item in result:
        extracted_text += item[1] + "\n"

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Extracted Text")

    st.text_area(
        "",
        extracted_text,
        height=250
    )

    st.download_button(
        "Download Text",
        extracted_text,
        file_name="output.txt"
    )