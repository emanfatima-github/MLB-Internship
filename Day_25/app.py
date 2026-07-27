import os
import cv2
import easyocr
import gradio as gr

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Create output folder
os.makedirs("output", exist_ok=True)


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return image, processed


# -----------------------------
# OCR Function
# -----------------------------
def extract_text(image):

    original, processed = preprocess_image(image)

    results = reader.readtext(processed)

    extracted_text = ""

    for result in results:
        extracted_text += result[1] + "\n"

    return original, extracted_text


# -----------------------------
# Save Text File
# -----------------------------
def save_text(text):

    output_path = "output/extracted_text.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    return output_path


# -----------------------------
# Main Function
# -----------------------------
def ocr_document(image):

    original, text = extract_text(image)

    output_file = save_text(text)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    return original, text, output_file


# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=ocr_document,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Original Image"),
        gr.Textbox(label="Extracted Text", lines=15),
        gr.File(label="Download Extracted Text")
    ],
    title=" Document OCR Web Application",
    description="Upload a document image to extract readable text using EasyOCR."
)


# -----------------------------
# Launch App
# -----------------------------
if __name__ == "__main__":
    demo.launch()