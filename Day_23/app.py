import cv2
import numpy as np
import gradio as gr
from PIL import Image

def process_image(image, operation):

    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if operation == "Grayscale":
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == "Blur":
        result = cv2.GaussianBlur(img, (9, 9), 0)
    elif operation == "Edge Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
    elif operation == "Image Rotation":
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
        result = cv2.warpAffine(img, matrix, (w, h))
    elif operation == "Image Enhancement":
        result = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    elif operation == "Contour Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    elif operation == "Shape Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                ratio = w / float(h)
                if 0.95 <= ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"

            cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
            cv2.putText(result, shape, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)
    else:
        result = img

    # Convert back to PIL image
    if len(result.shape) == 2:
        output = Image.fromarray(result)
    else:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(result)

    # Save processed image
    output.save("processed_image.png")

    return output

interface = gr.Interface(
    fn=lambda image, operation: (
        image,
        process_image(image, operation)
    ),

    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(
            choices=[
                "Grayscale",
                "Blur",
                "Edge Detection",
                "Image Rotation",
                "Image Enhancement",
                "Contour Detection",
                "Shape Detection"
            ],
            label="Select Operation"
        )
    ],

    outputs=[
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Processed Image")
    ],

    title="Computer Vision Image Processing Studio",

    description="Upload an image, select an operation, and view the processed result."
)

interface.launch()