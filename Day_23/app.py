import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Computer Vision Image Processing Studio")

st.title("Computer Vision Image Processing Studio")
st.write("Upload an image, select an operation, and view the processed result.")

# -----------------------------
# Image Processing Function
# -----------------------------
def process_image(image, operation):

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

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()
        cv2.drawContours(result, contours, -1, (0,255,0), 2)

    elif operation == "Shape Detection":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        _, thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY)

        contours,_ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = img.copy()

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 500:
                continue

            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.04*peri,True)

            x,y,w,h = cv2.boundingRect(approx)

            if len(approx)==3:
                shape="Triangle"

            elif len(approx)==4:
                ratio=w/float(h)

                if 0.95<=ratio<=1.05:
                    shape="Square"
                else:
                    shape="Rectangle"

            elif len(approx)==5:
                shape="Pentagon"

            elif len(approx)==6:
                shape="Hexagon"

            else:
                shape="Circle"

            cv2.drawContours(result,[approx],-1,(0,255,0),2)
            cv2.putText(
                result,
                shape,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                2
            )

    else:
        result = img

    if len(result.shape)==2:
        output = Image.fromarray(result)

    else:
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        output = Image.fromarray(result)

    return output


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

operation = st.selectbox(
    "Select Operation",
    [
        "Grayscale",
        "Blur",
        "Edge Detection",
        "Image Rotation",
        "Image Enhancement",
        "Contour Detection",
        "Shape Detection"
    ]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    processed = process_image(image, operation)

    with col2:
        st.subheader("Processed Image")
        st.image(processed, use_container_width=True)

    buffer = io.BytesIO()
    processed.save(buffer, format="PNG")

    st.download_button(
        label="Download Processed Image",
        data=buffer.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )