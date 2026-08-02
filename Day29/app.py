import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

st.set_page_config(
    page_title="Construction Equipment Detection",
    page_icon="🚜",
    layout="centered"
)

st.title("🚜 Construction Equipment Detection System")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader(
    "Upload Image or Video",
    type=["jpg","jpeg","png","mp4"]
)

if uploaded:

    suffix = os.path.splitext(uploaded.name)[1]

    temp = tempfile.NamedTemporaryFile(delete=False,suffix=suffix)

    temp.write(uploaded.read())

    temp.close()

    if suffix.lower()==".mp4":

        result = model.predict(
            source=temp.name,
            save=True,
            conf=0.25
        )

        output = result[0].save_dir

        files=os.listdir(output)

        video=[f for f in files if f.endswith(".mp4")][0]

        st.video(os.path.join(output,video))

        with open(os.path.join(output,video),"rb") as f:
            st.download_button(
                "Download Video",
                f,
                file_name=video
            )

    else:

        result=model.predict(
            source=temp.name,
            save=True,
            conf=0.25
        )

        output=result[0].save_dir

        files=os.listdir(output)

        image=[f for f in files if f.endswith((".jpg",".png",".jpeg"))][0]

        st.image(os.path.join(output,image))

        with open(os.path.join(output,image),"rb") as f:
            st.download_button(
                "Download Image",
                f,
                file_name=image
            )