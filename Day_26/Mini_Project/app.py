import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Feature Matching", layout="wide")

st.title(" Image Feature Matching using ORB")

st.write("Upload two images to detect and match features.")

image1 = st.file_uploader("Upload First Image", type=["jpg","jpeg","png"])

image2 = st.file_uploader("Upload Second Image", type=["jpg","jpeg","png"])


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    return image


if image1 is not None and image2 is not None:

    img1 = load_image(image1)
    img2 = load_image(image2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    st.write("### Keypoints")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption=f"Image 1 ({len(kp1)} Keypoints)", use_container_width=True)

    with col2:
        st.image(img2, caption=f"Image 2 ({len(kp2)} Keypoints)", use_container_width=True)

    if des1 is not None and des2 is not None:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:50]

        matched_image = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        st.write("## Feature Matching")

        st.image(matched_image, use_container_width=True)

        st.success(f"Image 1 Keypoints : {len(kp1)}")
        st.success(f"Image 2 Keypoints : {len(kp2)}")
        st.success(f"Good Matches : {len(good_matches)}")

    else:

        st.error("Could not detect descriptors in one or both images.")