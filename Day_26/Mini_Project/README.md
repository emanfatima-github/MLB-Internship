# Image Feature Matching System

## Project Overview

This project compares two images and finds matching features using ORB (Oriented FAST and Rotated BRIEF).
It detects important keypoints in both images, matches them, and displays the matching features.

## Features

* Upload two images.
* Detect keypoints using ORB.
* Match features using Brute Force Matcher.
* Display the matched features.
* Show the number of keypoints detected in each image.
* Show the number of good matches found.

## Dataset

The project uses 10 pairs of images, such as:

* Buildings
* Book covers
* Product images
* Mobile Phone
* Birds
* Objects from different angles

## Technologies Used

* Python
* OpenCV
* NumPy
* Pillow
* Streamlit


## Project Files

* app.py – Streamlit application
* Image_Feature_Matching_System.ipynb – Jupyter Notebook implementation
* requirements.txt – Required Python libraries
* runtime.txt – Python version
* images/ – Sample image pairs

## Output

The application displays:

* Uploaded images
* ORB keypoints
* Matched features
* Number of keypoints in each image
* Number of good matches


