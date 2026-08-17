# Day 39 – Bottle Detection AI

## Project Overview

This project uses a custom **YOLO model to detect bottles in images and videos**.

## Dataset

The model was trained using a **custom bottle detection dataset from Roboflow**. The test set contains **20 unseen images** for model evaluation.

## Model Performance

The model was evaluated using:

Precision: 0.5429395491088681
Recall: 0.5774647887323944
mAP@50: 0.5348673534310038
mAP@50-95: 0.2788802243824062

## Main Errors

The model sometimes struggled with:

* Small bottles
* Overlapping bottles
* Blurry images
* Difficult backgrounds
* Partially visible bottles

Five difficult examples were reviewed to understand these errors.

## Application

A simple Streamlit application was created. The user can:

1. Upload an image or short video.
2. Adjust the confidence threshold.
3. Detect bottles using the custom YOLO model.
4. View bounding boxes, class names, and confidence scores.
5. See basic detection statistics.
6. Download the prediction result.

## Future Improvements

With more time, I would:

* Add more training images.
* Improve the quality of annotations.
* Add more difficult examples to the dataset.
* Train for more epochs.
* Improve video processing speed.
* Further tune the confidence threshold.