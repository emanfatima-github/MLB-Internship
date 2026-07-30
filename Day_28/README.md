# Smart Object Detection Application

## What is Object Detection?

Object Detection is a computer vision technique that identifies objects in an image and shows their locations by drawing bounding boxes around them.

## How YOLO is Different from Image Classification

Image classification only tells what object is present in an image. YOLO detects multiple objects, identifies them, 
and shows their locations with bounding boxes and confidence scores.

## Which YOLO Model Was Used

This project uses the YOLOv8 Nano (yolov8n.pt) pre-trained model from Ultralytics. It is lightweight, fast, and suitable for real-time object detection.

## What Objects the Application Detected

The application detected common objects such as:

* Book
* Car
* Bicycle
* Bottle
* Chair
* Laptop

The detected objects depend on the uploaded image or video.

## Challenges Faced During Implementation

* Processing videos efficiently while maintaining good detection speed.
* Managing file uploads and downloads in the Streamlit application.
* Ensuring the application worked correctly for both images and videos.
