# Smart Video Analytics System

## How It Works

The system takes a video and processes it frame by frame using YOLO. It detects and tracks objects, draws bounding boxes, shows tracking IDs, 
counts objects, calculates FPS, and monitors a selected ROI.

## Tracking IDs

Each detected object gets a unique tracking ID. The ID helps the system recognize the same object across different frames and prevents duplicate counting.

## Entry and Exit Detection

An object is considered to have entered when its center point moves from outside the ROI to inside it. 
It is considered to have exited when it moves from inside the ROI to outside.

## FPS Results

The system was tested with 640px, 480px, and frame skipping configurations. FPS was recorded for each configuration.

## Best Configuration

The configuration with the highest FPS and lowest processing time performed best.

## Problems and Solutions

* Video could not be opened: Fixed the video paths.
* Slow processing: Tested smaller image size and frame skipping.
* Tracking and ROI issues: Used YOLO tracking IDs and center points to detect entry and exit.
