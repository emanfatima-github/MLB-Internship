# Real-Time Video Processing Tool

## Project Overview

This project is developed using Python and OpenCV. It processes both recorded videos and live webcam video by applying different image processing techniques to each frame.

## Features

- Read a recorded video
- Display the original video
- Convert each frame to grayscale
- Apply Gaussian Blur
- Apply Canny Edge Detection
- Display the processed video
- Save the processed video
- Process live webcam video in real time

## Technologies Used

- Python
- OpenCV
- Gradio
- NumPy

## Files Included

- app.py
- requirements.txt
- README.md
- Video_Challenge.ipynb
- videos/
- output/
- examples/

## Testing

The application was tested on:

- Three different recorded videos
- Live webcam feed

## Observations

### Video 1

- Edge detection worked clearly on objects.
- Gaussian Blur reduced image noise before edge detection.
- The processed video was saved successfully.

### Video 2

- More moving objects produced more visible edges.
- Blur removed small unwanted details.
- The processed output remained smooth.

### Video 3

- Bright areas showed clearer edges.
- Dark regions contained fewer visible edges.
- The processing worked correctly throughout the video.

## Example Images

The **examples** folder contains sample screenshots from the processed videos.

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the application:

   python app.py

3. Upload a video.

4. The application will process the video and return the processed output.

## Output

The processed video contains:

- Grayscale Conversion
- Gaussian Blur
- Canny Edge Detection
