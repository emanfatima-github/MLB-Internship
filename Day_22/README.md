## How OpenCV Reads Videos

OpenCV reads a video one frame at a time. Each frame is treated like an image, so we can apply different image processing techniques to every frame.
After processing all the frames, OpenCV combines them to create a new video.

## What FPS Means

FPS stands for Frames Per Second. It tells us how many frames are shown every second in a video. A higher FPS makes the video look smoother.

## Processing Techniques Applied

In this project, I applied the following techniques:

- Converted each frame to grayscale.
- Applied Gaussian Blur to reduce noise.
- Applied Canny Edge Detection to detect the edges of objects.
- Saved the processed video as a new video file.

## Challenges Faced

While working with video frames, I faced some challenges such as video path errors, creating the output folder,
and saving the processed video correctly. I also had to make sure each frame was processed in the correct order so that the output video played smoothly.