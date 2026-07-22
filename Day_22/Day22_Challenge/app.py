import cv2
import os
import tempfile
import gradio as gr

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    output_width = 720
    output_height = 1280

    output_path = os.path.join(
        tempfile.gettempdir(),
        "processed_video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (output_width, output_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 100, 200)

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        out.write(edges)

        frame_count += 1

    cap.release()
    out.release()

    print("Frames Processed:", frame_count)

    return output_path


demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Real-Time Video Processing Tool",
    description="Upload a video to apply Grayscale, Gaussian Blur, and Canny Edge Detection."
)

demo.launch(
    inbrowser=True
)