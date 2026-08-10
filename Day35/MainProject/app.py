import cv2
import time
import pandas as pd
import gradio as gr
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def process_video(
    video_path,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
    image_size=640,
    frame_skip=1
):
    if video_path is None:
        return None, None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, "Could not open video."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        video_fps = 30

    x1 = int(width * roi_x1 / 100)
    y1 = int(height * roi_y1 / 100)
    x2 = int(width * roi_x2 / 100)
    y2 = int(height * roi_y2 / 100)

    output_path = "processed_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        video_fps,
        (width, height)
    )

    previous_inside = set()
    all_ids = set()
    entered_ids = set()
    exited_ids = set()
    events = []

    max_roi_objects = 0
    frame_number = 0
    processed_frames = 0
    processing_start = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % frame_skip != 0:
            out.write(frame)
            continue

        processed_frames += 1

        resized_frame = cv2.resize(
            frame,
            (image_size, image_size)
        )

        results = model.track(
            resized_frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        current_inside = set()

        if results[0].boxes is not None:

            for box in results[0].boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                all_ids.add(track_id)

                bx1, by1, bx2, by2 = map(
                    int,
                    box.xyxy[0]
                )

                bx1 = int(bx1 * width / image_size)
                bx2 = int(bx2 * width / image_size)
                by1 = int(by1 * height / image_size)
                by2 = int(by2 * height / image_size)

                center_x = int((bx1 + bx2) / 2)
                center_y = int((by1 + by2) / 2)

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                inside_roi = (
                    x1 <= center_x <= x2 and
                    y1 <= center_y <= y2
                )

                if inside_roi:
                    current_inside.add(track_id)

                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id} - {class_name}",
                    (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        entries = current_inside - previous_inside

        for track_id in entries:
            entered_ids.add(track_id)

            events.append({
                "Tracking ID": track_id,
                "Event Type": "Entry",
                "Timestamp": frame_number / video_fps
            })

        exits = previous_inside - current_inside

        for track_id in exits:
            exited_ids.add(track_id)

            events.append({
                "Tracking ID": track_id,
                "Event Type": "Exit",
                "Timestamp": frame_number / video_fps
            })

        previous_inside = current_inside.copy()

        max_roi_objects = max(
            max_roi_objects,
            len(current_inside)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            "ROI",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        elapsed = time.time() - processing_start
        fps = processed_frames / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"Current Objects: {len(current_inside)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Unique Objects: {len(all_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        out.write(frame)

    cap.release()
    out.release()

    total_time = time.time() - processing_start
    average_fps = (
        processed_frames / total_time
        if total_time > 0
        else 0
    )

    events_path = "events.csv"

    events_df = pd.DataFrame(
        events,
        columns=[
            "Tracking ID",
            "Event Type",
            "Timestamp"
        ]
    )

    events_df.to_csv(
        events_path,
        index=False
    )

    summary = (
        f"Total Objects: {len(all_ids)}\n"
        f"Total Entries: {len(entered_ids)}\n"
        f"Total Exits: {len(exited_ids)}\n"
        f"Maximum Objects in ROI: {max_roi_objects}\n"
        f"Average FPS: {average_fps:.2f}\n"
        f"Processing Time: {total_time:.2f} seconds"
    )

    return output_path, events_path, summary


def gradio_process(
    video,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
    image_size,
    frame_skip
):
    return process_video(
        video,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        image_size,
        frame_skip
    )


with gr.Blocks() as app:

    gr.Markdown("# Smart Video Analytics System")

    video_input = gr.Video(
        label="Upload Video"
    )

    with gr.Row():

        roi_x1 = gr.Slider(
            0, 100, 20,
            label="ROI X1 (%)"
        )

        roi_y1 = gr.Slider(
            0, 100, 20,
            label="ROI Y1 (%)"
        )

    with gr.Row():

        roi_x2 = gr.Slider(
            0, 100, 80,
            label="ROI X2 (%)"
        )

        roi_y2 = gr.Slider(
            0, 100, 80,
            label="ROI Y2 (%)"
        )

    image_size = gr.Radio(
        choices=[640, 480],
        value=640,
        label="Image Size"
    )

    frame_skip = gr.Radio(
        choices=[1, 2],
        value=1,
        label="Frame Skipping"
    )

    process_button = gr.Button(
        "Start Processing"
    )

    output_video = gr.Video(
        label="Processed Video"
    )

    summary = gr.Textbox(
        label="Analytics Summary"
    )

    events_file = gr.File(
        label="Download events.csv"
    )

    process_button.click(
        fn=gradio_process,
        inputs=[
            video_input,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
            image_size,
            frame_skip
        ],
        outputs=[
            output_video,
            events_file,
            summary
        ]
    )


app.launch()