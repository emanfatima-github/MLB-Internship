import cv2
import time
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# -----------------------------
# Load YOLO Model
# -----------------------------

model = YOLO("yolov8n.pt")


# -----------------------------
# Video Processing
# -----------------------------

def process_video(
    video_path,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
    image_size=640,
    frame_skip=1
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, "Could not open video."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        video_fps = 30

    # ROI coordinates
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

    start_time = time.time()

    # -----------------------------
    # Process Frames
    # -----------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Frame skipping
        if frame_number % frame_skip != 0:
            out.write(frame)
            continue

        processed_frames += 1

        # Resize for inference
        resized_frame = cv2.resize(
            frame,
            (image_size, image_size)
        )

        # YOLO Tracking
        results = model.track(
            resized_frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        current_inside = set()

        # -----------------------------
        # Object Detection & Tracking
        # -----------------------------

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

                # Convert coordinates back
                bx1 = int(bx1 * width / image_size)
                bx2 = int(bx2 * width / image_size)
                by1 = int(by1 * height / image_size)
                by2 = int(by2 * height / image_size)

                center_x = int((bx1 + bx2) / 2)
                center_y = int((by1 + by2) / 2)

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Check ROI
                inside_roi = (
                    x1 <= center_x <= x2
                    and
                    y1 <= center_y <= y2
                )

                if inside_roi:
                    current_inside.add(track_id)

                # Bounding box
                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 255, 0),
                    2
                )

                # Tracking ID
                cv2.putText(
                    frame,
                    f"ID {track_id} - {class_name}",
                    (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

        # -----------------------------
        # Entry Detection
        # -----------------------------

        entries = current_inside - previous_inside

        for track_id in entries:

            entered_ids.add(track_id)

            events.append({
                "Tracking ID": track_id,
                "Event Type": "Entry",
                "Timestamp": round(
                    frame_number / video_fps,
                    2
                )
            })

        # -----------------------------
        # Exit Detection
        # -----------------------------

        exits = previous_inside - current_inside

        for track_id in exits:

            exited_ids.add(track_id)

            events.append({
                "Tracking ID": track_id,
                "Event Type": "Exit",
                "Timestamp": round(
                    frame_number / video_fps,
                    2
                )
            })

        previous_inside = current_inside.copy()

        # Maximum objects in ROI
        max_roi_objects = max(
            max_roi_objects,
            len(current_inside)
        )

        # -----------------------------
        # Draw ROI
        # -----------------------------

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

        # -----------------------------
        # FPS
        # -----------------------------

        elapsed_time = time.time() - start_time

        current_fps = (
            processed_frames / elapsed_time
            if elapsed_time > 0
            else 0
        )

        # -----------------------------
        # Display Statistics
        # -----------------------------

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
            f"FPS: {current_fps:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Save processed frame
        out.write(frame)

    # -----------------------------
    # Release Video
    # -----------------------------

    cap.release()
    out.release()

    total_time = time.time() - start_time

    average_fps = (
        processed_frames / total_time
        if total_time > 0
        else 0
    )

    # -----------------------------
    # Save Events CSV
    # -----------------------------

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

    # -----------------------------
    # Final Summary
    # -----------------------------

    summary = (
        f"Total Objects: {len(all_ids)}\n"
        f"Total Entries: {len(entered_ids)}\n"
        f"Total Exits: {len(exited_ids)}\n"
        f"Maximum Objects in ROI: {max_roi_objects}\n"
        f"Average FPS: {average_fps:.2f}\n"
        f"Processing Time: {total_time:.2f} seconds"
    )

    return output_path, events_path, summary


# -----------------------------
# Streamlit Interface
# -----------------------------

st.set_page_config(
    page_title="Smart Video Analytics",
    layout="wide"
)

st.title("Smart Video Analytics System")

st.write(
    "Upload a video to detect, track, count, "
    "and analyze moving objects."
)


# -----------------------------
# Upload Video
# -----------------------------

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)


# -----------------------------
# ROI Settings
# -----------------------------

st.subheader("ROI Settings")

col1, col2 = st.columns(2)

with col1:

    roi_x1 = st.slider(
        "ROI X1 (%)",
        0,
        100,
        20
    )

    roi_y1 = st.slider(
        "ROI Y1 (%)",
        0,
        100,
        20
    )

with col2:

    roi_x2 = st.slider(
        "ROI X2 (%)",
        0,
        100,
        80
    )

    roi_y2 = st.slider(
        "ROI Y2 (%)",
        0,
        100,
        80
    )


# -----------------------------
# Performance Settings
# -----------------------------

st.subheader("Processing Settings")

image_size = st.selectbox(
    "Image Size",
    [640, 480],
    index=0
)

frame_skip = st.selectbox(
    "Frame Skipping",
    [1, 2],
    index=0
)


# -----------------------------
# Start Processing
# -----------------------------

if st.button(
    "Start Processing",
    type="primary"
):

    if uploaded_video is None:

        st.warning(
            "Please upload a video first."
        )

    else:

        input_path = "input_video.mp4"

        with open(input_path, "wb") as file:

            file.write(
                uploaded_video.getbuffer()
            )

        progress = st.progress(0)

        with st.spinner(
            "Processing video..."
        ):

            output_path, events_path, summary = process_video(
                input_path,
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2,
                image_size,
                frame_skip
            )

        progress.progress(100)

        if output_path is None:

            st.error(summary)

        else:

            st.success(
                "Video processing completed!"
            )

            # -----------------------------
            # Summary
            # -----------------------------

            st.subheader(
                "Analytics Summary"
            )

            st.text(summary)

            # -----------------------------
            # Processed Video
            # -----------------------------

            st.subheader(
                "Processed Video"
            )

            st.video(output_path)

            # -----------------------------
            # Download CSV
            # -----------------------------

            st.subheader(
                "Events"
            )

            with open(
                events_path,
                "rb"
            ) as file:

                st.download_button(
                    label="Download events.csv",
                    data=file,
                    file_name="events.csv",
                    mime="text/csv"
                )