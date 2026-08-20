
import cv2
import time
from pathlib import Path

from app.services.detector import model


def process_video(input_path: str, output_path: str, job_id: str, jobs: dict):

    start_time = time.time()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open or corrupted video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video is empty or invalid")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    processed_frames = 0
    total_detections = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = model(frame, verbose=False)

            detection_count = 0

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    detection_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    label = f"{class_name} {confidence:.2f}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            processed_frames += 1
            total_detections += detection_count

            progress = int(
                (processed_frames / total_frames) * 100
            )

            cv2.putText(
                frame,
                f"Frame: {processed_frames}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            jobs[job_id]["progress"] = progress
            jobs[job_id]["processed_frames"] = processed_frames
            jobs[job_id]["total_detections"] = total_detections

    finally:

        cap.release()
        writer.release()

    processing_time = time.time() - start_time

    average_fps = (
        processed_frames / processing_time
        if processing_time > 0
        else 0
    )

    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "average_fps": round(average_fps, 2),
        "processing_time": round(processing_time, 2)
    }