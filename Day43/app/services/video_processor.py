import cv2
import os
import time

from app.services.detector import get_model
from app.utils.logging_config import logger


def process_video(
    input_path: str,
    output_path: str,
    confidence: float,
    request_id: str,
    job_id: str
):
    start_time = time.time()

    logger.info(
        "Job started",
        extra={
            "request_id": request_id,
            "job_id": job_id
        }
    )

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video. The video may be corrupted.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise ValueError("Unable to create output video.")

    model = get_model()

    frame_count = 0
    detection_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False
            )

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            frame_count += 1

            if results[0].boxes is not None:
                detection_count += len(results[0].boxes)

    finally:
        cap.release()
        out.release()

    if frame_count == 0:
        raise ValueError("Video contains no readable frames.")

    processing_time = time.time() - start_time

    logger.info(
        f"Processing completed in {processing_time:.2f} seconds",
        extra={
            "request_id": request_id,
            "job_id": job_id
        }
    )

    return {
        "frames_processed": frame_count,
        "detections": detection_count,
        "processing_time_seconds": round(processing_time, 2)
    }