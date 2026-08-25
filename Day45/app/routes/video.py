import os
import time
import uuid

import cv2

from datetime import datetime
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile
)
from sqlalchemy.orm import Session
from ultralytics import YOLO

from .. import crud
from ..database import get_db
from ..dependencies import get_current_user
from ..models import Job, User
from ..schemas import JobResponse


router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/best.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


@router.post(
    "/video/process",
    response_model=JobResponse
)
async def process_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    job_id = str(uuid.uuid4())

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}{extension}"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_processed.mp4"
    )

    try:
        content = await file.read()

        with open(input_path, "wb") as buffer:
            buffer.write(content)

        job = Job(
            job_id=job_id,
            user_id=current_user.id,
            filename=file.filename,
            status="processing",
            created_at=datetime.utcnow(),
            total_detections=0
        )

        crud.create_job(db, job)

        start_time = time.time()

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        total_detections = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = model(
                frame,
                verbose=False
            )

            for result in results:

                if result.boxes is not None:
                    total_detections += len(
                        result.boxes
                    )

                annotated_frame = result.plot()

                writer.write(
                    annotated_frame
                )

        cap.release()
        writer.release()

        processing_time = time.time() - start_time

        crud.update_job(
            db,
            job_id,
            status="completed",
            completed_at=datetime.utcnow(),
            processing_time=round(
                processing_time,
                2
            ),
            total_detections=total_detections,
            output_file=output_path
        )

        return crud.get_job(
            db,
            job_id
        )

    except HTTPException:
        raise

    except Exception as e:

        crud.update_job(
            db,
            job_id,
            status="failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )


@router.get(
    "/jobs",
    response_model=list[JobResponse]
)
def get_jobs(
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    allowed_statuses = {
        "queued",
        "processing",
        "completed",
        "failed"
    }

    if status and status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Invalid status."
        )

    if current_user.role == "admin":
        return crud.get_jobs(
            db,
            status=status
        )

    return crud.get_jobs(
        db,
        status=status,
        user_id=current_user.id
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    job = crud.get_job(
        db,
        job_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if (
        current_user.role != "admin"
        and job.user_id != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this job."
        )

    return job


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    job = crud.get_job(
        db,
        job_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if (
        current_user.role != "admin"
        and job.user_id != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to delete this job."
        )

    if (
        job.output_file
        and os.path.exists(job.output_file)
    ):
        os.remove(job.output_file)

    result = crud.delete_job(
        db,
        job_id
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return {
        "message": "Job deleted successfully",
        "job_id": job_id
    }