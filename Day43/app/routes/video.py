import os
import time
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Query,
    UploadFile,
    status
)
from fastapi.responses import FileResponse

from app.services.video_processor import process_video
from app.utils.file_utils import (
    MAX_FILE_SIZE,
    UPLOAD_DIR,
    OUTPUT_DIR,
    generate_job_id,
    generate_request_id,
    is_allowed_file,
    get_file_extension
)
from app.utils.logging_config import logger


router = APIRouter(prefix="/video", tags=["Video"])

jobs = {}


def process_video_background(
    input_path: str,
    output_path: str,
    confidence: float,
    request_id: str,
    job_id: str
):
    try:
        jobs[job_id]["status"] = "processing"

        result = process_video(
            input_path=input_path,
            output_path=output_path,
            confidence=confidence,
            request_id=request_id,
            job_id=job_id
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

        logger.error(
            f"Video processing failed: {str(e)}",
            extra={
                "request_id": request_id,
                "job_id": job_id
            }
        )

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="YOLO confidence threshold between 0 and 1"
    )
):
    request_id = generate_request_id()

    logger.info(
        "Video upload received",
        extra={"request_id": request_id}
    )

    if not file.filename:
        logger.warning(
            "Empty filename received",
            extra={"request_id": request_id}
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "No video file provided",
                "request_id": request_id
            }
        )

    if not is_allowed_file(file.filename):
        logger.warning(
            "Unsupported file format",
            extra={"request_id": request_id}
        )

        raise HTTPException(
            status_code=415,
            detail={
                "success": False,
                "error": "Unsupported video format",
                "request_id": request_id
            }
        )

    job_id = generate_job_id()

    extension = get_file_extension(file.filename)

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}{extension}"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_processed.mp4"
    )

    total_size = 0

    try:
        with open(input_path, "wb") as buffer:

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    buffer.close()

                    if os.path.exists(input_path):
                        os.remove(input_path)

                    logger.warning(
                        "Video file exceeds maximum size",
                        extra={
                            "request_id": request_id,
                            "job_id": job_id
                        }
                    )

                    raise HTTPException(
                        status_code=413,
                        detail={
                            "success": False,
                            "error": "Video file is too large. Maximum size is 100 MB.",
                            "request_id": request_id
                        }
                    )

                buffer.write(chunk)

    except HTTPException:
        raise

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)

        logger.error(
            f"File upload failed: {str(e)}",
            extra={
                "request_id": request_id,
                "job_id": job_id
            }
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Failed to save uploaded video",
                "request_id": request_id
            }
        )

    if total_size == 0:
        if os.path.exists(input_path):
            os.remove(input_path)

        logger.warning(
            "Empty video file received",
            extra={
                "request_id": request_id,
                "job_id": job_id
            }
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Uploaded video is empty",
                "request_id": request_id
            }
        )

    jobs[job_id] = {
        "status": "queued",
        "request_id": request_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": confidence
    }

    background_tasks.add_task(
        process_video_background,
        input_path,
        output_path,
        confidence,
        request_id,
        job_id
    )

    return {
        "success": True,
        "message": "Video processing started",
        "request_id": request_id,
        "job_id": job_id,
        "status": "queued"
    }


@router.get("/status/{job_id}")
async def get_video_status(job_id: str):
    request_id = generate_request_id()

    if job_id not in jobs:
        logger.warning(
            "Invalid job ID requested",
            extra={
                "request_id": request_id,
                "job_id": job_id
            }
        )

        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Job not found",
                "request_id": request_id
            }
        )

    job = jobs[job_id]

    return {
        "success": True,
        "request_id": job["request_id"],
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error")
    }


@router.get("/result/{job_id}")
async def get_video_result(job_id: str):
    request_id = generate_request_id()

    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Job not found",
                "request_id": request_id
            }
        )

    job = jobs[job_id]

    if job["status"] == "queued":
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error": "Video processing has not started yet",
                "request_id": request_id
            }
        )

    if job["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error": "Video is still being processed",
                "request_id": request_id
            }
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": job.get("error", "Video processing failed"),
                "request_id": request_id
            }
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_processed.mp4"
    )

    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Processed video not found",
                "request_id": request_id
            }
        )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}_processed.mp4"
    )