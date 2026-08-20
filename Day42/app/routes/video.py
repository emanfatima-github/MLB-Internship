from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from pathlib import Path
import uuid
import shutil

from app.utils.file_utils import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    is_allowed_video
)

from app.services.video_processor import process_video


router = APIRouter(
    prefix="/video",
    tags=["Video"]
)


jobs = {}


def process_video_background(
    job_id: str,
    input_path: str,
    output_path: str
):

    try:

        jobs[job_id]["status"] = "processing"

        statistics = process_video(
            input_path,
            output_path,
            job_id,
            jobs
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["statistics"] = statistics

    except Exception as e:

        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@router.post("/process")
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="Video file is required"
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing"
        )

    if not is_allowed_video(file.filename):

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, AVI, MOV or MKV."
        )

    job_id = uuid.uuid4().hex[:8]

    extension = Path(file.filename).suffix.lower()

    input_path = UPLOAD_DIR / f"{job_id}{extension}"

    output_path = OUTPUT_DIR / f"{job_id}_processed.mp4"

    try:

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded video"
        )

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "processed_frames": 0,
        "total_detections": 0,
        "statistics": None,
        "error": None,
        "output_path": str(output_path)
    }

    background_tasks.add_task(
        process_video_background,
        job_id,
        str(input_path),
        str(output_path)
    )

    return {
        "job_id": job_id,
        "status": "processing"
    }


@router.get("/status/{job_id}")
async def get_video_status(job_id: str):

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"]
    }

    if job["status"] == "completed":
        response["statistics"] = job["statistics"]

    if job["status"] == "failed":
        response["error"] = job["error"]

    return response


@router.get("/result/{job_id}")
async def get_video_result(job_id: str):

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    job = jobs[job_id]

    if job["status"] != "completed":

        raise HTTPException(
            status_code=400,
            detail="Video processing is not completed yet"
        )

    output_path = Path(job["output_path"])

    if not output_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Processed video not found"
        )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=output_path.name
    )
