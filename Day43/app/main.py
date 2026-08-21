import logging
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routes.video import router as video_router
from app.services.detector import load_model
from app.utils.file_utils import create_directories
from app.utils.logging_config import logger


APP_VERSION = "1.0.0"

app = FastAPI(
    title="Production-Ready AI Video API",
    description="FastAPI + YOLO video processing API with validation and logging",
    version=APP_VERSION
)

model_loaded = False


@app.on_event("startup")
def startup_event():
    global model_loaded

    create_directories()

    try:
        load_model()
        model_loaded = True

        logger.info("YOLO model loaded successfully")

    except Exception as e:
        model_loaded = False

        logger.error(
            f"YOLO model loading failed: {str(e)}"
        )


@app.middleware("http")
async def add_request_id(request: Request, call_next):

    start_time = time.time()

    try:
        response = await call_next(request)

        processing_time = time.time() - start_time

        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"time={processing_time:.3f}s"
        )

        return response

    except Exception as e:

        logger.error(
            f"Unhandled request error: {str(e)}"
        )

        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    logger.warning(
        f"Request validation failed: {exc.errors()}"
    )

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    logger.error(
        f"Unhandled server error: {str(exc)}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "Something went wrong while processing the request."
        }
    )


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Production-Ready AI Video API is running",
        "version": APP_VERSION
    }


@app.get("/health")
def health():

    return {
        "success": True,
        "api_status": "healthy",
        "model_status": "loaded" if model_loaded else "not loaded",
        "version": APP_VERSION
    }


app.include_router(video_router)