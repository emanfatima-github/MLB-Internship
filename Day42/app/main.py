from fastapi import FastAPI

from app.routes.video import router as video_router


app = FastAPI(
    title="AI Video Processing API",
    description="FastAPI + YOLO Video Processing System",
    version="1.0.0"
)


app.include_router(video_router)


@app.get("/")
def home():
    return {
        "message": "AI Video Processing API is running"
    }
