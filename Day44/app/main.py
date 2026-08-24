from fastapi import FastAPI

from .database import engine, Base
from .routers.video import router as video_router


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI Video Processing API",
    description="FastAPI + YOLO + SQLite + SQLAlchemy",
    version="1.0.0"
)


app.include_router(video_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "SQLite"
    }