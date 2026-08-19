from fastapi import FastAPI
from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="Custom YOLO Prediction API",
    description="REST API for custom YOLO object detection",
    version="1.0.0"
)


app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "message": "Custom YOLO Prediction API is running"
    }
