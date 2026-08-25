from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"

model = None


def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)
    return model


def get_model():
    global model

    if model is None:
        load_model()

    return model