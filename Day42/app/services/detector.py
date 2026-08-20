from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))
