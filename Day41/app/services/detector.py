from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))


def predict_image(image_bytes: bytes, confidence: float = 0.5):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:
        for box in result.boxes:
            coordinates = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2)
                }
            })

    processed_image = result.plot()

    return predictions, processed_image