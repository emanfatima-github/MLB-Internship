from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import tempfile
import os


# Create FastAPI application
app = FastAPI()


# Load your custom YOLO model
model = YOLO("best.pt")


# GET /health
@app.get("/health")
def health():
    return {
        "status": "API is running",
        "model_loaded": True
    }


# POST /predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1. Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # 2. Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    try:
        # 3. Read uploaded image
        contents = await file.read()

        # Save image to temporary file
        temp_file.write(contents)
        temp_file.close()

        # 4. Validate that the file is actually a valid image
        try:
            image = Image.open(temp_file.name)
            image.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image."
            )

        # 5. Run YOLO inference
        results = model(temp_file.name)

        # Store detections
        detections = []

        # 6. Process YOLO results
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                confidence = float(box.conf[0])

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Add detection
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                })

        # 7. Return results
        return {
            "detections": detections,
            "total": len(detections)
        }

    finally:

        # 8. Delete temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)