from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2

from app.services.detector import predict_image
from app.schemas.response import PredictionResponse


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg"
}


@router.post("", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG or PNG image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        predictions, _ = predict_image(
            image_bytes,
            confidence
        )

        return {
            "detection_count": len(predictions),
            "predictions": predictions
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG or PNG image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        _, processed_image = predict_image(
            image_bytes,
            confidence
        )

        success, encoded_image = cv2.imencode(
            ".jpg",
            processed_image
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Could not process output image."
            )

        return StreamingResponse(
            BytesIO(encoded_image.tobytes()),
            media_type="image/jpeg"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )
