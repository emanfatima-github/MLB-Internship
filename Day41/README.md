## API Explanation

### How FastAPI Connects with YOLO

FastAPI receives the uploaded image and sends it to the YOLO model through the detector service. 
The YOLO model performs object detection and returns the detected objects, confidence scores, and bounding box coordinates.

### `/predict` Endpoint

The `/predict` endpoint accepts an image and a confidence threshold. It sends the image to the YOLO model for inference 
and returns the detection results in JSON format.

Example:

`POST /predict?confidence=0.5`

### Request Format

The request uses `multipart/form-data`:

- `file`: JPG or PNG image
- `confidence`: Confidence threshold between `0` and `1`

### Response Format

The API returns JSON containing:

- Detection count
- Class ID
- Class name
- Confidence score
- Bounding box coordinates

### Error Handling

The API handles:

- Unsupported file types
- Empty files
- Invalid images
- Invalid confidence values
- Prediction errors

### Example API Response

```json
{
  "detection_count": 1,
  "predictions": [
    {
      "class_id": 0,
      "class_name": "object",
      "confidence": 0.87,
      "bbox": {
        "x1": 100.5,
        "y1": 80.2,
        "x2": 400.7,
        "y2": 350.4
      }
    }
  ]
}