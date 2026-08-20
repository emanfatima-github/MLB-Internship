# Video Processing API

### How It Works

The API accepts a video, processes each frame using YOLO, detects objects, draws bounding boxes, and saves the processed video.

### Background Processing

Background processing allows the API to process videos without making the user wait. A `job_id` is returned to check the progress.

### Job Workflow

`Upload → Job Created → Processing → Completed → Download Result`

### API Endpoints

* `POST /video/process` — Upload video
* `GET /video/status/{job_id}` — Check status
* `GET /video/result/{job_id}` — Download processed video

### Performance

The API records total frames, processed frames, detections, FPS, and processing time.

