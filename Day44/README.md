# Day-43 Production-Ready AI Video API

## Overview

This project improves the Day-42 FastAPI + YOLO video processing API.

The API now validates user input, handles errors, creates unique request and job IDs, and records important events using structured logging.

## Features

- Video upload
- YOLO video inference
- Background video processing
- Maximum file size validation
- Video format validation
- Confidence threshold validation
- HTTP status codes
- Global exception handling
- Structured logging
- Unique request IDs
- Unique job IDs
- Processing time logging
- Health check endpoint

## Validation

The API checks:

- File is provided
- File is not empty
- File format is supported
- File size is not greater than 100 MB
- Confidence threshold is between 0 and 1
- Job ID exists

Supported formats:

- MP4
- AVI
- MOV
- MKV

## Error Handling

The API handles:

- Unsupported files
- Large files
- Empty files
- Corrupted videos
- Invalid confidence values
- Invalid job IDs
- Missing parameters
- Unexpected server errors

The API returns clear error messages instead of crashing.

## Logging

Logs are stored in:

logs/app.log

Important events include:

- Video upload received
- Job started
- YOLO model loaded
- Processing completed
- Unsupported file format
- Video processing failure
- Request validation failure

Sensitive information is not stored in the logs.

## HTTP Status Codes

| Status Code | Meaning |
|-------------|---------|
| 200 | Successful request |
| 202 | Processing job accepted |
| 400 | Bad request |
| 404 | Job not found |
| 409 | Processing is still in progress |
| 413 | File too large |
| 415 | Unsupported video format |
| 422 | Validation error |
| 500 | Internal server error |

## API Endpoints

### GET /

Checks that the API is running.

### GET /health

Returns:

- API status
- Model status
- Application version

### POST /video/process

Uploads a video and starts background processing.

### GET /video/status/{job_id}

Checks the processing status of a video job.

### GET /video/result/{job_id}

Returns the processed video.

## Failed Request Examples

### Example 1: Unsupported Format

Request:

Upload `test.pdf`

Response:

```json
{
  "success": false,
  "error": "Unsupported video format",
  "request_id": "req_12345"
}