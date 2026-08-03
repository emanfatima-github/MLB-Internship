## What is Object Tracking?

Object tracking is the process of detecting an object and following the same object across all video frames by assigning it a unique ID.

## Difference Between Detection and Tracking

* Object Detection: Finds objects in each frame but does not remember them.
* Object Tracking: Detects objects and keeps the same ID for each object as it moves through the video.

## Which Tracking Algorithm Was Used?

This project uses ByteTrack with the Ultralytics YOLOv8 model. ByteTrack helps keep object IDs consistent while tracking.

## Challenges Faced

* Processing videos smoothly while tracking multiple objects.

