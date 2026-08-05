## How People Counting Works

The system uses YOLOv8 to detect people in images and videos. It counts all detected people in each frame and displays the total number. 
For videos, ByteTrack is used to keep the same ID for each person while tracking.

## Difference Between Detection and Counting

-Detection: Finds and draws a bounding box around each person.
-Counting: Counts the total number of detected people.

## Challenges

- People overlapping each other in crowded scenes.
- Some people were partially hidden (occlusion).
- Fast movement sometimes reduced detection accuracy.

## Future Improvements

- Add entry and exit counting.
- Add region-based counting.
- Improve accuracy in crowded scenes.
- Support real-time webcam detection.
