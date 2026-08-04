# Smart Vehicle Counting System

## How Vehicle Counting Works

The system uses YOLOv8 to detect vehicles in each frame of the video. A counting line is drawn on the road, and whenever a vehicle crosses the line, 
the vehicle count increases.

## How Tracking IDs Prevent Duplicate Counting

Each detected vehicle is given a unique tracking ID. The system counts a vehicle only once using its ID, even if it appears in many video frames. 
This prevents duplicate counting.

## Types of Vehicles Counted

* Car
* Motorcycle
* Bus
* Truck

