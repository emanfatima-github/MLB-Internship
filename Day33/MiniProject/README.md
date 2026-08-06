## Problem Solved

This application automatically monitors a video to detect people entering and leaving a selected area. 
It helps track occupancy, record security events, and generate useful reports without manual monitoring.

## How Entry and Exit Detection Works

The system uses YOLOv8 to detect and track people with unique IDs. A predefined Region of Interest (ROI) is monitored. 
When a person's center point enters the ROI, an entry event is recorded. When the person leaves the ROI, an exit event is recorded.

## How Event Logging Works

Every entry and exit event is saved in a CSV file. Each record contains:
- Tracking ID
- Event Type (Entry/Exit)
- Timestamp

This creates a complete log of all detected events.

## Biggest Challenge

The biggest challenge was avoiding duplicate entry and exit events for the same person. 
This was solved by using YOLO tracking IDs, which allow the system to identify each person consistently across video frames.