import streamlit as st
import cv2
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(page_title="Intelligent Security Monitoring System")

st.title(" Intelligent Security Monitoring System")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video = "processed_video.mp4"

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    os.makedirs("snapshots", exist_ok=True)

    ROI = (150,100,500,450)

    inside_people = set()
    entry_time = {}
    stay_time = {}

    logs = []

    total_entries = 0
    total_exits = 0
    max_occupancy = 0

    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        progress.progress(min(frame_no/total_frames,1.0))

        x1,y1,x2,y2 = ROI

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(frame,"ROI",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,255),2)

        results = model.track(frame,persist=True,verbose=False)

        current_inside = set()

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box,track_id,cls in zip(boxes,ids,classes):

                if cls != 0:
                    continue

                bx1,by1,bx2,by2 = map(int,box)

                cx = (bx1+bx2)//2
                cy = (by1+by2)//2

                cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,255,0),2)
                cv2.putText(frame,f"ID {track_id}",
                            (bx1,by1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

                if x1<cx<x2 and y1<cy<y2:

                    current_inside.add(track_id)

                    if track_id not in inside_people:

                        inside_people.add(track_id)

                        total_entries += 1

                        now = datetime.now()

                        entry_time[track_id]=now

                        logs.append([
                            track_id,
                            "Entry",
                            now.strftime("%Y-%m-%d %H:%M:%S")
                        ])

                        cv2.imwrite(
                            f"snapshots/person_{track_id}.jpg",
                            frame
                        )

            exited = inside_people-current_inside

            for pid in exited:

                total_exits += 1

                now = datetime.now()

                duration = (now-entry_time[pid]).total_seconds()

                stay_time[pid]=duration

                logs.append([
                    pid,
                    "Exit",
                    now.strftime("%Y-%m-%d %H:%M:%S")
                ])

                inside_people.remove(pid)

                del entry_time[pid]

        max_occupancy=max(max_occupancy,len(inside_people))

        cv2.putText(frame,
                    f"Inside:{len(inside_people)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2)

        cv2.putText(frame,
                    f"Entry:{total_entries}",
                    (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    f"Exit:{total_exits}",
                    (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

        writer.write(frame)

    cap.release()
    writer.release()

    df = pd.DataFrame(
        logs,
        columns=[
            "Tracking ID",
            "Event Type",
            "Timestamp"
        ]
    )

    csv_file = "event_logs.csv"

    df.to_csv(csv_file,index=False)

    if len(stay_time)>0:
        avg_time = sum(stay_time.values())/len(stay_time)
    else:
        avg_time = 0

    st.success("Processing Complete")

    st.subheader("Summary")

    st.write("Current People Inside:",len(inside_people))
    st.write("Total Entries:",total_entries)
    st.write("Total Exits:",total_exits)
    st.write("Maximum Occupancy:",max_occupancy)
    st.write(f"Average Time Inside: {avg_time:.2f} seconds")

    st.video(output_video)

    with open(output_video,"rb") as f:
        st.download_button(
            "Download Processed Video",
            f,
            file_name="processed_video.mp4"
        )

    with open(csv_file,"rb") as f:
        st.download_button(
            "Download Event Log",
            f,
            file_name="event_logs.csv"
        )