from ultralytics import YOLO
import cv2
import imageio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from io import BytesIO
import tempfile
import os
from datetime import datetime
from pytz import timezone

def load_model(weights_path="yolov8n.pt"):
    return YOLO(weights_path)

def process_and_return_gif(video_path, model, batch_size=16, skip_frames=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    batch = []
    all_outputs = []
    frame_idx = 0
    MAX_FRAMES = 400

    while True:
        if len(frames) >= MAX_FRAMES:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frames == 0:
            frames.append(frame)
            batch.append(frame)
            if len(batch) == batch_size:
                results = model(batch)
                for r in results:
                    all_outputs.append(r.plot())
                batch = []
        frame_idx += 1

    if batch:
        results = model(batch)
        for r in results:
            all_outputs.append(r.plot())

    annotated_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_outputs]
    
    # Get original video FPS
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Adjust FPS based on skip frames
    output_fps = original_fps // skip_frames

    cap.release()

    # Handle time logging
    et_timezone = timezone('US/Eastern')
    current_datetime_et = datetime.now(et_timezone)

    # save temp file
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_gif:
        output_path = temp_gif.name
        print(f"Saving file to path: {output_path}")
        imageio.mimsave(output_path, annotated_frames, format='GIF', fps=output_fps)
        print("Saved")
        print("Exists?", os.path.exists(output_path))
        print(f"\nExecuted: {current_datetime_et}")
        return output_path
    