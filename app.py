import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import tempfile
from main import process_and_return_gif, load_model

def handle_video_upload(video_path):
    if not video_path:
        raise ValueError("No video path provided")
    gif_path = process_and_return_gif(video_path, model=load_model())
    print("Returning GIF Path:", gif_path)
    return gif_path

gr.Interface(
    fn=handle_video_upload,
    inputs=gr.Video(label="Upload MP4 Video"),
    outputs=gr.Image(type="filepath", label="YOLOv8 GIF"),
    title="YOLO Object Detection to GIF",
    description="Upload a video and receive an annotated object detection GIF using YOLOv8n."
).launch()
