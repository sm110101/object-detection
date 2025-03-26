import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import tempfile
from main import process_and_return_gif, load_model

def handle_video_upload(video):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid:
        temp_vid.write(video.read())
        temp_vid.flush()
        gif = process_and_return_gif(temp_vid.name, model=load_model())
        return gif

gr.Interface(
    fn=handle_video_upload,
    inputs=gr.Video(label="Upload MP4 Video"),
    outputs=gr.Image(type="filepath", label="YOLOv8 GIF"),
    title="YOLO Object Detection to GIF",
    description="Upload a video and receive an annotated object detection GIF using YOLOv8n."
).launch()
