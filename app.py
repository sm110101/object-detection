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

description = """Upload a video and receive an annotated object detection GIF using YOLOv8n.

## 🚀 How to Use
1. Upload your video file (supported formats: mp4, mov, avi)
2. Wait for processing - this may take a few moments depending on video length
3. The result will be displayed as an animated GIF showing object detections

## ✨ Features
- Real-time object detection using YOLOv8n model
- Automatic conversion of video to annotated GIF format
- Support for multiple common video formats
- Bounding box visualization with class labels and confidence scores

## ⚠️ Important Notes
- Processing is limited to 400 frames maximum to ensure reasonable processing times
  - Longer videos will be automatically truncated
  - For best results, consider trimming very long videos before uploading
- The model uses YOLOv8n, which is optimized for speed while maintaining good detection accuracy
- Detection works best on clear, well-lit footage
- The model can detect 80 different object classes including:
  - Common objects (person, car, chair, etc.)
  - Animals (dog, cat, bird, etc.)
  - Vehicles (bicycle, motorcycle, truck, etc.)

## 🛠️ Technical Details
- Backend: Uses the Ultralytics YOLOv8 implementation
- Processing: Each frame is individually processed for object detection
- Output: Results are compiled into an animated GIF for easy viewing and sharing
- Resolution: Input videos are processed at their original resolution
  - Very high resolution videos may take longer to process
"""

gr.Interface(
    fn=handle_video_upload,
    inputs=gr.Video(label="Upload Video (mp4, mov, avi, etc.)"),
    outputs=gr.Image(type="filepath", label="YOLOv8 GIF"),
    title="YOLO Object Detection to GIF",
    description=description
).launch()
