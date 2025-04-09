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

with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection to GIF")
    gr.Markdown("Upload a video and receive an annotated object detection GIF using YOLOv8n.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## 🚀 How to Use
            1. Upload your video file (supported formats: mp4, mov, avi)  
            2. Wait for processing - this may take a few moments depending on video length  
            3. The result will be displayed as an animated GIF showing object detections
            """)
        with gr.Column():
            gr.Markdown("""
            ## ✨ Features
            - Real-time object detection using YOLOv8n model  
            - Automatic conversion of video to annotated GIF format  
            - Support for multiple common video formats  
            - Bounding box visualization with class labels and confidence scores
            """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## ⚠️ Important Notes
            - Processing is limited to 400 frames maximum  
              - Longer videos will be automatically truncated  
              - For best results, trim long videos before uploading  
            - YOLOv8n is optimized for speed with good accuracy  
            - Works best on clear, well-lit footage  
            - Can detect 80 object classes, including:  
              - Common objects (person, car, chair, etc.)  
              - Animals (dog, cat, bird, etc.)  
              - Vehicles (bicycle, motorcycle, truck, etc.)
            """)
        with gr.Column():
            gr.Markdown("""
            ## 🛠️ Technical Details
            - Backend: Ultralytics YOLOv8 implementation  
            - Each frame is processed individually for detection  
            - Output: Animated GIF for easy viewing & sharing  
            - Resolution: Original video resolution is retained  
              - Very high-resolution videos may process slower
            """)

    gr.Markdown(
        "### 🧪 Share your experience & report bugs here: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfT0rNwGL4CJitVAanQzqfIOomwoQbAF0L0caiAh_ZhuZiv_g/viewform?usp=sharing)"
    )

    with gr.Row():
        video_input = gr.Video(label="Upload Video (mp4, mov, avi, etc.)")
        output_gif = gr.Image(type="filepath", label="YOLOv8 GIF")

    submit_btn = gr.Button("Generate GIF")

    submit_btn.click(fn=handle_video_upload, inputs=video_input, outputs=output_gif)

demo.launch()

