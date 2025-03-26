import torch
from IPython.display import Image
import shutil
import os
import cv2
import warnings
warnings.filterwarnings('ignore')



# Save Video Frames every n seconds
def save_frames(video_path, n_seconds, output_base='./frames'):
  """
  Saves frames from a video at n_seconds intervals.
  Params:
    - video_path (str): path to .mp4 input
    - n_seconds (int): interval in seconds at which a frame is captures
    - output_base (str): base path to save frames into
  """

  # Delete and recreate the output dir (for new runs w/ diff)
  if os.path.exists(output_base):
    shutil.rmtree(output_base)
  os.makedirs(output_base)

  # Open the video file
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError(f"Could not open video at {video_path}")

  # get video properties
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration_sec = int(total_frames / fps)

  # compute frame interval
  frame_interval = int(fps * n_seconds)

  # extract and save frames
  frame_id = 0
  saved_count = 0
  while frame_id < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    if not success:
      break

    # Save file
    filename = os.path.join(output_base, f"{str(saved_count + 1).zfill(5)}.jpg")
    cv2.imwrite(filename, frame)
    saved_count += 1
    frame_id += frame_interval

  cap.release()
  print(f"Saved {saved_count} frames to {output_base}")



# BATCHING LOGIC
def batch(iterable, batch_size=32):
  for i in range(0, len(iterable), batch_size):
    yield iterable[i:i+batch_size]

# running YOLO in batches
def run_yolo_in_batches(model, image_paths, batch_size=64, save_dir="runs/detect/exp"):

  for i, batch_paths in enumerate(tqdm(batch(image_paths, batch_size), desc="Running YOLO")):
    results = model(batch_paths)
    results.save()
  print("YOLO detections complete.")

# Get last output
def get_latest_yolo_output(base_dir='runs/detect'):
  from pathlib import Path
  paths = sorted(Path(base_dir).glob('exp*'), key=os.path.getmtime)
  return str(paths[-1]) if paths else None

# Create gif 
def create_gif(image_dir, output_path='yolo_output.gif', fps=5):

  image_paths = sorted([
      os.path.join(image_dir, f)
      for f in os.listdir(image_dir)
      if f.endswith('.jpg')
  ])

  frames = []
  for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      frames.append(img_rgb)
    else:
      print(f"Failed to load image: {path}")

  imageio.mimsave(output_path, frames, fps=fps)
  print(f"GIF Saved to {output_path}")