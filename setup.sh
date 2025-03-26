#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install torch torchvision opencv-python imageio tqdm gradio

echo "Freezing dependencies to requirements.txt..."
pip freeze > requirements.txt

echo "✅ Setup complete. To activate your environment, run:"
echo "source venv/bin/activate"

