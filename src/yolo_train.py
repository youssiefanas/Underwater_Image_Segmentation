import os
from ultralytics import YOLO
import torch
import yaml

# ================================
# Paths Configuration
current_dir = os.getcwd()
# ================================
# Update these paths with your dataset and configuration
DATASET_PATH = os.path.join(current_dir, 'dataset', 'data.yaml')
PRETRAINED_WEIGHTS = os.path.join(current_dir, 'yolo11x-seg.pt') #you can change this to the model available , but i hope you can run this on the xl , just check how much you need
OUTPUT_DIR = os.path.join(current_dir, 'runs', 'segment', 'train')

# ================================
# Training Parameters
# ================================
EPOCHS = 50  # Number of epochs to train change this if needed but the minimum should be 30
BATCH_SIZE = 16  # Adjust based on GPU memory
IMG_SIZE = 640  # Input image size (adjust based on your dataset)
DEVICE = "cuda" 

# ================================
# Train YOLOv11 Segmentation Model
# ================================
def train_yolo():
    print("Starting YOLOv11 segmentation training...")
    
    # Initialize model
    model = YOLO(PRETRAINED_WEIGHTS)  # Load YOLOv11 segmentation pretrained weights

    # Train the model
    model.train(
        data=DATASET_PATH,         # Path to dataset.yaml
        epochs=EPOCHS,             # Number of epochs
        batch=BATCH_SIZE,          # Batch size
        imgsz=IMG_SIZE,            # Image size
        device=DEVICE,             # Device: 'cuda' or 'cpu'
        workers=8,                 # Number of workers for data loading
        project=OUTPUT_DIR,        # Where to save training results
        name="exp1",              # Experiment name
        exist_ok=True              # Overwrite if directory exists
    )

    print("Training complete! Results saved in:", OUTPUT_DIR)

# ================================
# Run the Training Script
# ================================
if __name__ == "__main__":
    train_yolo()
