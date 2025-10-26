# train_yolov11.py


from ultralytics import YOLO
import os

# --- Configuration ---
# Set the path to your dataset configuration file
DATASET_CONFIG = 'D:\\06 - ACIE Project\\dataset\\data.yaml'

# Select your model size (n=nano, s=small, m=medium, l=large, x=x-large)
# The .pt file will be downloaded automatically if not present.
MODEL_WEIGHTS = 'yolo11m.pt' 

# Training Hyperparameters
EPOCHS = 50        # Total number of training cycles
IMAGE_SIZE = 640     # Input image dimension (square)
BATCH_SIZE = 16      # Number of images per batch (adjust based on your GPU memory)
PROJECT_NAME = 'YOLOv11_Skin_Detection_Project' # Main results folder
RUN_NAME = 'Yolo_medium_50_Epochs' # Specific run folder within the project

# --- Training Execution ---
if __name__ == '__main__':
    try:
        print(f"Loading model: {MODEL_WEIGHTS}")
        # Initialize the model with pre-trained weights
        model = YOLO(MODEL_WEIGHTS)

        print(f"Starting training for {EPOCHS} epochs...")
        
        # Start the training process
        results = model.train(
            data=DATASET_CONFIG,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            name=RUN_NAME,
            project=PROJECT_NAME,
            # device=0, # Uncomment and set to a specific GPU ID if needed
            # patience=50, # Optional: stop training if validation metrics don't improve after 50 epochs
            # cache='ram' # Optional: Caching images can speed up training significantly
        )

        print("\n--- Training Complete ---")
        # The best model is saved at: PROJECT_NAME/RUN_NAME/weights/best.pt
        best_model_path = os.path.join(
            'runs', 'detect', PROJECT_NAME, RUN_NAME, 'weights', 'best.pt'
        )
        print(f"Best model saved to: {best_model_path}")
        
        # --- Optional: Run validation/testing on the best model ---
        print("\nRunning final validation...")
        best_model = YOLO(best_model_path)
        metrics = best_model.val()
        print(f"Validation mAP50-95: {metrics.box.map}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure 'ultralytics' is installed and your 'skin_dataset.yaml' file and data paths are correct.")