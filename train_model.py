import os
from ultralytics import YOLO

def train():
    # Path to existing model weights
    model_path = os.path.join("models", "aerominds_dumping_v3.pt")
    
    # Check if the v3 model exists, otherwise fallback to v2
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "aerominds_dumping_v2.pt")
        
    print(f"Loading model weights from {model_path}...")
    model = YOLO(model_path)
    
    # Path to the data.yaml file
    data_yaml = os.path.join("training_data", "Aerial-Dumping-Sites.v6i.yolov8", "data.yaml")
    
    # Training configuration
    epochs = 100
    imgsz = 640
    batch = 16
    
    print(f"Starting training for {epochs} epochs on dataset {data_yaml}...")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs",
        name="aerominds_dumping_v4",
        exist_ok=True
    )
    
    print("Training complete!")
    print(f"Best weights saved to: {os.path.join('runs', 'aerominds_dumping_v4', 'weights', 'best.pt')}")

if __name__ == "__main__":
    train()
