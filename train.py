from ultralytics import YOLO
import argparse

def train_model(data_path, model_type='yolov8n.pt', epochs=50, imgsz=640):
    """
    Fine-tune a YOLO model on a custom dataset.
    :param data_path: Path to the data.yaml file (Roboflow format)
    :param model_type: Pre-trained model to start from
    :param epochs: Number of training epochs
    :param imgsz: Image size
    """
    print(f"Starting training with {model_type} on {data_path}...")
    model = YOLO(model_type)
    
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz,
        device='cpu' # Force CPU since CUDA is not available
    )
    print("Training complete. Results saved in 'runs/detect/train'")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pre-trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default 5 for testing)")
    
    args = parser.parse_args()
    train_model(args.data, args.model, args.epochs)
