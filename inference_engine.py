import cv2
import numpy as np
from ultralytics import YOLO
import time

class DetectionEngine:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLO detection engine.
        :param model_path: Path to the model file (e.g., 'yolov8n.pt', 'yolov10n.pt')
        """
        print(f"Loading model: {model_path}...")
        self.model = YOLO(model_path)
        self.model_name = model_path
        
    def predict(self, frame, conf=0.25, imgsz=640):
        """
        Run inference on a single frame.
        :param frame: OpenCV image (BGR)
        :param conf: Confidence threshold
        :param imgsz: Inference image size
        :return: annotated_frame, results
        """
        results = self.model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame, results

    def get_detections(self, results):
        """
        Extract detection details from results.
        """
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                name = self.model.names[int(cls)]
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": int(cls),
                    "name": name
                })
        return detections

if __name__ == "__main__":
    # Test with a blank image if run directly
    engine = DetectionEngine('yolov8n.pt')
    blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
    annotated, results = engine.predict(blank_image)
    print("Inference successful. Found:", len(engine.get_detections(results)), "objects.")
