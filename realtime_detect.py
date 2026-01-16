import cv2
from inference_engine import DetectionEngine
import time

def run_realtime(model_path='yolov8n.pt', source=0):
    """
    Run real-time object detection using a webcam or video source.
    :param model_path: Path to the YOLO model
    :param source: 0 for webcam, or path to a video file
    """
    engine = DetectionEngine(model_path)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting real-time detection. Press 'q' to quit.")
    
    # FPS calculation
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        annotated_frame, results = engine.predict(frame)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show results
        cv2.imshow("YOLO Real-time Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change to 'yolov10n.pt' once downloaded
    run_realtime('yolov8n.pt')
