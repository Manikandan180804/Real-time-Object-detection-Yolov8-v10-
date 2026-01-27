from flask import Flask, render_template, Response, request, jsonify
from inference_engine import DetectionEngine
import cv2
import os
import base64
import numpy as np

app = Flask(__name__)
# Initialize engine with YOLOv8n by default
engine = DetectionEngine('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    Handle frames sent from the client-side JavaScript.
    """
    try:
        data = request.json
        if not data or 'image' not in data:
            print("Error: No image data in request")
            return jsonify({"error": "No image data"}), 400

        # Decode base64 image
        img_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Error: Failed to decode image")
            return jsonify({"error": "Invalid image"}), 400

        # Perform detection
        annotated_frame, results = engine.predict(frame, add_timestamp=True)
        
        # Extract detections for UI
        detections = engine.get_detections(results)
        print(f"Input shape: {frame.shape}, Output shape: {annotated_frame.shape}, Detected: {len(detections)}")

        # Encode back to base64 to show on UI
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "image": f"data:image/jpeg;base64,{encoded_img}",
            "detections": detections
        })
    except Exception as e:
        print(f"Inference error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/change_model', methods=['POST'])
def change_model():
    global engine
    model_name = request.json.get('model', 'yolov8n.pt')
    try:
        engine = DetectionEngine(model_name)
        return jsonify({"status": "success", "message": f"Model changed to {model_name}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Ensure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Flask app on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
