from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import sys
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Add YOLOv5 to path
YOLO_PATH = os.path.expanduser('/Users/tsang/yolov5')
if os.path.exists(YOLO_PATH):
    sys.path.append(YOLO_PATH)
else:
    print(f"Warning: YOLOv5 path not found at {YOLO_PATH}")

# Load YOLOv5 model
try:
    model = torch.hub.load(YOLO_PATH, 'custom', 
                         path=str(BASE_DIR / 'models' / 'best.pt'), 
                         source='local')
    model.conf = 0.25  # Set confidence threshold
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def generate_frames():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            try:
                if model is not None:
                    # Run YOLOv5 detection
                    results = model(frame)
                    # Get the processed frame with detections
                    processed_frame = results.render()[0]
                else:
                    processed_frame = frame
                
                # Convert to jpg format
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
                
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        if 'camera' in locals():
            camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001))) 