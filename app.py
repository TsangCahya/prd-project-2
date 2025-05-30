from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import sys
import os
from pathlib import Path
import logging
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"Base directory: {BASE_DIR}")

# Add YOLOv5 to path
YOLO_PATH = os.path.join(BASE_DIR, 'yolov5')
if os.path.exists(YOLO_PATH):
    sys.path.append(YOLO_PATH)
    logger.info(f"YOLOv5 path added: {YOLO_PATH}")
else:
    logger.error(f"YOLOv5 path not found at {YOLO_PATH}")

# Global variables
model = None
model_lock = Lock()
camera = None
camera_lock = Lock()

def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:  # Double-checked locking pattern
                try:
                    logger.info("Loading YOLOv5 model...")
                    # Use a smaller model size for faster loading
                    model = torch.hub.load(YOLO_PATH, 'custom', 
                                        path=str(BASE_DIR / 'models' / 'best.pt'), 
                                        source='local',
                                        device='cpu')  # Force CPU to avoid CUDA initialization
                    model.conf = 0.25
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    return None
    return model

def get_camera():
    global camera
    if camera is None:
        with camera_lock:
            if camera is None:
                try:
                    logger.info("Opening camera...")
                    camera = cv2.VideoCapture(0)
                    if not camera.isOpened():
                        raise Exception("Could not open camera")
                    logger.info("Camera opened successfully")
                except Exception as e:
                    logger.error(f"Error opening camera: {e}")
                    return None
    return camera

def generate_frames():
    camera = get_camera()
    if camera is None:
        logger.error("No camera available")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame")
                break

            try:
                model = get_model()
                if model is not None:
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    results = model(frame)
                    processed_frame = results.render()[0]
                else:
                    processed_frame = frame

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    logger.error("Failed to encode frame")
                    continue

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"Camera error: {e}")
    finally:
        if camera is not None:
            camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    model_loaded = get_model() is not None
    camera_available = get_camera() is not None
    return jsonify({
        "model_loaded": model_loaded,
        "camera_available": camera_available
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port) 