import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import re
import os

# --- Drowsiness Detection Logic ---

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Global state variables
frame_counter = 0
alarm_on = False

# --- Absolute Path to Your Model ---
# This is the hardcoded path you provided.
LANDMARK_PREDICTOR_PATH = r"C:\Users\joshi\Downloads\Drowsiness_Detector\Drowsiness_Detector\shape_predictor_68_face_landmarks.dat"
# ---

# Initialize dlib's face detector and landmark predictor
print("[INFO] Loading facial landmark predictor...")
print(f"[INFO] Looking for model at: {LANDMARK_PREDICTOR_PATH}")

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
except RuntimeError as e:
    print(f"[ERROR] Could not load '{LANDMARK_PREDICTOR_PATH}'.")
    print(f"[ERROR] Specific error: {e}")
    print("Please ensure the file exists at this exact path and is not corrupted.")
    exit()

# Get eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame):
    """Processes a single frame and returns the EAR and alarm status."""
    global frame_counter, alarm_on

    # Resize and convert to grayscale
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Process a smaller image for speed
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    current_ear = 0.0
    status = "awake"

    if len(rects) == 0:
        # No face detected.
        frame_counter = 0
        alarm_on = False
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        current_ear = ear

        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                alarm_on = True
        else:
            frame_counter = 0
            alarm_on = False
            
    status = "drowsy" if alarm_on else "awake"
    return status, current_ear

# --- Flask Server ---

app = Flask(__name__)
# Enable CORS to allow our JS frontend to call the Python backend
CORS(app) 

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

def base64_to_image(base64_string):
    """Converts a Base64 string to an OpenCV image."""
    # Remove the "data:image/jpeg;base64," prefix
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
        
    # Decode the string
    img_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    np_arr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode numpy array into an OpenCV image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route('/detect', methods=['POST'])
def detect():
    """Receives a video frame, processes it, and returns status."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Convert base64 string to OpenCV image
        frame = base64_to_image(image_data)
        
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Process the frame using our drowsiness logic
        status, ear = process_frame(frame)
        
        # Return the result
        return jsonify({"status": status, "ear": ear})

    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)