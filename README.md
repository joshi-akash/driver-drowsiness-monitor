# 🚗 Real-Time Driver Drowsiness Detection System 💤

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)
![dlib](https://img.shields.io/badge/dlib-Facial_Landmarks-orange)

A full-stack, web-based computer vision application designed to prevent fatigue-induced accidents by monitoring driver alertness in real-time. 

![Demo GIF Placeholder](https://via.placeholder.com/800x400?text=Replace+this+image+with+a+demo.gif+of+your+app+working)
*(Note: Replace the placeholder image above with a `demo.gif` showing the app detecting drowsiness)*

---

## ❓ Why This Was Made (The Problem)

Driver fatigue is a leading cause of severe road accidents worldwide. Micro-sleeps (brief, uncontrollable lapses into sleep) can occur with the driver's eyes partially open, making them difficult to self-diagnose. 

This project provides a **non-intrusive, accessible solution**. Instead of requiring expensive hardware, this system uses a standard web browser and a basic webcam to monitor facial micro-expressions. It serves as a proof-of-concept for integrating heavy computer vision pipelines (C++) with lightweight web frontends via asynchronous REST APIs.

---

## 🛠️ How It Was Made (The Architecture)

This project bridges the gap between a web browser and a Python computer vision environment.

### 1. The Frontend (Client-Side)
*   **Video Capture:** Uses the HTML5 `MediaDevices.getUserMedia()` API to access the local webcam.
*   **Frame Processing:** An invisible HTML5 `<canvas>` continuously captures frames from the video stream (~7 FPS).
*   **Data Transmission:** Frames are compressed into `base64` JPEG strings and sent asynchronously via the JavaScript `Fetch API` to the backend.

### 2. The Backend (Server-Side)
*   **Flask API:** A Python Flask server (`app.py`) listens on a `/detect` endpoint.
*   **Decoding:** It receives the `base64` string, decodes it into a byte array, and transforms it into an OpenCV (`cv2`) image matrix.
*   **Computer Vision Pipeline:** 
    *   Converts the image to grayscale for faster processing.
    *   Detects frontal faces using `dlib.get_frontal_face_detector()`.
    *   Extracts 68 specific facial coordinates using the `shape_predictor_68_face_landmarks.dat` model.

### 3. The Math: Eye Aspect Ratio (EAR)
The system isolates the coordinates for the left and right eyes. It calculates the **Eye Aspect Ratio (EAR)**, an elegant mathematical equation that tracks the distance between vertical and horizontal eye landmarks.

$$EAR = \frac{\vert{}\vert{}p_2 - p_6\vert{}\vert{} + \vert{}\vert{}p_3 - p_5\vert{}\vert{}}{2 \vert{}\vert{}p_1 - p_4\vert{}\vert{}}$$

*   **$p_1, ..., p_6$**: 2D landmark coordinates on the eye.
*   When the eye is open, the EAR remains relatively constant. 
*   When the eye blinks or closes, the EAR drops rapidly toward zero.
*   **The Trigger:** If the calculated EAR stays below our threshold (`0.25`) for `20` consecutive frames, the system flags the user as **DROWSY** and triggers a visual alarm.

---

## 📂 Project Structure

```text
driver-drowsiness-monitor/
│
├── app.py                      # Main Flask application and CV logic
├── requirements.txt            # Python dependencies
├── .gitignore                  # Prevents uploading large model files
│
├── static/                     # Static assets (CSS/JS/Images)
│
└── templates/
    └── index.html              # The frontend web interface