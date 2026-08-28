# 🚗 Real-Time Driver Drowsiness Detection System 💤

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)
![dlib](https://img.shields.io/badge/dlib-Facial_Landmarks-orange)

A full-stack, web-based computer vision application designed to prevent fatigue-induced accidents by monitoring driver alertness in real-time. 


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
``` 

🚀 How to Use This Project
Because this project relies on a large pre-trained machine learning model (~97 MB), the model file is not hosted directly in this repository. Follow these steps to set up and run the application locally:

Step 1: Clone the Repository
Bash
git clone [https://github.com/joshi-akash/driver-drowsiness-monitor.git](https://github.com/joshi-akash/driver-drowsiness-monitor.git)
cd driver-drowsiness-monitor
Step 2: Set Up Virtual Environment & Dependencies
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
Step 3: Download the Pre-trained Landmark Model
Download the required model file: shape_predictor_68_face_landmarks.dat.bz2

Extract the .bz2 archive using 7-Zip or WinRAR.

Place the extracted shape_predictor_68_face_landmarks.dat file directly in the root folder alongside app.py.

Step 4: Run the Application
Bash
python app.py
Step 5: Open in Browser
Navigate to http://127.0.0.1:5000/ in your browser and grant permission to access the webcam.

🔮 Future Enhancements
Audio Alarms: Web Audio API integration for loud sound warnings when drowsiness is detected.

Yawn Detection: Measuring Mouth Aspect Ratio (MAR) to detect yawning before micro-sleep occurs.

Head Pose Estimation: Tracking head drop or side-to-side distraction.

👨‍💻 Author
Akash Joshi

📄 License
This project is open-source under the MIT License.
