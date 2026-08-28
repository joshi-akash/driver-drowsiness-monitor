<div align="center">

# 🚗 Real-Time Driver Drowsiness Detection System 💤

**A lightweight, full-stack computer vision web application for monitoring driver fatigue and preventing micro-sleep accidents in real time.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV2-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![dlib](https://img.shields.io/badge/dlib-Facial%20Landmarks-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

</div>

---

## 📌 Table of Contents

- [1. Problem Statement & Business Need](#1-problem-statement--business-need)
- [2. Tech Stack & Dependencies](#2-tech-stack--dependencies)
- [3. Technical Architecture & System Working](#3-technical-architecture--system-working)
- [4. Repository Directory Structure](#4-repository-directory-structure)
- [5. Setup & Installation Guide](#5-setup--installation-guide)
- [6. Future Enhancements & Roadmap](#6-future-enhancements--roadmap)
- [7. Author & License](#7-author--license)

---

## 1. Problem Statement & Business Need

### The Problem

Driver fatigue is one of the leading — and most underreported — causes of fatal road accidents worldwide. Unlike alcohol impairment, which can be measured objectively, fatigue is insidious: it creeps up gradually and often culminates in a **micro-sleep**, an involuntary lapse into sleep lasting anywhere from a fraction of a second to several seconds.

Micro-sleeps are particularly dangerous because:

- They frequently occur with the eyes **partially open or fully closed** for very brief intervals, making them nearly impossible for the driver to notice in themselves.
- At highway speeds, even a **1–2 second** lapse translates to the vehicle traveling the length of a football field with no active control input.
- Traditional countermeasures (rolling down a window, loud music, caffeine) treat symptoms of alertness rather than measuring the physiological signal directly.

### The Solution

This project implements a **non-intrusive, hardware-agnostic** fatigue monitoring system that runs entirely through a web browser:

- No specialized IR cameras, wearables, or in-cabin dedicated hardware — only a **standard webcam**.
- Continuous, real-time analysis of **facial landmarks** to compute eye closure state.
- A responsive web dashboard that streams live video, overlays detection metrics, and raises an active alert the moment drowsy behavior is detected — buying the driver critical seconds to react before a micro-sleep event occurs.

---

## 2. Tech Stack & Dependencies

### Categorized Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.8+ | Core application & CV processing logic |
| **Backend Framework** | Flask | Lightweight REST API server handling frame ingestion |
| **Computer Vision** | OpenCV (`opencv-python`) | Image decoding, preprocessing, grayscale conversion |
| **Facial Landmark ML** | dlib | Frontal face detection & 68-point landmark prediction |
| **Numerical Computing** | NumPy | Vectorized EAR calculations & coordinate array handling |
| **Frontend Markup/Style** | HTML5, CSS3 | Web dashboard structure & styling |
| **Frontend Logic** | JavaScript (ES6+) | Frame capture, Fetch API calls, Canvas rendering |
| **Frontend Capture API** | HTML5 Canvas API | Encoding raw webcam frames to Base64 JPEG |

### `requirements.txt` Breakdown

| Package | Role in the Project |
|---|---|
| `Flask` | Serves the web dashboard and exposes the `/detect` REST endpoint |
| `opencv-python` | Handles image decoding (`cv2.imdecode`), color-space conversion, and drawing overlays |
| `dlib` | Provides the pretrained frontal face detector and the 68-point facial landmark predictor |
| `numpy` | Performs the Euclidean distance math required for the EAR formula |
| `imutils` | Convenience utilities for resizing frames and simplifying `dlib` shape-to-NumPy-array conversions |

```txt
Flask
opencv-python
dlib
numpy
imutils
```

---

## 3. Technical Architecture & System Working

### End-to-End Data Flow

```
┌──────────────────────┐        Base64 JPEG POST         ┌───────────────────────┐
│   BROWSER (Client)    │ ───────────────────────────────▶ │   FLASK SERVER (app.py)│
│                        │        /detect endpoint          │                        │
│  1. getUserMedia()     │                                   │  2. Decode Base64      │
│  2. Canvas captures    │                                   │     → cv2.imdecode     │
│     frame @ ~7 FPS     │                                   │  3. Grayscale convert  │
│  3. Encode → Base64    │                                   │  4. dlib face detect   │
│  4. fetch() POST       │ ◀─────────────────────────────── │  5. 68-pt landmarks    │
│  5. Render JSON state  │        JSON { state, ear }        │  6. Compute EAR        │
│     on dashboard        │                                   │  7. Frame-count logic  │
└──────────────────────┘                                    └───────────────────────┘
```

### Step-by-Step Breakdown

**1. Frontend Capture**
The dashboard uses `navigator.mediaDevices.getUserMedia()` to access the webcam feed. An HTML5 `<canvas>` element captures a snapshot of the video stream at approximately **7 frames per second** — a rate chosen to balance detection responsiveness against network/CPU overhead. Each captured frame is encoded into a **Base64 JPEG string** and sent via an asynchronous `fetch()` POST request to the `/detect` endpoint.

**2. Backend Decoding**
The Flask server receives the Base64 payload, strips the data-URI header, and decodes the string back into a raw byte array. This byte array is then passed through `cv2.imdecode()` to reconstruct an OpenCV (NumPy `ndarray`) image matrix suitable for processing.

**3. Landmark Extraction**
The decoded frame is converted to grayscale (reducing computational load for detection). `dlib.get_frontal_face_detector()` locates the bounding box of the driver's face within the frame. Once a face is found, `shape_predictor_68_face_landmarks.dat` — a pretrained ensemble-of-regression-trees model — maps **68 (x, y) landmark coordinates** onto key facial features, including both eyes, eyebrows, nose, mouth, and jawline.

**4. Eye Aspect Ratio (EAR) Calculation**

The six landmark points surrounding each eye (`p1` through `p6`) are used to compute the **Eye Aspect Ratio**, a scale-invariant metric that stays roughly constant while the eye is open and drops sharply toward zero when the eye closes:

$$EAR = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2\,\lVert p_1 - p_4 \rVert}$$

Where the numerator sums the two vertical eye-landmark distances and the denominator is twice the horizontal eye-landmark distance. The EAR is computed independently for both eyes and averaged for stability against partial occlusion or head tilt.

**Detection & Alert Logic:**

| Condition | Resulting State |
|---|---|
| EAR ≥ 0.25 | `AWAKE` — counter resets |
| EAR < 0.25 for < 20 consecutive frames | `MONITORING` — counter increments |
| EAR < 0.25 for ≥ 20 consecutive frames | `DROWSY` — active alert triggered |

This consecutive-frame requirement is critical: it filters out normal, voluntary blinking (which typically lasts only 2–4 frames) while reliably catching sustained eye closure indicative of micro-sleep onset.

---

## 4. Repository Directory Structure

```
driver-drowsiness-monitor/
│
├── app.py                              # Flask server entrypoint & CV processing logic
│                                        #   - /detect POST endpoint
│                                        #   - dlib detector/predictor initialization
│                                        #   - EAR computation & state machine
│
├── requirements.txt                    # Pinned Python package dependencies
│
├── shape_predictor_68_face_landmarks.dat   # (Not tracked in git — downloaded manually)
│
├── .gitignore                          # Excludes:
│                                        #   *.dat            (large pretrained model files)
│                                        #   venv/            (virtual environment)
│                                        #   __pycache__/     (compiled Python bytecode)
│
├── templates/
│   └── index.html                      # Main web dashboard (video feed + status panel)
│
├── static/
│   ├── css/
│   │   └── style.css                   # Dashboard styling
│   └── js/
│       └── script.js                   # Webcam capture, Canvas encoding, fetch() polling
│
└── README.md                           # Project documentation (this file)
```

---

## 5. Setup & Installation Guide

Follow these five steps to get the application running locally.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/joshi-akash/driver-drowsiness-monitor.git
cd driver-drowsiness-monitor
```

### Step 2 — Set Up a Virtual Environment

It's strongly recommended to isolate project dependencies inside a virtual environment.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, install all required packages:

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` compilation can require CMake and a C++ build toolchain on some systems. If installation fails, ensure you have `cmake` installed (`pip install cmake`) and, on Windows, the Visual Studio Build Tools.

### Step 3 — Download the Facial Landmark Model

The 68-point landmark predictor model (`shape_predictor_68_face_landmarks.dat`, ~97 MB) is **not included in this repository** due to its size and is excluded via `.gitignore`. You must download it manually:

1. Download the compressed model from dlib's official model repository:
   `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`
2. Decompress the `.bz2` archive:
   - **Linux/macOS:** `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`
   - **Windows:** Use 7-Zip or any archive tool supporting `.bz2`
3. Place the resulting `shape_predictor_68_face_landmarks.dat` file in the **project root directory**, alongside `app.py`.

### Step 4 — Run the Application

```bash
python app.py
```

The Flask development server will start, typically on port `5000`.

### Step 5 — Open in Browser

Navigate to:

```
http://127.0.0.1:5000/
```

Grant webcam permissions when prompted by your browser. The dashboard will begin streaming your video feed and displaying live EAR values and drowsiness state.

---

## 6. Future Enhancements & Roadmap

- [ ] **Auditory Alerts** — Integrate the Web Audio API to trigger a real-time buzzer/alarm sound directly in the browser when the `DROWSY` state is reached, rather than relying solely on visual cues.
- [ ] **Yawn Detection** — Extend the landmark analysis with a **Mouth Aspect Ratio (MAR)** metric to detect prolonged yawning as an additional fatigue signal.
- [ ] **Head Pose Estimation** — Add 3D head pose tracking to detect driver distraction (e.g., looking away from the road) as a complementary safety signal alongside eye-closure monitoring.
- [ ] Mobile-responsive dashboard for in-vehicle tablet/phone mounts.
- [ ] Configurable EAR threshold and frame-count sensitivity via the UI.

---

## 7. Author & License

**Author:** Akash Joshi

**License:** This project is open-source and available under the [MIT License](LICENSE).

