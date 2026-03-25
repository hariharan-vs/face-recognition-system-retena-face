# Face Recognition-Based Smart Attendance System

An automated face recognition attendance system built using **Streamlit**, **RetinaFace**, **FaceNet**, **InsightFace logic** and **SQLite**.

## Features

- **Face Detection**: Uses RetinaFace for high-accuracy face detection.
- **Face Embedding**: Extracts 128-D embeddings using FaceNet.
- **Face Matching**: High-precision cosine similarity matching mimicking InsightFace/ArcFace logic.
- **Database Architecture**: SQLite database stores secure embeddings (BLOBs) and attendance logs.
- **Excel Export**: Generates and appends dynamic Excel sheets per day.
- **Liveness & Anti-proxy**: Ensures continuous face capture logic.

## Project Structure

- `app.py`: Main Streamlit Application with GUI (Admin, Student Registration, Live Tracking).
- `face_utils.py`: Pipeline for face detection, alignment, embedding extraction, and matching.
- `database.py`: SQLite schema mappings and CRUD operations for students & attendance.
- `excel_utils.py`: Excel workbook appending routines using Pandas/OpenPyXL.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

### Troubleshooting Face Detection

If the system is not detecting faces correctly:

1. Make sure the `retina-face` package is installed (`pip install retina-face`).
2. Lighting/angle can affect detection; ensure the face is well-lit and centered.
3. Use the provided helper script to verify detection on sample images or your webcam:

```bash
python detect_test.py --image path/to/photo.jpg
# or
python detect_test.py  # to open the webcam
```

The script will draw bounding boxes and log raw detector output, which can help identify if the problem lies with the detector or later steps.

> **Optional fallback:** You can install `face_recognition` (which uses `dlib`) as
> a secondary detector. On Windows this requires Visual C++ build tools to compile
> `dlib`. If the wheel build fails with a message about Visual Studio, install
> the [Build Tools for Visual Studio](https://aka.ms/vs/17/release/vs_buildtools.exe)
> or skip this step. Once you have the prerequisites, run:
>
> ```bash
> pip install face_recognition
> ```
>
> The code automatically checks for its availability and works fine without it.

## Setup Instructions

1. **Create Virtual Environment** (Recommended):

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Application**:

```bash
streamlit run app.py
```

## System Usage

1. First, navigate to **Admin Login** in the sidebar. (Default credentials: `admin` / `admin`).
2. Go to **Student Registration**. Enter student details, look into the camera, and snap a picture. The system captures the embedding and saves it to the database.
3. Switch to **Live Attendance**. Check the "Toggle Camera Feed". The system will recognize the faces in front of the camera in real-time. If it matches a registered student with `>= 0.6` confidence, it marks them Present for the day. Duplicate marking on the same day is prevented.
4. Navigate to the **Admin Dashboard** to view all attendance entries for the selected date and export them to `Attendance_Log.xlsx`.
