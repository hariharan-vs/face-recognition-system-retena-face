import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time

from database import init_db, add_student, get_all_students, has_marked_period, mark_attendance, get_daily_attendance, get_all_attendance, delete_student
from face_utils import load_embedder, detect_and_align_faces, extract_embedding, match_face
from excel_utils import export_to_excel
import base64

# Streamlit config
st.set_page_config(page_title="Smart Attendance System", layout="wide")

# Inject Global CSS Micro-animations
st.markdown("""
<style>
/* Smooth fade-in up for main content groups */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Apply staggered fade-in to major Streamlit containers */
[data-testid="stVerticalBlock"] > div {
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) ease-in-out both;
}

[data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.1s; }
[data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.2s; }
[data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.3s; }

/* Input fields hover and focus professional effects */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    transition: all 0.3s ease;
    border-radius: 6px;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
    box-shadow: 0 4px 10px rgba(0, 240, 255, 0.15);
    border-color: rgba(0, 240, 255, 0.6);
}

/* Form container card shadow */
[data-testid="stForm"] {
    border-radius: 12px;
    transition: all 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stForm"]:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
}

/* Success/Error/Info message pop-in */
[data-testid="stNotification"] {
    animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.95); }
    100% { opacity: 1; transform: scale(1); }
}

/* Clean up header animations */
h1 {
    position: relative;
    display: inline-block;
    padding-bottom: 5px;
}
h1::after {
    content: '';
    position: absolute;
    width: 0;
    height: 3px;
    display: block;
    margin-top: 5px;
    right: 0;
    background: #00f0ff;
    transition: width .4s ease;
}
h1:hover::after {
    width: 100%;
    left: 0;
    background: #00f0ff;
}
</style>
""", unsafe_allow_html=True)

# Initialize DB on start
init_db()

# Load Model
embedder = load_embedder()

# Session State Tracking
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'app_loaded' not in st.session_state:
    st.session_state['app_loaded'] = False
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

def play_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true" style="display:none;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    except Exception:
        pass

if not st.session_state['app_loaded']:
    st.session_state['app_loaded'] = True
    splash_html = """
    <style>
    .scan-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 60vh;
    }
    .scan-box {
      position: relative;
      width: 120px;
      height: 120px;
      border: 4px solid #00f0ff;
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
      margin-bottom: 20px;
    }
    .scan-box::before {
      content: '';
      position: absolute;
      top: -100%;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(to bottom, transparent, rgba(0, 240, 255, 0.3), #00f0ff);
      animation: scan 1.5s cubic-bezier(0.53, 0.01, 0.46, 1) infinite;
    }
    @keyframes scan {
      0% { top: -100%; }
      50% { top: 100%; }
      100% { top: -100%; }
    }
    .face-emoji {
      font-size: 70px;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100%;
      margin-top: -5px;
    }
    .loading-text {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 24px;
      font-weight: 600;
      letter-spacing: 2px;
      color: #00f0ff;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.4; }
      50% { opacity: 1; }
    }
    </style>
    <div class="scan-container">
      <div class="scan-box">
        <div class="face-emoji">👤</div>
      </div>
      <div class="loading-text">INITIALIZING AI SYSTEM...</div>
    </div>
    """
    st.markdown(splash_html, unsafe_allow_html=True)
    time.sleep(2.5)  # Extended slightly to let the animation play thoroughly
    st.experimental_rerun()

if 'reg_embeddings' not in st.session_state:
    st.session_state['reg_embeddings'] = []

if 'reg_images' not in st.session_state:
    st.session_state['reg_images'] = []

if 'camera_key' not in st.session_state:
    st.session_state['camera_key'] = 0

def draw_bounding_box(img, bbox, name, confidence, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{name} ({confidence:.2f})" if name else "Unknown"
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

with st.sidebar:
    # Add custom CSS to animate navigation links
    st.markdown("""
    <style>
    /* Target the radio button container */
    [data-testid="stRadio"] > div {
        gap: 12px;
    }
    
    /* Target individual radio items */
    [data-testid="stRadio"] label {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: transparent;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border-left: 3px solid transparent;
        cursor: pointer;
    }
    
    /* Hover effects for radio items */
    [data-testid="stRadio"] label:hover {
        background-color: rgba(0, 240, 255, 0.08); /* Subtle neon blue tint */
        transform: translateX(5px);
        border-left: 3px solid #00f0ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Active state (selected item) */
    [data-testid="stRadio"] label[data-checked="true"] {
        background-color: rgba(0, 240, 255, 0.15);
        border-left: 4px solid #00f0ff;
        font-weight: bold;
    }
    
    /* Hide the actual radio circle entirely for a cleaner "button" look */
    [data-testid="stRadio"] div[role="radio"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎛️ Navigation Menu")
    
    pages = ["📸 Live Attendance"]
    if st.session_state['admin_logged_in']:
        pages.extend(["📈 Admin Statistics", "⚙️ System Dashboard", "👤 Register Student", "🚪 Logout"])
    else:
        pages.extend(["🛡️ Admin Auth"])
    
    selection = st.radio("Menu", pages, label_visibility="collapsed")

if selection == "🛡️ Admin Auth":
    st.title("🛡️ Secure Administrator Portal")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("### Secure Access")
            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", type="password", placeholder="Enter admin password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if username == "admin" and password == "admin":
                    st.session_state['admin_logged_in'] = True
                    st.success("Successfully logged in!")
                    time.sleep(0.5)
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

elif selection == "🚪 Logout":
    st.session_state['admin_logged_in'] = False
    st.success("Logged out successfully.")
    st.experimental_rerun()

elif selection == "👤 Register Student":
    st.title("👤 Biometric Registration")
    st.write("Capture multiple face samples to ensure high accuracy.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st_student_id = st.text_input("Student ID")
        st_name = st.text_input("Full Name")
        st_dept = st.text_input("Department")
        
        img_file_buffer = st.camera_input("Take a picture", key=f"cam_{st.session_state['camera_key']}")
        
        if img_file_buffer is not None:
            # Convert to cv2 image
            bytes_data = img_file_buffer.getvalue()
            # Must strictly cast to uint8 and explicitly flag memory as contiguous before giving it to deep learning detectors
            np_arr = np.frombuffer(bytes_data, np.uint8)
            cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Ensure proper array structure for deep learning model
            if cv2_img is not None:
                cv2_img = np.ascontiguousarray(cv2_img)
            with st.spinner("Detecting face..."):
                faces = detect_and_align_faces(cv2_img)
                if len(faces) == 0:
                    st.error("No face detected. Please try again.")
                elif len(faces) > 1:
                    st.warning("Multiple faces detected. Please make sure only you are in the frame.")
                else:
                    face = faces[0]
                    emb = extract_embedding(face['face_img'], embedder)
                    if emb is not None:
                        st.session_state['reg_embeddings'].append(emb)
                        # Save a copy of the actual photo, converted to RGB so colors save correctly
                        img_rgb_to_save = cv2.cvtColor(cv2_img.copy(), cv2.COLOR_BGR2RGB)
                        st.session_state['reg_images'].append(img_rgb_to_save) 
                        st.success(f"Sample {len(st.session_state['reg_embeddings'])}/3 captured! Please prepare for the next sample.")
                        # Increment camera key to force the widget to remount and clear the picture
                        st.session_state['camera_key'] += 1
                        time.sleep(1.5)
                        st.experimental_rerun()
                        
    with col2:
        st.subheader("Registration Status")
        st.write(f"Samples collected: {len(st.session_state['reg_embeddings'])} / 3")
        
        if len(st.session_state['reg_embeddings']) >= 3:
            st.success("Sufficient samples collected. You can now register.")
            if st.button("Register Student", type="primary"):
                if not st_student_id or not st_name:
                    st.error("Student ID and Name are required.")
                else:
                    # Average the embeddings
                    avg_emb = np.mean(st.session_state['reg_embeddings'], axis=0)
                    # Normalize again
                    avg_emb = avg_emb / np.linalg.norm(avg_emb)
                    
                    if add_student(st_student_id, st_name, st_dept, avg_emb):
                        st.success("Student registered successfully!")
                        
                        # Save the raw picture samples to a folder
                        if not os.path.exists("dataset"):
                            os.makedirs("dataset")
                            
                        # Save each image
                        for i, img in enumerate(st.session_state['reg_images']):
                            filename = f"dataset/{st_student_id}_{i+1}.jpg"
                            cv2.imwrite(filename, img)
                            
                        st.info(f"💾 Photos saved to 'dataset/' folder on the computer.")
                        
                        # Clear states
                        st.session_state['reg_embeddings'] = []
                        st.session_state['reg_images'] = []
                    else:
                        st.error("Student ID already exists.")
                        
        if len(st.session_state['reg_embeddings']) > 0:
            if st.button("Clear Samples"):
                st.session_state['reg_embeddings'] = []
                st.session_state['reg_images'] = []
                st.experimental_rerun()

elif selection == "📸 Live Attendance":
    st.title("📸 Real-Time Face Verification")
    st.markdown("Monitor real-time face verification and automated attendance marking powered by AI.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Settings")
        c1, c2 = st.columns(2)
        with c1:
            attendance_date = st.date_input("Attendance Date", datetime.today())
        with c2:
            attendance_period = st.selectbox("Select Period", options=[1, 2, 3, 4, 5, 6, 7])
            
        c_cam1, c_cam2 = st.columns(2)
        with c_cam1:
            # Custom CSS specifically targeting the camera control buttons to make them interesting
            st.markdown("""
            <style>
            /* The primary button (Start Camera) */
            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
                border: none;
                color: #1E1E1E;
                font-weight: 800;
                letter-spacing: 1px;
                text-transform: uppercase;
                box-shadow: 0 0 15px rgba(0, 201, 255, 0.4);
                transition: all 0.3s ease;
                animation: pulse-glow 2s infinite;
            }
            
            div[data-testid="stButton"] button[kind="primary"]:hover {
                transform: scale(1.05);
                box-shadow: 0 0 25px rgba(0, 201, 255, 0.7);
                border: none;
            }
            
            @keyframes pulse-glow {
                0% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.4); }
                50% { box-shadow: 0 0 25px rgba(0, 201, 255, 0.8); }
                100% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.4); }
            }
            
            /* The secondary button (Stop Camera) */
            div[data-testid="stButton"] button[kind="secondary"] {
                background-color: transparent;
                border: 2px solid #FF4B4B;
                color: #FF4B4B;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            div[data-testid="stButton"] button[kind="secondary"]:hover {
                background-color: rgba(255, 75, 75, 0.1);
                box-shadow: 0 0 15px rgba(255, 75, 75, 0.4);
                transform: scale(1.05);
                color: #FF4B4B;
                border: 2px solid #FF4B4B;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if not st.session_state['camera_active']:
                if st.button("▶️ START ATTENDANCE CAPTURE", type="primary", use_container_width=True):
                    st.session_state['camera_active'] = True
                    st.experimental_rerun()
            else:
                if st.button("⏹️ END ATTENDANCE CAPTURE", type="secondary", use_container_width=True):
                    st.session_state['camera_active'] = False
                    st.experimental_rerun()
                    
        FRAME_WINDOW = st.image([])
        
    with col2:
        st.subheader("📋 Recent Entries")
        # Scrollable container for logs (height handled via CSS/inner container)
        log_container = st.container()
    
    audio_placeholder = st.empty()
    
    if st.session_state['camera_active']:
        students = get_all_students()
        
        # Optimize camera startup and frame size for faster inference
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Fast startup on Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Keep track of recent updates to not spam the db
        recent_attendances = set()
        recent_logs = []
        
        # Display initial empty state
        with log_container:
            log_placeholder = st.empty()
            log_placeholder.info("Waiting for users...")
        
        while st.session_state['camera_active']:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Process every frame for live tracking
            faces = detect_and_align_faces(frame)
            for face in faces:
                bbox = face['bbox']
                emb = extract_embedding(face['face_img'], embedder)
                
                if emb is not None:
                    match, conf = match_face(emb, students, threshold=0.85)
                    
                    if match:
                        student_id = match['student_id']
                        name = match['name']
                        
                        today_str = attendance_date.strftime("%Y-%m-%d")
                        current_time = datetime.now().strftime("%H:%M:%S")
                        
                        # Check if already marked for THIS period
                        if has_marked_period(student_id, today_str, attendance_period):
                            color = (0, 165, 255) # Orange for already marked
                            name_display = f"{name} (Already Marked P{attendance_period})"
                            
                            if (student_id, attendance_period) not in recent_attendances:
                                recent_attendances.add((student_id, attendance_period))
                                with audio_placeholder:
                                    play_audio("already.wav")
                                    
                        else:
                            color = (0, 255, 0) # Green for newly marked
                            name_display = name
                            
                            # Mark present
                            mark_attendance(student_id, today_str, current_time, attendance_period, "Present", float(conf))
                            recent_attendances.add((student_id, attendance_period))
                            
                            # Log and Audio
                            with audio_placeholder:
                                play_audio("success.wav")
                                
                            # UI Notification changes
                            msg = {"text": f"**{name}** marked Present at {current_time}", "period": attendance_period}
                            recent_logs.insert(0, msg) # Prepend
                            
                            st.balloons()
                            st.toast(f"✅ Period {attendance_period} Logged: {name}", icon="✅")
                            
                            # Update logs UI showing ONLY entries for the currently selected period
                            with log_placeholder.container():
                                for log in recent_logs:
                                    if log['period'] == attendance_period:
                                        st.success(log['text'], icon="✅")
                                
                    else:
                        name_display = "Unknown - Access Denied"
                        conf = face['score']
                        color = (0, 0, 255)
                        
                        if "unknown_recent" not in recent_attendances:
                            with audio_placeholder:
                                play_audio("error.wav")
                            recent_attendances.add("unknown_recent")
                        
                    frame = draw_bounding_box(frame, bbox, name_display, conf, color)
                    
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()

elif selection == "📈 Admin Statistics":
    st.title("📈 Attendance Analytics Hub")
    st.write("Overview of system-wide attendance patterns and trends.")
    
    stats_data = get_all_attendance()
    if not stats_data:
        st.info("No attendance data available yet to generate statistics.")
    else:
        df = pd.DataFrame(stats_data)
        
        # High-level Metrics
        total_logs = len(df)
        unique_students = df['student_id'].nunique()
        total_active_days = df['date'].nunique()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Attendances Logged", total_logs)
        c2.metric("Unique Students Attended", unique_students)
        c3.metric("Active Tracking Days", total_active_days)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Attendance by Date")
            date_counts = df['date'].value_counts().sort_index()
            st.bar_chart(date_counts)
            
        with col_chart2:
            st.subheader("Attendance by Period")
            period_counts = df['period'].value_counts().sort_index()
            st.bar_chart(period_counts)
            
        st.markdown("---")
        st.subheader("Student-wise Summary")
        student_counts = df['name'].value_counts().reset_index()
        student_counts.columns = ['Student Name', 'Total Classes Attended']
        st.dataframe(student_counts, use_container_width=True)

elif selection == "⚙️ System Dashboard":
    st.title("⚙️ Database Management")
    st.write("View and export daily records.")
    
    date_filter = st.date_input("Select Date", datetime.today())
    
    records = get_daily_attendance(date_filter.strftime("%Y-%m-%d"))
    
    if len(records) > 0:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Export to Excel"):
            success = export_to_excel(records, date_filter.strftime("%Y-%m-%d"))
            if success:
                st.success(f"Successfully exported to Attendance_Log.xlsx!")
            else:
                st.error("Failed to export to Excel.")
    else:
        st.info("No attendance records found for this date.")

    st.markdown("---")
    st.subheader("Manage Students")
    st.write("Remove a student and all their associated attendance records.")
    
    # We can fetch the list of students to make a friendly dropdown or just use text input
    all_students_db = get_all_students()
    if len(all_students_db) > 0:
        student_options = {s['student_id']: f"{s['student_id']} - {s['name']}" for s in all_students_db}
        selected_to_delete = st.selectbox("Select Student to Delete", options=list(student_options.keys()), format_func=lambda x: student_options[x])
        
        if st.button("Delete Student", type="primary"):
            if delete_student(selected_to_delete):
                st.success(f"Student {selected_to_delete} successfully removed.")
                time.sleep(1) # short delay before reload
                st.experimental_rerun()
            else:
                st.error("Error removing student.")
    else:
        st.info("No students registered yet.")
