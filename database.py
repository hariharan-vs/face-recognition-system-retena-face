import sqlite3
import numpy as np
import io

DB_FILE = 'attendance_system.db'

# Adapt numpy array to SQLite BLOB
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

# Convert SQLite BLOB back to numpy array
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    try:
        return np.load(out)
    except:
        # Fallback if it's just raw bytes and not a numpy save
        return np.frombuffer(text, dtype=np.float32)

# Register the adapters
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def get_connection():
    # Detect types allows us to automatically use convert_array on columns defined as 'array'
    return sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Students table
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            face_embedding array,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            date DATE,
            time TIME,
            period INTEGER,
            status TEXT,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    ''')
    
    # Try adding the period column in case the table already exists from older version
    try:
        c.execute('ALTER TABLE attendance ADD COLUMN period INTEGER DEFAULT 1')
    except sqlite3.OperationalError:
        pass # Column already exists
    
    conn.commit()
    conn.close()

def add_student(student_id, name, department, face_embedding):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO students (student_id, name, department, face_embedding)
            VALUES (?, ?, ?, ?)
        ''', (student_id, name, department, face_embedding))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Student ID already exists
        return False
    finally:
        conn.close()

def get_all_students():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT student_id, name, department, face_embedding FROM students')
    rows = c.fetchall()
    conn.close()
    
    students = []
    for row in rows:
        students.append({
            'student_id': row[0],
            'name': row[1],
            'department': row[2],
            'face_embedding': row[3]
        })
    return students

def delete_student(student_id):
    conn = get_connection()
    c = conn.cursor()
    # Delete attendance records first to maintain referential integrity
    c.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
    # Delete student
    c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
    conn.commit()
    conn.close()
    return True

def has_marked_period(student_id, date, period):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT attendance_id FROM attendance
        WHERE student_id = ? AND date = ? AND period = ?
    ''', (student_id, date, period))
    result = c.fetchone()
    conn.close()
    return result is not None

def mark_attendance(student_id, date, time, period, status, confidence):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO attendance (student_id, date, time, period, status, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (student_id, date, time, period, status, confidence))
    conn.commit()
    conn.close()

def get_daily_attendance(date):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT a.attendance_id, s.student_id, s.name, s.department, a.date, a.time, a.period, a.status, a.confidence
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE a.date = ?
        ORDER BY a.period ASC, a.time DESC
    ''', (date,))
    rows = c.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            'attendance_id': row[0],
            'student_id': row[1],
            'name': row[2],
            'department': row[3],
            'date': row[4],
            'time': row[5],
            'period': row[6],
            'status': row[7],
            'confidence': row[8]
        })
    return records

def get_all_attendance():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT a.attendance_id, s.student_id, s.name, s.department, a.date, a.time, a.period, a.status, a.confidence
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.date DESC, a.period ASC, a.time DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            'attendance_id': row[0],
            'student_id': row[1],
            'name': row[2],
            'department': row[3],
            'date': row[4],
            'time': row[5],
            'period': row[6],
            'status': row[7],
            'confidence': row[8]
        })
    return records
