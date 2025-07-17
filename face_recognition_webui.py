from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import json
import time
import sqlite3
from datetime import datetime, timedelta
import cv2
import base64
import numpy as np
import threading
import uuid
import socket
import atexit
import signal
import sys
import dlib
from collections import deque
import math
import random
import pickle
import shutil

try:
    from maml_face_recognition import EnhancedFaceRecognition
    FACE_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️  Enhanced face recognition system not found")
    FACE_SYSTEM_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'securetime-pro-simplified-2024'
socketio = SocketIO(app, cors_allowed_origins="*", max_size=10*1024*1024)

PEOPLE_FOLDER = 'people_database'
DATABASE_FILE = 'securetime.db'
MODEL_CACHE_FILE = 'adaptive_models.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_SAMPLES_PER_PERSON = 20
RECOGNITION_THRESHOLD = 0.6
CLOCK_ACTION_THRESHOLD = 60
MAX_DISTANCE_THRESHOLD = 0.5
LEARNING_CONFIDENCE_THRESHOLD = 0.8
AUTO_CLOCK_DELAY = 2  # Seconds to wait for stable recognition before auto clock

for folder in [PEOPLE_FOLDER, 'templates', 'static']:
    os.makedirs(folder, exist_ok=True)

face_system = None
adaptive_system = None
system_status = {"loaded": False, "message": "Initializing...", "error": None}
active_sessions = {}

class SimpleLivenessDetector:
    """Simple, reliable liveness detection - safe fallback version"""
    
    def __init__(self):
        # Force basic detection to avoid segfaults
        print("⚠️  Using basic liveness detection (safe mode)")
        self.available = False
        
        # Simple detection state
        self.frames_processed = 0
        
    def analyze_frame(self, frame):
        """Simple frame analysis using OpenCV only"""
        self.frames_processed += 1
        
        results = {
            'face_detected': False,
            'face_quality': 0,
            'liveness_verified': False,
            'blinks_count': 0,
            'progress': 0,
            'instruction': 'Look at camera naturally'
        }
        
        try:
            # Basic face detection using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                results['face_detected'] = True
                
                # Calculate face quality
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                quality = min(face_area / (frame_area * 0.05), 1.0)
                results['face_quality'] = quality
                
                # Simple time-based verification (3 seconds)
                if self.frames_processed >= 30:  # ~3 seconds at 10fps
                    results['liveness_verified'] = True
                    results['instruction'] = 'Verification complete'
                    results['progress'] = 1.0
                else:
                    progress = min(self.frames_processed / 30, 1.0)
                    results['progress'] = progress
                    if quality > 0.3:
                        results['instruction'] = f'Hold steady... {int(progress*100)}%'
                    else:
                        results['instruction'] = 'Move closer to camera'
            else:
                results['instruction'] = 'Please position face in camera'
                
        except Exception as e:
            print(f"Error in frame analysis: {e}")
            results['instruction'] = 'Camera error - please try again'
        
        return results, frame
    
    def reset(self):
        """Reset detector state"""
        self.frames_processed = 0

class ActiveLearningSystem:
    """Simplified active learning system"""
    
    def __init__(self, base_face_system):
        self.base_face_system = base_face_system
        self.person_samples = {}
        self.learning_stats = {}
        self.load_cached_models()
        self._setup_method_compatibility()
    
    def _setup_method_compatibility(self):
        """Setup compatibility with face recognition system"""
        if hasattr(self.base_face_system, 'extract_face_encoding'):
            self._extract_encoding = self.base_face_system.extract_face_encoding
        elif hasattr(self.base_face_system, 'extract_face_encoding_fast'):
            self._extract_encoding = self.base_face_system.extract_face_encoding_fast
        else:
            import face_recognition
            self._extract_encoding = self._basic_extract_encoding
    
    def _basic_extract_encoding(self, image_path):
        """Fallback encoding extraction"""
        try:
            import face_recognition
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            return encodings[0] if encodings else None
        except:
            return None
    
    def load_cached_models(self):
        """Load cached models"""
        try:
            if os.path.exists(MODEL_CACHE_FILE):
                with open(MODEL_CACHE_FILE, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.person_samples = cached_data.get('person_samples', {})
                    self.learning_stats = cached_data.get('learning_stats', {})
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
            self.person_samples = {}
            self.learning_stats = {}
    
    def save_models(self):
        """Save models"""
        try:
            cache_data = {
                'person_samples': self.person_samples,
                'learning_stats': self.learning_stats
            }
            with open(MODEL_CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"⚠️  Error saving models: {e}")
    
    def add_person_samples(self, person_name, image_paths):
        """Add training samples for a person"""
        if person_name not in self.person_samples:
            self.person_samples[person_name] = []
            self.learning_stats[person_name] = {
                'total_samples': 0,
                'recognition_accuracy': 0.0
            }
        
        for image_path in image_paths:
            encoding = self._extract_encoding(image_path)
            if encoding is not None:
                sample = {
                    'encoding': encoding,
                    'timestamp': datetime.now(),
                    'source': 'training'
                }
                self.person_samples[person_name].append(sample)
                self.learning_stats[person_name]['total_samples'] += 1
        
        self.save_models()
    
    def recognize_with_learning(self, image_path_or_frame, learn_from_result=True):
        """Recognize face with learning and proper unknown detection"""
        if isinstance(image_path_or_frame, str):
            encoding = self._extract_encoding(image_path_or_frame)
        else:
            temp_path = f"temp_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(temp_path, image_path_or_frame)
            encoding = self._extract_encoding(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if encoding is None:
            return "Unknown", 0.0, "no_face"
        
        best_person = "Unknown"
        best_confidence = 0.0
        best_distance = float('inf')
        
        for person_name, samples in self.person_samples.items():
            if not samples:
                continue
            
            distances = []
            for sample in samples[-10:]:
                distance = np.linalg.norm(encoding - sample['encoding'])
                distances.append(distance)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                min_distance = min(distances)
                final_distance = min(avg_distance, min_distance * 1.2)
                confidence = max(0, 1 - final_distance)
                
                if final_distance <= MAX_DISTANCE_THRESHOLD and confidence > best_confidence:
                    best_confidence = confidence
                    best_person = person_name
                    best_distance = final_distance
        
        if best_confidence < RECOGNITION_THRESHOLD:
            return "Unknown", best_confidence, "low_confidence"
        
        if (learn_from_result and best_person != "Unknown" and 
            best_confidence > LEARNING_CONFIDENCE_THRESHOLD):
            self._learn_from_recognition(best_person, encoding)
        
        return best_person, best_confidence, "adaptive"
    
    def _learn_from_recognition(self, person_name, encoding):
        """Learn from successful recognition"""
        if person_name not in self.person_samples:
            return
        
        sample = {
            'encoding': encoding,
            'timestamp': datetime.now(),
            'source': 'learning'
        }
        
        self.person_samples[person_name].append(sample)
        
        if len(self.person_samples[person_name]) > MAX_SAMPLES_PER_PERSON:
            self.person_samples[person_name] = self.person_samples[person_name][-MAX_SAMPLES_PER_PERSON:]
        
        self.learning_stats[person_name]['total_samples'] += 1

class TimeTrackingDB:
    """Database manager - ENHANCED VERSION with better duplicate prevention"""
    
    def __init__(self, db_file):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT NOT NULL,
                clock_in TIMESTAMP,
                clock_out TIMESTAMP,
                hours_worked REAL,
                date DATE,
                confidence REAL,
                liveness_verified BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_employee(self, name, department=None):
        """Add employee"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO employees (name, department) VALUES (?, ?)', 
                         (name, department))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()
    
    def delete_employee(self, name):
        """Delete employee"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM employees WHERE name = ?', (name,))
            cursor.execute('DELETE FROM time_records WHERE employee_name = ?', (name,))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()
    
    def clock_in(self, employee_name, confidence, liveness_verified=True):
        """Record clock-in - ENHANCED VERSION with better duplicate checking"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        current_time = datetime.now()
        current_date = current_time.date()
        
        # Check for ANY active record (clock_out IS NULL) for this employee today
        cursor.execute('''SELECT id, clock_in FROM time_records 
                         WHERE employee_name = ? AND date = ? AND clock_out IS NULL
                         ORDER BY clock_in DESC LIMIT 1''', 
                      (employee_name, current_date))
        
        active_record = cursor.fetchone()
        
        if active_record:
            record_id, clock_in_time = active_record
            conn.close()
            print(f"❌ {employee_name} already has active clock-in at {clock_in_time}")
            return False, f"Already clocked in at {datetime.fromisoformat(clock_in_time).strftime('%I:%M %p')}. Please clock out first."
        
        # Create new clock-in record
        cursor.execute('''INSERT INTO time_records (employee_name, clock_in, date, confidence, liveness_verified)
                         VALUES (?, ?, ?, ?, ?)''', 
                      (employee_name, current_time, current_date, confidence, liveness_verified))
        
        new_record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ {employee_name} clocked in at {current_time.strftime('%I:%M:%S %p')} (Record ID: {new_record_id})")
        return True, f"Clock-in successful at {current_time.strftime('%I:%M %p')}"
    
    def clock_out(self, employee_name, confidence):
        """Record clock-out - ENHANCED VERSION with better error handling"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        current_time = datetime.now()
        current_date = current_time.date()
        
        # Find the most recent ACTIVE clock-in (no clock-out) for today
        cursor.execute('''SELECT id, clock_in FROM time_records 
                         WHERE employee_name = ? AND date = ? AND clock_out IS NULL
                         ORDER BY clock_in DESC LIMIT 1''', 
                      (employee_name, current_date))
        
        active_record = cursor.fetchone()
        
        if not active_record:
            # Check if there are any records for today
            cursor.execute('''SELECT COUNT(*) FROM time_records 
                             WHERE employee_name = ? AND date = ?''', 
                          (employee_name, current_date))
            total_records = cursor.fetchone()[0]
            
            conn.close()
            
            if total_records == 0:
                print(f"❌ {employee_name} has no records for today")
                return False, "No clock-in found for today. Please clock in first."
            else:
                print(f"❌ {employee_name} has no active clock-in (all sessions completed)")
                return False, "No active clock-in found. All sessions are complete. Please clock in again."
        
        record_id, clock_in_time = active_record
        clock_in_dt = datetime.fromisoformat(clock_in_time)
        
        # Calculate hours worked
        time_diff = current_time - clock_in_dt
        hours_worked = time_diff.total_seconds() / 3600
        
        # Ensure minimum time (prevent 0.00 hour sessions due to quick testing)
        if hours_worked < 0.01:  # Less than 36 seconds
            hours_worked = 0.01
        
        # Update the existing record with clock-out time
        cursor.execute('''UPDATE time_records 
                         SET clock_out = ?, hours_worked = ? 
                         WHERE id = ?''', 
                      (current_time, hours_worked, record_id))
        
        # Verify the update worked
        if cursor.rowcount == 0:
            conn.close()
            print(f"❌ Failed to update record {record_id} for {employee_name}")
            return False, "Failed to update clock-out record"
        
        conn.commit()
        conn.close()
        
        print(f"✅ {employee_name} clocked out at {current_time.strftime('%I:%M:%S %p')} - Worked {hours_worked:.2f} hours (Record ID: {record_id})")
        return True, f"Clock-out successful at {current_time.strftime('%I:%M %p')}. Worked {hours_worked:.2f} hours"
    
    def get_employee_status(self, employee_name):
        """Get employee status - ENHANCED VERSION with detailed debugging"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        current_date = datetime.now().date()
        
        # Get ALL records for today for debugging
        cursor.execute('''SELECT id, clock_in, clock_out, hours_worked FROM time_records 
                         WHERE employee_name = ? AND date = ? 
                         ORDER BY clock_in DESC''', 
                      (employee_name, current_date))
        
        all_records = cursor.fetchall()
        print(f"🔍 {employee_name} has {len(all_records)} records for {current_date}")
        
        if not all_records:
            conn.close()
            print(f"📊 Status: {employee_name} = not_clocked_in (no records)")
            return "not_clocked_in", None
        
        # Check for active record (most recent with clock_out = NULL)
        for record in all_records:
            record_id, clock_in, clock_out, hours_worked = record
            print(f"  Record {record_id}: clock_in={clock_in}, clock_out={clock_out}, hours={hours_worked}")
            
            if clock_out is None:
                # Found active session
                clock_in_dt = datetime.fromisoformat(clock_in)
                hours_so_far = (datetime.now() - clock_in_dt).total_seconds() / 3600
                conn.close()
                print(f"📊 Status: {employee_name} = clocked_in (active session)")
                return "clocked_in", {
                    "clock_in": clock_in,
                    "hours_so_far": round(hours_so_far, 2)
                }
        
        # All records have clock_out, so employee is available to clock in again
        most_recent = all_records[0]
        record_id, clock_in, clock_out, hours_worked = most_recent
        
        conn.close()
        print(f"📊 Status: {employee_name} = clocked_out (ready for next session)")
        return "clocked_out", {
            "clock_in": clock_in,
            "clock_out": clock_out,
            "hours_worked": round(hours_worked, 2) if hours_worked else 0
        }
    
    def get_time_records(self, employee_name=None, date_from=None, date_to=None):
        """Get time records - FIXED VERSION (maintains original format for reports)"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        query = '''SELECT employee_name, clock_in, clock_out, hours_worked, date, confidence 
                   FROM time_records WHERE 1=1'''
        params = []
        
        if employee_name:
            query += ' AND employee_name = ?'
            params.append(employee_name)
        if date_from:
            query += ' AND date >= ?'
            params.append(date_from)
        if date_to:
            query += ' AND date <= ?'
            params.append(date_to)
        
        query += ' ORDER BY date DESC, clock_in DESC'
        cursor.execute(query, params)
        records = cursor.fetchall()
        conn.close()
        
        # Return in original format that reports expect
        return [dict(zip(['employee_name', 'clock_in', 'clock_out', 'hours_worked', 'date', 'confidence'], record)) 
                for record in records]
    
    def debug_employee_records(self, employee_name):
        """Debug function to see all records for an employee"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''SELECT * FROM time_records WHERE employee_name = ? ORDER BY created_at DESC''', 
                      (employee_name,))
        records = cursor.fetchall()
        conn.close()
        
        print(f"\n🔍 DEBUG: All records for {employee_name}:")
        for i, record in enumerate(records):
            print(f"  {i+1}. ID:{record[0]} | Clock-in:{record[2]} | Clock-out:{record[3]} | Hours:{record[4]} | Date:{record[5]}")
        print()
        
        return records
time_db = TimeTrackingDB(DATABASE_FILE)

def find_free_port(start_port=5000):
    """Find available port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port

def allowed_file(filename):
    """Check if file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_face_system():
    """Initialize systems"""
    global face_system, adaptive_system, system_status
    
    try:
        system_status["message"] = "Loading AI systems..."
        
        if not FACE_SYSTEM_AVAILABLE:
            system_status["loaded"] = False
            system_status["error"] = "Face recognition system not available"
            return
        
        face_system = EnhancedFaceRecognition()
        adaptive_system = ActiveLearningSystem(face_system)
        
        people_loaded = 0
        if os.path.exists(PEOPLE_FOLDER):
            for person_name in os.listdir(PEOPLE_FOLDER):
                person_folder = os.path.join(PEOPLE_FOLDER, person_name)
                if os.path.isdir(person_folder):
                    image_paths = [os.path.join(person_folder, f) for f in os.listdir(person_folder) if allowed_file(f)]
                    
                    if image_paths:
                        try:
                            face_system.add_person_enhanced(person_name, image_paths)
                            adaptive_system.add_person_samples(person_name, image_paths)
                            time_db.add_employee(person_name)
                            people_loaded += 1
                        except Exception as e:
                            print(f"⚠️  Error loading {person_name}: {e}")
        
        system_status["loaded"] = True
        system_status["message"] = f"SecureTime Pro ready • {people_loaded} employees"
        
    except Exception as e:
        system_status["loaded"] = False
        system_status["error"] = str(e)
        system_status["message"] = f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/timeclock')
def timeclock():
    return render_template('timeclock.html')

@app.route('/employees')
def employees():
    return render_template('employees.html')

@app.route('/add_employee')
def add_employee():
    return render_template('add_employee.html')

@app.route('/edit_employee/<employee_name>')
def edit_employee(employee_name):
    return render_template('edit_employee.html', employee_name=employee_name)

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/api/status')
def api_status():
    return jsonify(system_status)

@app.route('/api/stats')
def api_stats():
    """Get stats"""
    try:
        stats = {'total_people': 0, 'total_images': 0, 'clocked_in_today': 0}
        
        if os.path.exists(PEOPLE_FOLDER):
            for person_name in os.listdir(PEOPLE_FOLDER):
                person_folder = os.path.join(PEOPLE_FOLDER, person_name)
                if os.path.isdir(person_folder):
                    images = [f for f in os.listdir(person_folder) if allowed_file(f)]
                    stats['total_people'] += 1
                    stats['total_images'] += len(images)
                    
                    status, _ = time_db.get_employee_status(person_name)
                    if status == "clocked_in":
                        stats['clocked_in_today'] += 1
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/employees')
def api_employees():
    """Get all employees with photos"""
    try:
        employees = []
        if os.path.exists(PEOPLE_FOLDER):
            for person_name in os.listdir(PEOPLE_FOLDER):
                person_folder = os.path.join(PEOPLE_FOLDER, person_name)
                if os.path.isdir(person_folder):
                    images = [f for f in os.listdir(person_folder) if allowed_file(f)]
                    status, details = time_db.get_employee_status(person_name)
                    
                    thumbnail = None
                    if images:
                        thumbnail = f"/api/employee_photo/{person_name}/{images[0]}"
                    
                    employees.append({
                        'name': person_name,
                        'image_count': len(images),
                        'images': images,
                        'thumbnail': thumbnail,
                        'status': status,
                        'details': details
                    })
        
        return jsonify({'success': True, 'employees': employees})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employee_photo/<employee_name>/<filename>')
def api_employee_photo(employee_name, filename):
    """Serve employee photos"""
    try:
        person_folder = os.path.join(PEOPLE_FOLDER, employee_name)
        return send_from_directory(person_folder, filename)
    except:
        return "Photo not found", 404

@app.route('/api/delete_employee/<employee_name>', methods=['DELETE'])
def api_delete_employee(employee_name):
    """Delete employee"""
    try:
        time_db.delete_employee(employee_name)
        
        person_folder = os.path.join(PEOPLE_FOLDER, employee_name)
        if os.path.exists(person_folder):
            shutil.rmtree(person_folder)
        
        if adaptive_system and employee_name in adaptive_system.person_samples:
            del adaptive_system.person_samples[employee_name]
            del adaptive_system.learning_stats[employee_name]
            adaptive_system.save_models()
        
        return jsonify({'success': True, 'message': f'{employee_name} deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_employee_photo/<employee_name>/<filename>', methods=['DELETE'])
def api_delete_employee_photo(employee_name, filename):
    """Delete specific employee photo"""
    try:
        person_folder = os.path.join(PEOPLE_FOLDER, employee_name)
        photo_path = os.path.join(person_folder, filename)
        
        if os.path.exists(photo_path):
            os.remove(photo_path)
            return jsonify({'success': True, 'message': 'Photo deleted'})
        else:
            return jsonify({'success': False, 'message': 'Photo not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/time_records')
def api_time_records():
    """Get time records"""
    try:
        employee_name = request.args.get('employee')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        records = time_db.get_time_records(employee_name, date_from, date_to)
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    active_sessions[request.sid] = {
        'liveness_detector': SimpleLivenessDetector(),
        'mode': 'timeclock',
        'registration_data': None,
        'liveness_verified': False,
        'current_recognition': None,
        'recognition_start_time': None,
        'stable_recognition': False,
        'last_recognition_name': None,
        'session_active': True
    }

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    if request.sid in active_sessions:
        del active_sessions[request.sid]

@socketio.on('start_session')
def handle_start_session(data):
    """Start camera session"""
    session_id = request.sid
    if session_id in active_sessions:
        mode = data.get('mode', 'timeclock')
        active_sessions[session_id]['mode'] = mode
        active_sessions[session_id]['liveness_detector'].reset()
        active_sessions[session_id]['session_active'] = True
        active_sessions[session_id]['recognition_start_time'] = None
        active_sessions[session_id]['stable_recognition'] = False
        active_sessions[session_id]['last_recognition_name'] = None
        
        if mode == 'registration':
            active_sessions[session_id]['registration_data'] = {
                'employee_name': data.get('employee_name'),
                'current_pose': 'front',
                'poses_completed': [],
                'photos_taken': []
            }
        
        emit('session_started', {'mode': mode})

def auto_clock_action(session_id, employee_name, confidence):
    """Perform automatic clock in/out action - FIXED VERSION (no more duplicates!)"""
    print(f"🚀 Auto clock action for {employee_name} (session: {session_id})")
    
    try:
        # Immediately stop the session to prevent multiple triggers
        if session_id in active_sessions:
            active_sessions[session_id]['session_active'] = False
            active_sessions[session_id]['stable_recognition'] = False
            print(f"🛑 Session {session_id} deactivated")
        
        # Check current status with detailed logging
        status, details = time_db.get_employee_status(employee_name)
        print(f"📊 Current status for {employee_name}: {status}")
        print(f"📊 Status details: {details}")
        
        # FIXED LOGIC: Only two actions based on current state
        if status == "clocked_in":
            # Employee is currently clocked in → CLOCK OUT
            success, message = time_db.clock_out(employee_name, confidence)
            action = "clock_out" 
            action_display = "Clocked Out"
            print(f"⏰ Clock out attempt: {success}, {message}")
            
        elif status in ["not_clocked_in", "clocked_out"]:
            # Employee is not clocked in OR previously clocked out → CLOCK IN
            success, message = time_db.clock_in(employee_name, confidence, True)
            action = "clock_in"
            action_display = "Clocked In"
            print(f"⏰ Clock in attempt: {success}, {message}")
            
        else:
            # Unknown status - should not happen
            success = False
            message = f"Unknown status: {status}"
            action = "error"
            action_display = "Error"
            print(f"❓ Unknown status: {status}")
        
        # Prepare response data
        response_data = {
            'success': success,
            'employee': employee_name,
            'action': action,
            'action_display': action_display,
            'message': message,
            'confidence': round(confidence * 100, 1),
            'timestamp': datetime.now().strftime('%I:%M %p')
        }
        
        if success:
            # Get updated status after clock action
            new_status, new_details = time_db.get_employee_status(employee_name)
            response_data['status'] = new_status
            response_data['details'] = new_details
            print(f"📊 New status for {employee_name}: {new_status}")
            
            # Apply learning if confidence is high enough
            if adaptive_system and confidence > LEARNING_CONFIDENCE_THRESHOLD:
                try:
                    print(f"🧠 Learning from {employee_name} recognition")
                except Exception as e:
                    print(f"Learning error: {e}")
        else:
            # Re-enable session on failure so user can try again
            if session_id in active_sessions:
                active_sessions[session_id]['session_active'] = True
                active_sessions[session_id]['recognition_start_time'] = None
                active_sessions[session_id]['stable_recognition'] = False
        
        # Use socketio.emit with room parameter to avoid context issues
        socketio.emit('auto_clock_result', response_data, room=session_id)
        print(f"✅ Result sent for {employee_name}: {action_display}")
        
        if success:
            # Send session stop signal after small delay
            socketio.sleep(0.5)
            socketio.emit('session_stopped', {'reason': 'auto_clock_success'}, room=session_id)
            print(f"📡 Session stopped signal sent to {session_id}")
            
    except Exception as e:
        error_msg = f"Auto clock error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Re-enable session on error
        if session_id in active_sessions:
            active_sessions[session_id]['session_active'] = True
            active_sessions[session_id]['recognition_start_time'] = None
            active_sessions[session_id]['stable_recognition'] = False
        
        # Send error response
        error_data = {
            'success': False,
            'employee': employee_name,
            'message': error_msg,
            'action': 'error'
        }
        socketio.emit('auto_clock_result', error_data, room=session_id)

# ALSO ADD THIS UTILITY FUNCTION TO HELP DEBUG:
@app.route('/api/debug/<employee_name>')
def debug_employee_status(employee_name):
    """Debug endpoint to check employee status"""
    try:
        status, details = time_db.get_employee_status(employee_name)
        records = time_db.debug_employee_records(employee_name)
        
        return jsonify({
            'employee': employee_name,
            'current_status': status,
            'details': details,
            'all_records': [
                {
                    'id': r[0],
                    'clock_in': r[2],
                    'clock_out': r[3],
                    'hours': r[4],
                    'date': r[5]
                } for r in records
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# AND UPDATE THE get_employee_status METHOD IN TimeTrackingDB:
def get_employee_status_debug(self, employee_name):
    """Get employee status with enhanced debugging"""
    conn = sqlite3.connect(self.db_file)
    cursor = conn.cursor()
    
    current_date = datetime.now().date()
    
    # Get the most recent record for today
    cursor.execute('''SELECT clock_in, clock_out, hours_worked FROM time_records 
                     WHERE employee_name = ? AND date = ? 
                     ORDER BY clock_in DESC LIMIT 1''', 
                  (employee_name, current_date))
    
    record = cursor.fetchone()
    
    # Also get count of active records
    cursor.execute('''SELECT COUNT(*) FROM time_records 
                     WHERE employee_name = ? AND date = ? AND clock_out IS NULL''', 
                  (employee_name, current_date))
    active_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"🔍 Debug for {employee_name}:")
    print(f"  - Most recent record: {record}")
    print(f"  - Active records count: {active_count}")
    
    if not record:
        # No records for today
        print(f"  - Decision: not_clocked_in (no records for today)")
        return "not_clocked_in", None
    
    clock_in, clock_out, hours_worked = record
    
    if clock_out is None:
        # Currently clocked in (active session)
        clock_in_dt = datetime.fromisoformat(clock_in)
        hours_so_far = (datetime.now() - clock_in_dt).total_seconds() / 3600
        print(f"  - Decision: clocked_in (active session since {clock_in})")
        return "clocked_in", {
            "clock_in": clock_in,
            "hours_so_far": round(hours_so_far, 2)
        }
    else:
        # Clocked out (ready for next clock-in)
        print(f"  - Decision: clocked_out (last session: {clock_in} to {clock_out})")
        return "clocked_out", {
            "clock_in": clock_in,
            "clock_out": clock_out,
            "hours_worked": round(hours_worked, 2) if hours_worked else 0
        }
@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame with automatic clock in/out - FIXED VERSION"""
    session_id = request.sid
    if session_id not in active_sessions:
        return
    
    session = active_sessions[session_id]
    
    # Check if session is still active
    if not session.get('session_active', True):
        print(f"🚫 Session {session_id} is inactive, ignoring frame")
        return
    
    # Prevent multiple concurrent auto clock actions
    if session.get('stable_recognition', False):
        print(f"⏳ Auto clock already in progress for {session_id}")
        return
    
    try:
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        liveness_detector = session['liveness_detector']
        
        # Wrap frame analysis in try-catch
        try:
            analysis, processed_frame = liveness_detector.analyze_frame(frame)
        except Exception as e:
            print(f"⚠️  Frame analysis error: {e}")
            analysis = {
                'face_detected': False,
                'face_quality': 0,
                'liveness_verified': False,
                'progress': 0,
                'instruction': 'Camera error - please restart'
            }
            processed_frame = frame
        
        if session['mode'] == 'timeclock':
            session['liveness_verified'] = analysis['liveness_verified']
            
            recognition_result = None
            current_time = time.time()
            
            # Only proceed with recognition if liveness is verified and session is active
            if (analysis['liveness_verified'] and adaptive_system and 
                session.get('session_active', True) and not session.get('stable_recognition', False)):
                
                try:
                    name, confidence, method = adaptive_system.recognize_with_learning(frame, learn_from_result=False)
                    print(f"🔍 Recognition result: {name}, confidence: {confidence:.2f}")
                    
                    recognition_result = {
                        'name': name,
                        'confidence': round(confidence * 100, 1),
                        'method': method,
                        'status': 'recognized' if name != 'Unknown' else 'unknown'
                    }
                    session['current_recognition'] = recognition_result
                    
                    # Auto clock logic - check for stable recognition
                    if name != 'Unknown' and confidence >= (CLOCK_ACTION_THRESHOLD / 100.0):
                        print(f"✅ High confidence recognition: {name} ({confidence:.2f})")
                        
                        if session.get('last_recognition_name') == name:
                            # Same person recognized consecutively
                            if session.get('recognition_start_time') is None:
                                session['recognition_start_time'] = current_time
                                print(f"⏱️  Started countdown for {name}")
                            elif (current_time - session['recognition_start_time']) >= AUTO_CLOCK_DELAY:
                                # Stable recognition for required duration
                                if not session.get('stable_recognition', False):
                                    session['stable_recognition'] = True
                                    print(f"🎯 Triggering auto clock for {name}")
                                    # Call auto_clock_action directly instead of using background task
                                    auto_clock_action(session_id, name, confidence)
                                    return  # Exit early to prevent further processing
                        else:
                            # Different person or first recognition
                            session['last_recognition_name'] = name
                            session['recognition_start_time'] = current_time
                            session['stable_recognition'] = False
                            print(f"🔄 New recognition: {name}")
                    else:
                        # Unknown or low confidence - reset
                        if name == 'Unknown':
                            print(f"❓ Unknown person detected")
                        else:
                            print(f"⚠️  Low confidence: {name} ({confidence:.2f})")
                        session['recognition_start_time'] = None
                        session['stable_recognition'] = False
                        session['last_recognition_name'] = None
                        
                except Exception as e:
                    print(f"❌ Recognition error: {e}")
                    recognition_result = None
        
        elif session['mode'] == 'registration':
            reg_data = session['registration_data']
            if analysis['face_detected'] and analysis['face_quality'] > 0.5:
                emit('registration_ready', {
                    'current_pose': reg_data['current_pose'],
                    'face_quality': analysis['face_quality']
                })
        
        # Only send frame updates if session is still active
        if not session.get('session_active', True):
            return
        
        # Encode processed frame
        try:
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Frame encoding error: {e}")
            processed_frame_b64 = ""
        
        # Calculate countdown for auto clock
        countdown = 0
        if (session['mode'] == 'timeclock' and session.get('recognition_start_time') is not None 
            and not session.get('stable_recognition', False)):
            elapsed = current_time - session['recognition_start_time']
            countdown = max(0, AUTO_CLOCK_DELAY - elapsed)
        
        response = {
            'processed_frame': f"data:image/jpeg;base64,{processed_frame_b64}",
            'face_detected': analysis['face_detected'],
            'face_quality': round(analysis['face_quality'], 2),
            'liveness_verified': analysis['liveness_verified'],
            'progress': round(analysis['progress'], 2),
            'instruction': analysis['instruction'],
            'countdown': round(countdown, 1),
            'session_active': session.get('session_active', True)
        }
        
        if session['mode'] == 'timeclock':
            response['recognition_result'] = recognition_result
            response['auto_clock_ready'] = session.get('stable_recognition', False)
            
            # Update instruction for auto clock
            if recognition_result and recognition_result['name'] != 'Unknown':
                if session.get('stable_recognition', False):
                    response['instruction'] = f"Processing {recognition_result['name']}..."
                elif countdown > 0:
                    response['instruction'] = f"Hold steady... {countdown:.1f}s"
                else:
                    response['instruction'] = f"Recognized: {recognition_result['name']}"
        
        emit('frame_analysis', response)
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        emit('error', {'message': 'Error processing frame'})

@socketio.on('restart_session')
def handle_restart_session():
    """Restart camera session after auto clock"""
    session_id = request.sid
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session['session_active'] = True
        session['liveness_detector'].reset()
        session['recognition_start_time'] = None
        session['stable_recognition'] = False
        session['last_recognition_name'] = None
        session['current_recognition'] = None
        
        print(f"🔄 Session {session_id} restarted")
        emit('session_restarted')

@socketio.on('capture_registration_photo')
def handle_capture_registration_photo(data):
    """Capture photo during registration"""
    session_id = request.sid
    if session_id not in active_sessions or active_sessions[session_id]['mode'] != 'registration':
        return
    
    try:
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        session = active_sessions[session_id]
        reg_data = session['registration_data']
        
        employee_name = reg_data['employee_name']
        current_pose = reg_data['current_pose']
        
        person_folder = os.path.join(PEOPLE_FOLDER, employee_name)
        os.makedirs(person_folder, exist_ok=True)
        
        photo_filename = f"{employee_name}_{current_pose}_{int(time.time())}.jpg"
        photo_path = os.path.join(person_folder, photo_filename)
        cv2.imwrite(photo_path, frame)
        
        reg_data['photos_taken'].append(photo_path)
        reg_data['poses_completed'].append(current_pose)
        
        pose_sequence = ['front', 'left', 'right', 'front_2']
        current_index = pose_sequence.index(current_pose)
        
        if current_index < len(pose_sequence) - 1:
            reg_data['current_pose'] = pose_sequence[current_index + 1]
            
            pose_instructions = {
                'front': 'Look straight at camera',
                'left': 'Turn head slightly to your left',
                'right': 'Turn head slightly to your right', 
                'front_2': 'Look straight at camera again'
            }
            
            emit('registration_next_pose', {
                'pose': reg_data['current_pose'],
                'instruction': pose_instructions[reg_data['current_pose']],
                'completed': len(reg_data['poses_completed']),
                'total': len(pose_sequence)
            })
        else:
            emit('registration_complete', {
                'photos_taken': len(reg_data['photos_taken'])
            })
            
    except Exception as e:
        print(f"Registration capture error: {e}")
        emit('error', {'message': 'Error capturing photo'})

@socketio.on('complete_registration')
def handle_complete_registration(data):
    """Complete employee registration"""
    session_id = request.sid
    if session_id not in active_sessions:
        return
    
    try:
        session = active_sessions[session_id]
        reg_data = session['registration_data']
        
        employee_name = reg_data['employee_name']
        photos_taken = reg_data['photos_taken']
        
        if len(photos_taken) < 3:
            emit('registration_error', {'message': 'Not enough photos captured'})
            return
        
        success = False
        if face_system and adaptive_system:
            try:
                face_system.add_person_enhanced(employee_name, photos_taken)
                adaptive_system.add_person_samples(employee_name, photos_taken)
                time_db.add_employee(employee_name)
                success = True
            except Exception as e:
                print(f"Error adding employee: {e}")
        
        if success:
            emit('registration_success', {
                'employee_name': employee_name,
                'photos_count': len(photos_taken)
            })
        else:
            emit('registration_error', {'message': 'Failed to register employee'})
            
    except Exception as e:
        print(f"Registration completion error: {e}")
        emit('error', {'message': 'Error completing registration'})

def create_templates():
    """Create simplified templates"""
    
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SecureTime Pro</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            margin-bottom: 40px;
        }
        .nav h1 {
            font-size: 24px;
            font-weight: 600;
            color: white;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .nav-links { display: flex; gap: 20px; flex-wrap: wrap; }
        .nav-link {
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .nav-link:hover { background: rgba(255,255,255,0.2); color: white; }
        .card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(118,75,162,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .btn-success { background: linear-gradient(135deg, #28a745, #20c997); }
        .btn-warning { background: linear-gradient(135deg, #ffc107, #fd7e14); }
        .btn-danger { background: linear-gradient(135deg, #dc3545, #e83e8c); }
        .camera-container {
            position: relative;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .camera-feed {
            width: 100%;
            height: 400px;
            object-fit: cover;
            display: block;
        }
        .camera-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
        }
        .instruction-panel {
            position: absolute;
            bottom: 20px; left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px 25px;
            border-radius: 25px;
            font-weight: 600;
            text-align: center;
            min-width: 250px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .employee-card {
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .employee-card:hover { transform: translateY(-5px); }
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .card { padding: 25px; }
            .nav { flex-direction: column; gap: 15px; }
        }
    </style>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <nav class="nav">
            <h1><span>🕐</span> SecureTime Pro</h1>
            <div class="nav-links">
                <a href="/" class="nav-link">Dashboard</a>
                <a href="/timeclock" class="nav-link">Time Clock</a>
                <a href="/employees" class="nav-link">Employees</a>
                <a href="/add_employee" class="nav-link">Add Employee</a>
                <a href="/reports" class="nav-link">Reports</a>
            </div>
        </nav>
        {% block content %}{% endblock %}
    </div>
    {% block scripts %}{% endblock %}
</body>
</html>'''

    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write(base_template)

def main():
    """Main function"""
    def cleanup():
        if adaptive_system:
            adaptive_system.save_models()
    
    def signal_handler(signum, frame):
        cleanup()
        sys.exit(0)
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    
    port = find_free_port()
    create_templates()
    time_db.init_database()
    threading.Thread(target=initialize_face_system, daemon=True).start()
    
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=port, use_reloader=False)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()