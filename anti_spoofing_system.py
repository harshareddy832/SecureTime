# anti_spoofing_system.py
import cv2
import numpy as np
import dlib
import time
import random
from collections import deque

class BlinkDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.blink_history = deque(maxlen=20)
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        self.blink_counter = 0
        self.frame_counter = 0
        
    def eye_aspect_ratio(self, eye):
        # Compute eye aspect ratio
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return False, 0
            
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
            
            # Extract eye coordinates
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Calculate eye aspect ratio
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Check for blink
            if ear < self.EYE_AR_THRESH:
                self.frame_counter += 1
            else:
                if self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.blink_counter += 1
                self.frame_counter = 0
                
            self.blink_history.append(ear)
            
        return True, self.blink_counter

class HeadMovementTracker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.previous_nose_tip = None
        self.movement_threshold = 15
        
    def track_head_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return False, "no_face"
            
        face = faces[0]
        landmarks = self.predictor(gray, face)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        
        if self.previous_nose_tip is None:
            self.previous_nose_tip = nose_tip
            return True, "initialized"
            
        # Calculate movement
        dx = nose_tip[0] - self.previous_nose_tip[0]
        dy = nose_tip[1] - self.previous_nose_tip[1]
        
        movement = ""
        if abs(dx) > self.movement_threshold:
            movement += "left" if dx < 0 else "right"
        if abs(dy) > self.movement_threshold:
            movement += "_up" if dy < 0 else "_down"
            
        self.previous_nose_tip = nose_tip
        return True, movement if movement else "still"

class ChallengeGenerator:
    def __init__(self):
        self.challenges = [
            {"type": "blink", "instruction": "Blink twice", "target": 2},
            {"type": "head_left", "instruction": "Look left", "target": "left"},
            {"type": "head_right", "instruction": "Look right", "target": "right"},
            {"type": "head_up", "instruction": "Look up", "target": "up"},
            {"type": "smile", "instruction": "Smile", "target": "smile"}
        ]
        
    def generate_challenge(self):
        return random.choice(self.challenges)

class AntiSpoofingSystem:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.head_tracker = HeadMovementTracker()
        self.challenge_generator = ChallengeGenerator()
        self.verification_timeout = 10  # seconds
        
    def verify_liveness(self, video_stream_callback, challenge_callback=None):
        """
        Verify liveness using multiple methods
        video_stream_callback: function that yields video frames
        challenge_callback: function to display challenges to user
        """
        start_time = time.time()
        challenge = self.challenge_generator.generate_challenge()
        
        if challenge_callback:
            challenge_callback(challenge["instruction"])
            
        blink_count = 0
        challenge_completed = False
        
        for frame in video_stream_callback():
            if time.time() - start_time > self.verification_timeout:
                break
                
            # Detect blinks
            face_detected, current_blinks = self.blink_detector.detect_blink(frame)
            if not face_detected:
                continue
                
            # Track head movement
            head_detected, movement = self.head_tracker.track_head_movement(frame)
            
            # Check challenge completion
            if challenge["type"] == "blink" and current_blinks >= challenge["target"]:
                challenge_completed = True
                break
            elif challenge["type"].startswith("head_") and challenge["target"] in movement:
                challenge_completed = True
                break
                
        return {
            "liveness_verified": challenge_completed,
            "challenge_type": challenge["type"],
            "blink_count": current_blinks if face_detected else 0,
            "time_taken": time.time() - start_time
        }

# Integration with your existing face_recognition_system.py
class SecureTimeProSystem:
    def __init__(self, face_recognition_system):
        self.face_system = face_recognition_system  # Your existing EnhancedFaceRecognition
        self.anti_spoofing = AntiSpoofingSystem()
        self.time_records = {}
        
    def secure_clock_in(self, video_stream_callback, challenge_callback=None):
        """
        Secure clock-in process with anti-spoofing
        """
        # Step 1: Verify liveness
        liveness_result = self.anti_spoofing.verify_liveness(
            video_stream_callback, challenge_callback
        )
        
        if not liveness_result["liveness_verified"]:
            return {
                "success": False,
                "message": "Liveness verification failed",
                "details": liveness_result
            }
            
        # Step 2: Capture frame for recognition
        frame = next(video_stream_callback())
        temp_path = f"temp_recognition_{int(time.time())}.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Step 3: Use your existing recognition system
        try:
            name, confidence, method = self.face_system.recognize_enhanced(temp_path)
            
            if name != "Unknown" and confidence > 0.8:
                # Record clock-in time
                clock_in_time = time.time()
                self.time_records[name] = {
                    "clock_in": clock_in_time,
                    "confidence": confidence,
                    "method": method,
                    "liveness_verified": True
                }
                
                return {
                    "success": True,
                    "employee": name,
                    "confidence": confidence,
                    "time": clock_in_time,
                    "liveness_details": liveness_result
                }
            else:
                return {
                    "success": False,
                    "message": "Face not recognized",
                    "confidence": confidence
                }
                
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Flask route integration example
"""
@app.route('/api/secure_clock_in', methods=['POST'])
def secure_clock_in():
    global secure_system
    
    def video_stream():
        # Implement video capture from webcam
        cap = cv2.VideoCapture(0)
        for _ in range(100):  # Process 100 frames max
            ret, frame = cap.read()
            if ret:
                yield frame
        cap.release()
    
    def challenge_display(instruction):
        # Send challenge to frontend via WebSocket or similar
        socketio.emit('challenge', {'instruction': instruction})
    
    result = secure_system.secure_clock_in(video_stream, challenge_display)
    return jsonify(result)
"""