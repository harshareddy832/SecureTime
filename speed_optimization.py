# Speed Optimization for Face Recognition System
import cv2
import face_recognition
import numpy as np
import time
from face_recognition_system import LocalFaceRecognition

class OptimizedFaceRecognition(LocalFaceRecognition):
    def __init__(self, model_type='small'):  # Default to small for speed
        super().__init__(model_type)
        self.face_detection_model = "hog"  # Faster than CNN
        
    def extract_face_encoding_fast(self, image_path, max_size=800):
        """
        Optimized face encoding extraction
        """
        # Load and resize image for speed
        image = face_recognition.load_image_file(image_path)
        
        # Resize if image is too large
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Use HOG model for speed (instead of CNN)
        face_locations = face_recognition.face_locations(
            image, 
            model="hog",  # Much faster than CNN
            number_of_times_to_upsample=1  # Reduce upsampling for speed
        )
        
        if not face_locations:
            return None
            
        # Extract face encodings with small model
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations,
            model="small"  # Faster model
        )
        
        if face_encodings:
            return face_encodings[0]
        return None
    
    def recognize_face_fast(self, image_path, tolerance=0.4, max_size=800):
        """
        Optimized face recognition
        """
        unknown_encoding = self.extract_face_encoding_fast(image_path, max_size)
        
        if unknown_encoding is None:
            return None, 0.0
            
        if not self.known_encodings:
            return "Unknown", 0.0
            
        # Use optimized distance calculation
        distances = face_recognition.face_distance(self.known_encodings, unknown_encoding)
        best_match_index = np.argmin(distances)
        
        if distances[best_match_index] <= tolerance:
            confidence = 1 - distances[best_match_index]
            return self.known_names[best_match_index], confidence
        else:
            return "Unknown", 1 - distances[best_match_index]

def speed_comparison_test():
    """
    Compare speed of original vs optimized system
    """
    print("🏃‍♂️ Speed Optimization Test")
    print("=" * 40)
    
    # Load your existing model
    original_system = LocalFaceRecognition(model_type='large')
    optimized_system = OptimizedFaceRecognition(model_type='small')
    
    # Load training data into both systems
    training_folder = "training_images"
    
    print("Loading training data...")
    for person_name in ["ElonMusk", "SamAltman"]:  # Adjust if different
        person_folder = f"{training_folder}/{person_name}"
        if os.path.exists(person_folder):
            image_paths = []
            for img_file in os.listdir(person_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(f"{person_folder}/{img_file}")
            
            if image_paths:
                original_system.add_person(person_name, image_paths)
                optimized_system.add_person(person_name, image_paths)
    
    # Find a test image
    test_image = None
    test_folder = "test_images"
    for person_folder in os.listdir(test_folder):
        person_path = f"{test_folder}/{person_folder}"
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = f"{person_path}/{image_file}"
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test image found")
        return
    
    print(f"Testing with: {test_image}")
    print()
    
    # Test original system
    print("🐌 Testing Original System (Large Model + CNN)...")
    original_times = []
    for i in range(3):
        start_time = time.time()
        name, confidence = original_system.recognize_face(test_image)
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        original_times.append(duration)
        print(f"  Run {i+1}: {duration:.0f}ms - {name} ({confidence:.3f})")
    
    original_avg = np.mean(original_times)
    
    print()
    print("🚀 Testing Optimized System (Small Model + HOG)...")
    optimized_times = []
    for i in range(3):
        start_time = time.time()
        name, confidence = optimized_system.recognize_face_fast(test_image)
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        optimized_times.append(duration)
        print(f"  Run {i+1}: {duration:.0f}ms - {name} ({confidence:.3f})")
    
    optimized_avg = np.mean(optimized_times)
    
    # Results
    speedup = original_avg / optimized_avg
    print()
    print("📊 SPEED COMPARISON RESULTS:")
    print("=" * 40)
    print(f"Original System:  {original_avg:.0f}ms average")
    print(f"Optimized System: {optimized_avg:.0f}ms average")
    print(f"Speed Improvement: {speedup:.1f}x faster")
    print(f"Time Saved: {original_avg - optimized_avg:.0f}ms per image")

    
def create_production_system():
    """
    Create an optimized production-ready system
    """
    print("\n🏭 Creating Production-Ready System...")
    
    # Use optimized settings
    system = OptimizedFaceRecognition(model_type='small')
    
    # Load your trained model if it exists
    if os.path.exists("my_face_model.pkl"):
        print("📦 Loading existing model...")
        
    return system

if __name__ == "__main__":
    import os
    
    # Run speed comparison
    results = speed_comparison_test()
    
    # Create optimized system
    production_system = create_production_system()    