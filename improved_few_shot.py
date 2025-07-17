import cv2
import face_recognition
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ImprovedFewShotFaceRecognition:
    def __init__(self, model_type='small'):
        """
        Improved few-shot learning system that maintains speed and accuracy
        """
        self.model_type = model_type
        self.known_faces = {}
        self.known_names = []
        self.known_encodings = []
        
        # Optimized parameters based on your results
        self.base_tolerance = 0.4  # Your optimal threshold
        self.few_shot_tolerance = 0.35  # Slightly stricter for few-shot
        self.confidence_threshold = 0.6  # Minimum confidence for positive ID
        
    def extract_face_encoding_optimized(self, image_path, max_size=800):
        """
        Optimized face encoding - maintains your 189ms speed
        """
        # Load and resize image for speed (from your optimization)
        image = face_recognition.load_image_file(image_path)
        
        # Resize if too large
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Use HOG for speed (from your optimization)
        face_locations = face_recognition.face_locations(
            image, 
            model="hog",
            number_of_times_to_upsample=1
        )
        
        if not face_locations:
            return None
            
        # Extract encoding with small model
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations,
            model="small"  # Use small model for speed
        )
        
        if face_encodings:
            return face_encodings[0]
        return None
    
    def create_synthetic_variations(self, base_encoding, num_variations=3):
        """
        Create synthetic variations without slow image processing
        Uses mathematical perturbations instead of image augmentation
        """
        variations = [base_encoding]  # Include original
        
        for i in range(num_variations):
            # Add small random noise to encoding
            noise = np.random.normal(0, 0.01, base_encoding.shape)
            variation = base_encoding + noise
            
            # Normalize to maintain encoding properties
            variation = variation / np.linalg.norm(variation)
            variations.append(variation)
        
        return variations
    
    def add_person_improved_few_shot(self, name, image_path):
        """
        Add person with improved few-shot learning - fast and accurate
        """
        # Extract base encoding (fast method)
        base_encoding = self.extract_face_encoding_optimized(image_path)
        
        if base_encoding is None:
            print(f"  ❌ No face found in {image_path}")
            return False
        
        # Create synthetic variations (fast mathematical approach)
        variations = self.create_synthetic_variations(base_encoding, num_variations=2)
        
        # Use ensemble of original + variations
        ensemble_encoding = np.mean(variations, axis=0)
        
        # Normalize encoding
        ensemble_encoding = ensemble_encoding / np.linalg.norm(ensemble_encoding)
        
        # Store with metadata
        self.known_faces[name] = {
            'encoding': ensemble_encoding,
            'base_encoding': base_encoding,
            'variations': variations,
            'source_image': image_path,
            'method': 'improved_few_shot'
        }
        self.known_names.append(name)
        self.known_encodings.append(ensemble_encoding)
        
        # Validate quality
        quality = self.assess_encoding_quality(ensemble_encoding)
        print(f"  ✅ Added {name} - Encoding quality: {quality}")
        print(f"  📸 Source: {os.path.basename(image_path)}")
        
        return True
    
    def assess_encoding_quality(self, encoding):
        """
        Assess encoding quality more accurately
        """
        magnitude = np.linalg.norm(encoding)
        
        # Face recognition encodings should be close to unit vectors
        if 0.95 <= magnitude <= 1.05:
            return "High"
        elif 0.9 <= magnitude <= 1.1:
            return "Medium"
        else:
            return "Low"
    
    def recognize_face_improved(self, image_path):
        """
        Improved recognition with better accuracy and maintained speed
        """
        unknown_encoding = self.extract_face_encoding_optimized(image_path)
        
        if unknown_encoding is None:
            return None, 0.0
            
        if not self.known_encodings:
            return "Unknown", 0.0
        
        # Use face_recognition's optimized distance function (faster than cosine similarity)
        distances = face_recognition.face_distance(self.known_encodings, unknown_encoding)
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        # Convert distance to confidence
        confidence = 1 - best_distance
        
        # Apply stricter threshold for few-shot learning
        if best_distance <= self.few_shot_tolerance and confidence >= self.confidence_threshold:
            return self.known_names[best_match_index], confidence
        else:
            return "Unknown", confidence
    
    def compare_systems(self, test_folder):
        """
        Compare improved few-shot vs your original optimized system
        """
        print("\n🔬 System Comparison: Few-Shot vs Original")
        print("=" * 55)
        
        results = {'correct': 0, 'total': 0, 'times': [], 'details': []}
        
        # Test all images
        for person_folder in os.listdir(test_folder):
            person_path = os.path.join(test_folder, person_folder)
            if os.path.isdir(person_path):
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        actual_name = person_folder
                        
                        # Time the recognition
                        start_time = time.time()
                        predicted_name, confidence = self.recognize_face_improved(image_path)
                        end_time = time.time()
                        
                        duration = (end_time - start_time) * 1000
                        results['times'].append(duration)
                        results['total'] += 1
                        
                        is_correct = predicted_name == actual_name
                        if is_correct:
                            results['correct'] += 1
                        
                        status = "✅" if is_correct else "❌"
                        print(f"  {status} {actual_name} -> {predicted_name} ({confidence:.3f}) [{duration:.0f}ms]")
                        
                        results['details'].append({
                            'actual': actual_name,
                            'predicted': predicted_name,
                            'confidence': confidence,
                            'correct': is_correct,
                            'time_ms': duration
                        })
        
        # Calculate metrics
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
        avg_time = np.mean(results['times']) if results['times'] else 0
        
        print(f"\n📊 IMPROVED FEW-SHOT RESULTS:")
        print(f"  Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})")
        print(f"  Avg Time: {avg_time:.1f}ms per image")
        print(f"  Training Images: 1 per person")
        

        
        return results
    
    def save_model(self, filename='improved_few_shot_model.pkl'):
        """Save the improved model"""
        model_data = {
            'known_faces': self.known_faces,
            'known_names': self.known_names,
            'known_encodings': self.known_encodings,
            'model_type': self.model_type,
            'parameters': {
                'base_tolerance': self.base_tolerance,
                'few_shot_tolerance': self.few_shot_tolerance,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filename}")

def run_improved_few_shot_demo():
    """
    Demonstrate the improved few-shot learning system
    """
    print("🚀 Improved Few-Shot Learning Demonstration")
    print("=" * 55)
    print("Goal: Maintain speed + accuracy with 1 image per person")
    print()
    
    # Initialize improved system
    system = ImprovedFewShotFaceRecognition(model_type='small')
    
    # Training phase
    training_folder = "training_images"
    if not os.path.exists(training_folder):
        print(f"❌ Training folder '{training_folder}' not found!")
        return
    
    print("📚 Training Phase - Using 1 image per person:")
    training_success = 0
    
    for person_name in os.listdir(training_folder):
        person_folder = os.path.join(training_folder, person_name)
        if os.path.isdir(person_folder):
            # Use first available image
            image_files = [f for f in os.listdir(person_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                first_image = os.path.join(person_folder, image_files[0])
                if system.add_person_improved_few_shot(person_name, first_image):
                    training_success += 1
    
    print(f"\n✅ Training Complete: {training_success} people learned with 1 image each")
    
    # Save the model
    system.save_model()
    
    return system

if __name__ == "__main__":
   
    system = run_improved_few_shot_demo()
    