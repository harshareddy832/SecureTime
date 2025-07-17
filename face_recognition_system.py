import cv2
import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class LocalFaceRecognition:
    def __init__(self, model_type='large'):
        """
        Initialize the face recognition system
        model_type: 'small' (faster) or 'large' (more accurate)
        """
        self.model_type = model_type
        self.known_faces = {}
        self.known_names = []
        self.known_encodings = []
        
    def extract_face_encoding(self, image_path):
        """
        Extract face encoding from an image
        Returns 128-dimensional face encoding vector
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations
        face_locations = face_recognition.face_locations(
            image, 
            model="cnn" if self.model_type == 'large' else "hog"
        )
        
        if not face_locations:
            return None
            
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations,
            model=self.model_type
        )
        
        if face_encodings:
            return face_encodings[0]  # Return first face encoding
        return None
    
    def add_person(self, name, image_paths):
        """
        Add a person to the database with multiple images
        """
        encodings = []
        for image_path in image_paths:
            encoding = self.extract_face_encoding(image_path)
            if encoding is not None:
                encodings.append(encoding)
                
        if encodings:
            # Average multiple encodings for better representation
            avg_encoding = np.mean(encodings, axis=0)
            self.known_faces[name] = {
                'encoding': avg_encoding,
                'individual_encodings': encodings,
                'image_count': len(encodings)
            }
            self.known_names.append(name)
            self.known_encodings.append(avg_encoding)
        else:
            print(f"No valid faces found for {name}")
    
    def recognize_face(self, image_path, tolerance=0.4):
        """
        Recognize a face in the given image
        Lower tolerance = more strict matching
        """
        unknown_encoding = self.extract_face_encoding(image_path)
        
        if unknown_encoding is None:
            return None, 0.0
            
        if not self.known_encodings:
            return "Unknown", 0.0
            
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(self.known_encodings, unknown_encoding)
        best_match_index = np.argmin(distances)
        
        if distances[best_match_index] <= tolerance:
            confidence = 1 - distances[best_match_index]  # Convert distance to confidence
            return self.known_names[best_match_index], confidence
        else:
            return "Unknown", 1 - distances[best_match_index]
    
    def batch_recognize(self, image_folder, tolerance=0.4):
        """
        Recognize faces in all images in a folder
        """
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for image_file in os.listdir(image_folder):
            if any(image_file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_folder, image_file)
                name, confidence = self.recognize_face(image_path, tolerance)
                results.append({
                    'image': image_file,
                    'predicted_name': name,
                    'confidence': confidence
                })
        
        return results
    
    def save_model(self, filename='face_recognition_model.pkl'):
        """
        Save the trained model
        """
        model_data = {
            'known_faces': self.known_faces,
            'known_names': self.known_names,
            'known_encodings': self.known_encodings,
            'model_type': self.model_type
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='face_recognition_model.pkl'):
        """
        Load a pre-trained model
        """
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.known_faces = model_data['known_faces']
            self.known_names = model_data['known_names']
            self.known_encodings = model_data['known_encodings']
            self.model_type = model_data['model_type']
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
    
    def evaluate_accuracy(self, test_folder):
        """
        Evaluate accuracy on a test dataset
        Assumes folder structure: test_folder/person_name/image.jpg
        """
        correct = 0
        total = 0
        results = []
        
        for person_folder in os.listdir(test_folder):
            person_path = os.path.join(test_folder, person_folder)
            if os.path.isdir(person_path):
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        predicted_name, confidence = self.recognize_face(image_path)
                        
                        actual_name = person_folder
                        is_correct = predicted_name == actual_name
                        
                        results.append({
                            'actual': actual_name,
                            'predicted': predicted_name,
                            'confidence': confidence,
                            'correct': is_correct
                        })
                        
                        if is_correct:
                            correct += 1
                        total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        return accuracy, results


# Real-time face recognition from webcam
class RealTimeFaceRecognition:
    def __init__(self, face_recognizer):
        self.face_recognizer = face_recognizer
        
    def run_webcam_recognition(self):
        """
        Run real-time face recognition from webcam
        """
        video_capture = cv2.VideoCapture(0)
        
        print("Starting webcam... Press 'q' to quit")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Recognize each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check if face matches known faces
                if self.face_recognizer.known_encodings:
                    distances = face_recognition.face_distance(
                        self.face_recognizer.known_encodings, 
                        face_encoding
                    )
                    best_match_index = np.argmin(distances)
                    
                    if distances[best_match_index] < 0.4:
                        name = self.face_recognizer.known_names[best_match_index]
                        confidence = 1 - distances[best_match_index]
                        label = f"{name} ({confidence:.2f})"
                    else:
                        label = "Unknown"
                else:
                    label = "No known faces"
                
                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()