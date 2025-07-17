import numpy as np
import face_recognition
import cv2
import os
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from speed_optimization import OptimizedFaceRecognition

class EnhancedFaceRecognition:
    """Enhanced face recognition with ensemble learning"""
    
    def __init__(self):
        self.base_system = OptimizedFaceRecognition(model_type='small')
        self.similarity_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.feature_scaler = StandardScaler()
        self.similarity_trained = False
        self.enhanced_features = {}
        self.training_pairs = []
        
    def extract_enhanced_features(self, encoding1, encoding2):
        """Extract enhanced features from a pair of face encodings"""
        cosine_sim = cosine_similarity([encoding1], [encoding2])[0][0]
        euclidean_dist = np.linalg.norm(encoding1 - encoding2)
        
        element_wise_diff = np.abs(encoding1 - encoding2)
        max_diff = np.max(element_wise_diff)
        min_diff = np.min(element_wise_diff)
        mean_diff = np.mean(element_wise_diff)
        std_diff = np.std(element_wise_diff)
        
        correlation = np.corrcoef(encoding1, encoding2)[0][1]
        if np.isnan(correlation):
            correlation = 0.0
        
        manhattan_dist = np.sum(element_wise_diff)
        
        mag1 = np.linalg.norm(encoding1)
        mag2 = np.linalg.norm(encoding2)
        mag_ratio = mag1 / mag2 if mag2 != 0 else 1.0
        
        dot_product = np.dot(encoding1, encoding2)
        cross_correlation = np.sum(encoding1 * encoding2)
        
        features = [
            cosine_sim, euclidean_dist, max_diff, min_diff, mean_diff,
            std_diff, correlation, manhattan_dist, mag_ratio, dot_product, cross_correlation
        ]
        
        return np.array(features)
    
    def create_training_data(self, training_folder):
        """Create training data for the enhanced classifier"""
        people_encodings = {}
        for person_name in os.listdir(training_folder):
            person_folder = os.path.join(training_folder, person_name)
            if os.path.isdir(person_folder):
                encodings = []
                for img_file in os.listdir(person_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_folder, img_file)
                        encoding = self.base_system.extract_face_encoding_fast(img_path)
                        if encoding is not None:
                            encodings.append(encoding)
                
                if len(encodings) >= 2:
                    people_encodings[person_name] = encodings
        
        X_features = []
        y_labels = []
        
        # Positive pairs (same person)
        for person, encodings in people_encodings.items():
            for i in range(len(encodings)):
                for j in range(i + 1, len(encodings)):
                    features = self.extract_enhanced_features(encodings[i], encodings[j])
                    X_features.append(features)
                    y_labels.append(1)
        
        # Negative pairs (different people)
        people_list = list(people_encodings.items())
        for i in range(len(people_list)):
            for j in range(i + 1, len(people_list)):
                person1_encodings = people_list[i][1]
                person2_encodings = people_list[j][1]
                
                for enc1 in person1_encodings:
                    for enc2 in person2_encodings:
                        features = self.extract_enhanced_features(enc1, enc2)
                        X_features.append(features)
                        y_labels.append(0)
        
        if len(X_features) == 0:
            return False
        
        X_features = np.array(X_features)
        y_labels = np.array(y_labels)
        
        return X_features, y_labels
    
    def train_enhanced_classifier(self, training_folder):
        """Train the enhanced classifier"""
        training_data = self.create_training_data(training_folder)
        if not training_data:
            return False
        
        X_features, y_labels = training_data
        X_scaled = self.feature_scaler.fit_transform(X_features)
        self.similarity_classifier.fit(X_scaled, y_labels)
        
        feature_names = [
            'cosine_similarity', 'euclidean_distance', 'max_diff', 'min_diff',
            'mean_diff', 'std_diff', 'correlation', 'manhattan_distance',
            'magnitude_ratio', 'dot_product', 'cross_correlation'
        ]
        
        importances = self.similarity_classifier.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        self.similarity_trained = True
        return True
    
    def add_person_enhanced(self, name, image_paths):
        """Add person using enhanced system"""
        self.base_system.add_person(name, image_paths)
        
        if name in self.base_system.known_faces:
            person_data = self.base_system.known_faces[name]
            if 'individual_encodings' in person_data and len(person_data['individual_encodings']) >= 2:
                encodings = person_data['individual_encodings']
                self_features = []
                
                for i in range(len(encodings)):
                    for j in range(i + 1, len(encodings)):
                        features = self.extract_enhanced_features(encodings[i], encodings[j])
                        self_features.append(features)
                
                if self_features:
                    self.enhanced_features[name] = {
                        'self_similarity_features': self_features,
                        'avg_self_similarity': np.mean([f[0] for f in self_features])
                    }
        
        return True
    
    def recognize_enhanced(self, image_path, use_enhanced=True):
        """Enhanced recognition using base system + ensemble learning"""
        base_name, base_confidence = self.base_system.recognize_face_fast(image_path)
        
        if not use_enhanced or not self.similarity_trained:
            return base_name, base_confidence, "base_only"
        
        unknown_encoding = self.base_system.extract_face_encoding_fast(image_path)
        if unknown_encoding is None:
            return base_name, base_confidence, "base_only"
        
        best_enhanced_confidence = 0.0
        best_enhanced_name = "Unknown"
        enhanced_scores = {}
        
        for person_name, person_data in self.base_system.known_faces.items():
            if 'individual_encodings' in person_data:
                person_scores = []
                
                for known_encoding in person_data['individual_encodings']:
                    features = self.extract_enhanced_features(unknown_encoding, known_encoding)
                    features_scaled = self.feature_scaler.transform([features])
                    similarity_prob = self.similarity_classifier.predict_proba(features_scaled)[0][1]
                    person_scores.append(similarity_prob)
                
                if person_scores:
                    max_score = max(person_scores)
                    enhanced_scores[person_name] = max_score
                    
                    if max_score > best_enhanced_confidence:
                        best_enhanced_confidence = max_score
                        best_enhanced_name = person_name
        
        # Combine base and enhanced predictions
        if base_name != "Unknown" and best_enhanced_name == base_name and best_enhanced_confidence > 0.7:
            combined_confidence = (base_confidence + best_enhanced_confidence) / 2
            return base_name, combined_confidence, "enhanced_agreement"
        
        elif base_name != "Unknown" and best_enhanced_confidence < 0.5:
            return base_name, base_confidence, "base_confident"
        
        elif base_name == "Unknown" and best_enhanced_confidence > 0.8:
            return best_enhanced_name, best_enhanced_confidence, "enhanced_rescue"
        
        else:
            return base_name, base_confidence, "base_default"
    
    def test_enhanced_system(self, test_folder):
        """Test the enhanced system"""
        results = {
            'base': {'correct': 0, 'total': 0, 'times': []},
            'enhanced': {'correct': 0, 'total': 0, 'times': []},
            'method_breakdown': {}
        }
        
        for person_folder in os.listdir(test_folder):
            person_path = os.path.join(test_folder, person_folder)
            if os.path.isdir(person_path):
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        actual_name = person_folder
                        
                        start_time = time.time()
                        base_pred, base_conf = self.base_system.recognize_face_fast(image_path)
                        base_time = (time.time() - start_time) * 1000
                        
                        start_time = time.time()
                        enhanced_pred, enhanced_conf, method = self.recognize_enhanced(image_path)
                        enhanced_time = (time.time() - start_time) * 1000
                        
                        results['base']['times'].append(base_time)
                        results['enhanced']['times'].append(enhanced_time)
                        results['base']['total'] += 1
                        results['enhanced']['total'] += 1
                        
                        base_correct = base_pred == actual_name
                        enhanced_correct = enhanced_pred == actual_name
                        
                        if base_correct:
                            results['base']['correct'] += 1
                        if enhanced_correct:
                            results['enhanced']['correct'] += 1
                        
                        if method not in results['method_breakdown']:
                            results['method_breakdown'][method] = {'total': 0, 'correct': 0}
                        results['method_breakdown'][method]['total'] += 1
                        if enhanced_correct:
                            results['method_breakdown'][method]['correct'] += 1
        
        return results
    
    def save_enhanced_model(self, filename='enhanced_face_model.pkl'):
        """Save the enhanced model"""
        model_data = {
            'base_system': {
                'known_faces': self.base_system.known_faces,
                'known_names': self.base_system.known_names,
                'known_encodings': self.base_system.known_encodings
            },
            'similarity_classifier': self.similarity_classifier,
            'feature_scaler': self.feature_scaler,
            'enhanced_features': self.enhanced_features,
            'similarity_trained': self.similarity_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

def run_enhanced_demo():
    """Run the enhanced face recognition demonstration"""
    enhanced_system = EnhancedFaceRecognition()
    
    training_folder = "training_images"
    if not os.path.exists(training_folder):
        return enhanced_system
    
    success = enhanced_system.train_enhanced_classifier(training_folder)
    
    if not success:
        enhanced_system.similarity_trained = False
    
    for person_name in os.listdir(training_folder):
        person_folder = os.path.join(training_folder, person_name)
        if os.path.isdir(person_folder):
            image_paths = []
            for img_file in os.listdir(person_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(person_folder, img_file))
            
            if image_paths:
                enhanced_system.add_person_enhanced(person_name, image_paths)
    
    test_folder = "test_images"
    if os.path.exists(test_folder):
        results = enhanced_system.test_enhanced_system(test_folder)
    
    enhanced_system.save_enhanced_model()
    
    return enhanced_system

if __name__ == "__main__":
    enhanced_system = run_enhanced_demo()