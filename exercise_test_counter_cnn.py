import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import time
import torch
import torch_directml
import torch.nn as nn
import collections

# Configuration - Set this to choose model type
USE_CNN = True # Set to False to use KNN
MODEL_PATH_CNN = 'best_exercise_cnn_model.pth'
MODEL_PATH_KNN = 'knn_exercise_model.pkl'
SCALER_PATH = 'cnn_scaler.pkl'  # Same scaler for both models
# Initialize video capture
cap = video_capture = cv2.VideoCapture(4, cv2.CAP_DSHOW)
# Uncomment the line below to use a video file instead of webcam
#cap = cv2.VideoCapture('pushup9.mp4')

# Initialize model manager
device = torch_directml.device() if USE_CNN else None
print(f"Using device: {device}")

# Define class names based on the model's training
class_names = ["Unknown", "Situps Down", "Situps UP", "Pushups Down", "Pushups UP", "Squat Up", "Squat Down", "Jump Jack Up", "Jump Jack Down"]

# Define exercise modes
# Add exercise mode selection
class ExerciseMode:
    SITUPS = 1
    PUSHUPS = 2
    SQUATS = 3
    JUMPING_JACKS = 4

# Global exercise mode variable
current_exercise_mode = ExerciseMode.PUSHUPS

# Exercise-specific class mappings
exercise_class_mappings = {
    ExerciseMode.SITUPS: {
        'down': 1,  # Situps Down
        'up': 2     # Situps Up
    },
    ExerciseMode.PUSHUPS: {
        'down': 3,  # Pushups Down
        'up': 4     # Pushups Up
    },
    ExerciseMode.SQUATS: {
        'up': 5,    # Squat Up
        'down': 6   # Squat Down
    },
    ExerciseMode.JUMPING_JACKS: {
        'up': 7,    # Jump Jack Up
        'down': 8   # Jump Jack Down
    }
}

class ModelManager:
    def __init__(self, use_cnn=True, device=None):
        self.use_cnn = use_cnn
        self.device = device
        self.model = None
        self.scaler = None
        self.model_type = "CNN" if use_cnn else "KNN"
        
        self.load_model()
        self.load_scaler()
    
    def load_model(self):
        """Load either CNN or KNN model"""
        try:
            if self.use_cnn:
                # Load CNN model
                num_classes = 8
                self.model = ExerciseCNN(num_classes).to(self.device)
                
                # Handle different model save formats
                loaded_model = torch.load(MODEL_PATH_CNN, weights_only=False)
                if isinstance(loaded_model, ExerciseCNN):
                    self.model.load_state_dict(loaded_model.state_dict())
                elif isinstance(loaded_model, dict):
                    self.model.load_state_dict(loaded_model)
                else:
                    raise ValueError(f"Unexpected model format: {type(loaded_model)}")
                
                self.model.eval()
                print("✅ CNN model loaded successfully")
            else:
                # Load KNN model
                with open(MODEL_PATH_KNN, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ KNN model loaded successfully")
                
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            print("Available models:")
            print(f"  CNN: {MODEL_PATH_CNN}")
            print(f"  KNN: {MODEL_PATH_KNN}")
            exit(1)
        except Exception as e:
            print(f"Error loading {self.model_type} model: {e}")
            exit(1)
    
    def load_scaler(self):
        """Load the data scaler"""
        try:
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")
        except FileNotFoundError:
            print(f"Scaler file not found: {SCALER_PATH}")
            exit(1)
    
    def predict_single(self, landmarks):
        """Make prediction for single frame"""
        if self.use_cnn:
            return self._predict_cnn_single(landmarks)
        else:
            return self._predict_knn_single(landmarks)
    
    def predict_batch(self, frames, batch_size=4):
        """Make batch predictions"""
        if self.use_cnn:
            return self._predict_cnn_batch(frames, batch_size)
        else:
            return self._predict_knn_batch(frames)
    
    def _predict_cnn_single(self, landmarks):
        """CNN single prediction"""
        # Scale the input
        X_scaled = self.scaler.transform([landmarks])
        
        # Reshape for CNN: (1, 33, 2) -> (1, 2, 33)
        X_reshaped = X_scaled.reshape(1, 33, 2).transpose(0, 2, 1)
        
        # Convert to tensor and predict
        frames_tensor = torch.from_numpy(X_reshaped.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            y_pred_prob = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
        
        return y_pred_prob
    
    def _predict_knn_single(self, landmarks):
        """KNN single prediction - improved"""
        try:
            # Scale the input (KNN uses flattened data)
            X_scaled = self.scaler.transform([landmarks])
            
            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            
            # Check if model has predict_proba method
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
            else:
                # Fallback for models without probability prediction
                probabilities = np.zeros(len(self.model.classes_))
                pred_idx = np.where(self.model.classes_ == prediction)[0]
                if len(pred_idx) > 0:
                    probabilities[pred_idx[0]] = 1.0
            
            return probabilities
            
        except Exception as e:
            print(f"KNN prediction error: {e}")
            # Return uniform probabilities as fallback
            return np.ones(len(self.model.classes_)) / len(self.model.classes_)
    
    def _predict_cnn_batch(self, frames, batch_size=4):
        """CNN batch prediction"""
        if len(frames) == 0:
            return []
        
        frames_np = np.array(frames, dtype=np.float32)
        frames_tensor = torch.from_numpy(frames_np).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = torch.softmax(outputs, dim=-1)
            return probs.cpu().numpy()
    
    def _predict_knn_batch(self, frames):
        """KNN batch prediction (processes one by one)"""
        results = []
        for frame in frames:
            # Flatten the frame for KNN
            frame_flat = frame.reshape(-1)  # (2, 33) -> (66,)
            
            # Transform using scaler
            frame_scaled = self.scaler.transform([frame_flat])
            
            # Get probabilities
            probabilities = self.model.predict_proba(frame_scaled)[0]
            results.append(probabilities)
        
        return np.array(results)

# Define the CNN architecture
class ExerciseCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=2):
        super(ExerciseCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 33 -> 16
            nn.Dropout(0.1),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 16 -> 8
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Temporal smoothing for predictions
class TemporalSmoother:
    def __init__(self, window_size=5, confidence_threshold=0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = []
        
    def add_prediction(self, prediction, confidence):
        self.prediction_history.append((prediction, confidence))
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
    
    def get_stable_prediction(self):
        if len(self.prediction_history) < 3:
            return None, 0.0
            
        # Get predictions with sufficient confidence
        valid_predictions = [
            (pred, conf) for pred, conf in self.prediction_history 
            if conf >= self.confidence_threshold
        ]
        
        if not valid_predictions:
            return None, 0.0
            
        # Count occurrences of each prediction
        from collections import Counter
        pred_counts = Counter([pred for pred, _ in valid_predictions])
        
        # Return most common prediction if it appears frequently enough
        most_common = pred_counts.most_common(1)[0]
        if most_common[1] >= max(2, len(valid_predictions) * 0.6):
            avg_confidence = sum([conf for pred, conf in valid_predictions if pred == most_common[0]]) / most_common[1]
            return most_common[0], avg_confidence
            
        return None, 0.0

# Initialize smoother
temporal_smoother = TemporalSmoother(window_size=10, confidence_threshold=0.75)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.prediction_times = collections.deque(maxlen=window_size)
        self.frame_times = collections.deque(maxlen=window_size)
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps_history = []
        self.fps = 0


    def update_prediction_time(self, duration):
        self.prediction_times.append(duration)

    def update_frame_time(self, duration):
        self.frame_times.append(duration)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.fps_history.append(self.fps)
            if len(self.fps_history) > self.window_size:
                self.fps_history.pop(0)
            self.frame_count = 0
            self.last_fps_time = now

    def get_avg_prediction_time(self):
        if not self.prediction_times:
            return 0
        return sum(self.prediction_times) / len(self.prediction_times)
    
    def get_avg_frame_time(self):
        if not self.frame_times:
            return 0
        return sum(self.frame_times) / len(self.frame_times)
    
    def get_avg_fps(self):
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_fps(self):
        return self.fps

# Initialize performance monitor
perf_monitor = PerformanceMonitor()
# Initialize model manager
model_manager = ModelManager(use_cnn=USE_CNN, device=device)
print(f"Model type: {model_manager.model_type}")

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

reps_counter=0
reps_duration=0
current_pos=''
prev_pos=''
pTime = time.time()
cTime = 0
count_reset = True

def predict_exercise(landmarks):
    """Unified prediction function for both models"""
    try:
        return model_manager.predict_single(landmarks)
    except Exception as e:
        print(f"Prediction error: {e}")
        return np.zeros(8)  # Return zero probabilities on error

def batch_predict(frames, batch_size=4):
    """Unified batch prediction function"""
    try:
        return model_manager.predict_batch(frames, batch_size)
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return []

# Create numeric to string class mapping for KNN
def create_knn_class_mapping():
    """Create mapping from KNN numeric classes to string names"""
    if not model_manager.use_cnn:
        # Your KNN model uses numeric classes: 1.0, 2.0, 3.0, etc.
        # Map them to the corresponding class names
        knn_classes = sorted(model_manager.model.classes_)  # [1.0, 2.0, 3.0, ...]
        
        # Create mapping: KNN index -> class name index
        knn_to_display_mapping = {}
        
        for knn_idx, knn_class in enumerate(knn_classes):
            # Convert float to int and use as index in class_names
            class_int = int(knn_class)
            if 1 <= class_int <= len(class_names) - 1:  # Skip "Unknown" at index 0
                knn_to_display_mapping[knn_idx] = class_int
        
        print("KNN Model Classes (numeric):", knn_classes)
        print("KNN to Display Mapping:", knn_to_display_mapping)
        print("Mapped class names:", [class_names[idx] for idx in knn_to_display_mapping.values()])
        
        return knn_to_display_mapping
    return None

# Replace the debug_model_classes function with this new one
knn_mapping = create_knn_class_mapping()

# Updated get_exercise_specific_prediction function
def get_exercise_specific_prediction(y_pred_prob, exercise_mode):
    """Get prediction filtered for specific exercise - Fixed KNN mapping"""
    if model_manager.use_cnn:
        # Original CNN logic
        exercise_classes = list(exercise_class_mappings[exercise_mode].values())
        relevant_probs = []
        relevant_indices = []
        
        for class_id in exercise_classes:
            if class_id - 1 < len(y_pred_prob):
                relevant_probs.append(y_pred_prob[class_id - 1])
                relevant_indices.append(class_id - 1)
        
        if not relevant_probs:
            return None, 0.0
        
        best_idx = np.argmax(relevant_probs)
        best_class_idx = relevant_indices[best_idx]
        best_confidence = relevant_probs[best_idx]
        
        return best_class_idx, best_confidence
    
    else:
        # KNN logic with proper numeric mapping
        if knn_mapping is None or len(knn_mapping) == 0:
            print("Warning: No KNN mapping available, using direct indexing")
            # Fallback: assume KNN predictions map directly to class indices
            best_idx = np.argmax(y_pred_prob)
            best_confidence = y_pred_prob[best_idx]
            # Try to map KNN index to class index
            if best_idx < len(model_manager.model.classes_):
                knn_class = model_manager.model.classes_[best_idx]
                class_idx = int(knn_class) - 1  # Convert 1.0->0, 2.0->1, etc.
                return class_idx, best_confidence
            return best_idx, best_confidence
        
        # Use the proper mapping
        mapped_probs = np.zeros(len(class_names) - 1)  # -1 for "Unknown"
        
        for knn_idx, prob in enumerate(y_pred_prob):
            if knn_idx in knn_mapping:
                class_id = knn_mapping[knn_idx]  # This is the class ID (1-8)
                display_idx = class_id - 1  # Convert to array index (0-7)
                if 0 <= display_idx < len(mapped_probs):
                    mapped_probs[display_idx] = prob
        
        # Now apply exercise filtering
        exercise_classes = list(exercise_class_mappings[exercise_mode].values())
        relevant_probs = []
        relevant_indices = []
        
        for class_id in exercise_classes:
            if class_id - 1 < len(mapped_probs):
                relevant_probs.append(mapped_probs[class_id - 1])
                relevant_indices.append(class_id - 1)
        
        if not relevant_probs:
            print(f"No relevant probabilities for exercise mode {exercise_mode}")
            print(f"Exercise classes: {exercise_classes}")
            print(f"Mapped probs shape: {len(mapped_probs)}")
            print(f"Available probs: {np.nonzero(mapped_probs)[0]}")
            return None, 0.0
        
        best_idx = np.argmax(relevant_probs)
        best_class_idx = relevant_indices[best_idx]
        best_confidence = relevant_probs[best_idx]
        
        return best_class_idx, best_confidence

# Add debug output to see what's happening
def debug_prediction(y_pred_prob, exercise_mode):
    """Debug function to understand prediction issues"""
    if not model_manager.use_cnn:
        print(f"\n--- Debug Prediction for Exercise Mode {exercise_mode} ---")
        print(f"Raw KNN probabilities: {y_pred_prob}")
        print(f"KNN classes: {model_manager.model.classes_}")
        print(f"KNN mapping: {knn_mapping}")
        
        best_knn_idx = np.argmax(y_pred_prob)
        best_knn_class = model_manager.model.classes_[best_knn_idx]
        print(f"Best KNN prediction: class {best_knn_class} (index {best_knn_idx}) with prob {y_pred_prob[best_knn_idx]:.3f}")
        
        if knn_mapping and best_knn_idx in knn_mapping:
            mapped_class_id = knn_mapping[best_knn_idx]
            class_name = class_names[mapped_class_id] if mapped_class_id < len(class_names) else "Unknown"
            print(f"Mapped to: class_id {mapped_class_id} = '{class_name}'")
        
        exercise_classes = list(exercise_class_mappings[exercise_mode].values())
        print(f"Expected exercise classes: {exercise_classes}")
        print("--- End Debug ---\n")

# Uncomment this line in your main loop for debugging:
# debug_prediction(y_pred_prob, current_exercise_mode)

# Create a buffer for frame processing
frame_buffer = []
prediction_buffer = []
stable_position = None
position_start_time = time.time()
min_position_duration = 0.3  # Minimum time to hold position (300ms)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        frame_start = time.time()
        success, image = cap.read()
        if not success:
            print("End of video.")
            break
            
        image = cv2.resize(image, (960, 540))
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks:
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            continue
            
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,215,14), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,1,18), thickness=2, circle_radius=1))
        
        # Modified prediction section in your main loop
        try:
            pred_start = time.time()
            # Extract pose landmarks
            row = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
            
            if USE_CNN:
                # CNN batch processing (your existing logic)
                X_scaled = model_manager.scaler.transform([row])
                X_reshaped = X_scaled.reshape(1, 33, 2)
                X_reshaped = X_reshaped.transpose(0, 2, 1)
                frame_buffer.append(X_reshaped[0])
                
                if len(frame_buffer) >= 4:
                    batch_frames = np.array(frame_buffer)
                    prediction_buffer = batch_predict(batch_frames, batch_size=4)
                    frame_buffer = []
                
                if len(prediction_buffer) > 0:
                    y_pred_prob = prediction_buffer[0]
                    prediction_buffer = prediction_buffer[1:]
                else:
                    y_pred_prob = predict_exercise(row)
            else:
                # KNN direct prediction (simpler)
                y_pred_prob = predict_exercise(row)

            pred_end = time.time()
            perf_monitor.update_prediction_time(pred_end - pred_start)

            # Filter prediction for current exercise mode
            class_idx, confidence = get_exercise_specific_prediction(y_pred_prob, current_exercise_mode)
            
            if class_idx is not None:
                # Add to temporal smoother
                temporal_smoother.add_prediction(class_idx, confidence)
                stable_pred, stable_conf = temporal_smoother.get_stable_prediction()
                
                # Only proceed if we have a stable prediction
                if stable_pred is not None and stable_conf >= 0.75:
                    class_id = stable_pred + 1
                    new_position = class_names[class_id]
                    
                    # Simplified rep counting for current exercise mode
                    current_time = time.time()
                    if new_position != stable_position:
                        if current_time - position_start_time >= min_position_duration:
                            stable_position = new_position
                            position_start_time = current_time
                            
                            # Exercise-specific rep counting
                            if current_exercise_mode == ExerciseMode.SITUPS:
                                if class_id == 1 and prev_pos == "Situps UP":  # Down position
                                    reps_counter += 1
                                    cTime = time.time()
                                    reps_duration = cTime - pTime if 'pTime' in locals() else 0
                                    pTime = cTime
                            
                            elif current_exercise_mode == ExerciseMode.PUSHUPS:
                                if class_id == 4 and prev_pos == "Pushups Down":  # Up position
                                    reps_counter += 1
                                    cTime = time.time()
                                    reps_duration = cTime - pTime if 'pTime' in locals() else 0
                                    pTime = cTime
                            
                            elif current_exercise_mode == ExerciseMode.SQUATS:
                                if class_id == 5 and prev_pos == "Squat Down":  # Up position
                                    reps_counter += 1
                                    cTime = time.time()
                                    reps_duration = cTime - pTime if 'pTime' in locals() else 0
                                    pTime = cTime
                            
                            elif current_exercise_mode == ExerciseMode.JUMPING_JACKS:
                                if class_id == 8 and prev_pos == "Jump Jack Up":  # Down position
                                    reps_counter += 1
                                    cTime = time.time()
                                    reps_duration = cTime - pTime if 'pTime' in locals() else 0
                                    pTime = cTime
                            
                            prev_pos = stable_position
                    
                        current_pos = stable_position if stable_position else "Unknown"
                    else:
                        current_pos = stable_position if stable_position else "Unknown"
    
                # Display current exercise mode
                exercise_names = {
                    ExerciseMode.SITUPS: "Sit-ups",
                    ExerciseMode.PUSHUPS: "Push-ups", 
                    ExerciseMode.SQUATS: "Squats",
                    ExerciseMode.JUMPING_JACKS: "Jumping Jacks"
                }
            
            reps_duration = round(reps_duration, 2)

            # Display information on the frame
            cv2.rectangle(image, (0,0), (250, 40), (245, 117, 16), -1)
            cv2.putText(image, current_pos, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Reps: {reps_counter}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Duration: {reps_duration}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {confidence:.2f}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(image, f"Model: {model_manager.model_type}", 
                       (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(image, f"Exercise: {exercise_names[current_exercise_mode]}", 
                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # --- Performance metrics ---
            frame_end = time.time()
            perf_monitor.update_frame_time(frame_end - frame_start)
            avg_pred_ms = perf_monitor.get_avg_prediction_time() * 1000
            avg_frame_ms = perf_monitor.get_avg_frame_time() * 1000
            fps = perf_monitor.get_fps()

            cv2.putText(image, f"FPS: {fps:.1f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Pred: {avg_pred_ms:.1f}ms", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, f"Frame: {avg_frame_ms:.1f}ms", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            prev_pos = current_pos
            
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        cv2.imshow('MediaPipe Pose', image)
        # Add keyboard controls for exercise switching
        key = cv2.waitKey(5) & 0xFF
        if key == ord('c'):  # Switch to CNN
            USE_CNN = True
            model_manager = ModelManager(use_cnn=True, device=device)
            frame_buffer = []  # Clear buffer
            prediction_buffer = []
        elif key == ord('k'):  # Switch to KNN
            USE_CNN = False
            model_manager = ModelManager(use_cnn=False, device=None)
            frame_buffer = []  # Clear buffer
            prediction_buffer = []
        elif key == ord('1'):
            current_exercise_mode = ExerciseMode.SITUPS
            reps_counter = 0# Reset counter when switching
        elif key == ord('2'):
            current_exercise_mode = ExerciseMode.PUSHUPS
            reps_counter = 0
        elif key == ord('3'):
            current_exercise_mode = ExerciseMode.SQUATS
            reps_counter = 0
        elif key == ord('4'):
            current_exercise_mode = ExerciseMode.JUMPING_JACKS
            reps_counter = 0
        elif key == 27:  # ESC key
            break

# Process any remaining frames in the buffer
if frame_buffer and len(frame_buffer) > 0:
    batch_predict(np.array(frame_buffer))

# Print performance summary
print("\n--- PERFORMANCE SUMMARY ---")
print(f"Average FPS: {perf_monitor.get_avg_fps():.2f}")
print(f"Average prediction time: {perf_monitor.get_avg_prediction_time()*1000:.2f} ms")
print(f"Average frame processing time: {perf_monitor.get_avg_frame_time()*1000:.2f} ms")
print(f"Total reps counted: {reps_counter}")

cap.release()
cv2.destroyAllWindows()