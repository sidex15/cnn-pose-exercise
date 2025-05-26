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

# Define the residual block for PyTorch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# Define the CNN model in PyTorch
class ExerciseCNN(nn.Module):
    def __init__(self, num_classes):
        super(ExerciseCNN, self).__init__()
        
        # Initial conv layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # First residual block
        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second residual block
        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Third residual block
        self.res_block3 = ResidualBlock(128, 128)
        
        # Calculate output size after convolution and pooling
        # Input: 33 timesteps
        # After pool1: 33/2 = 16 (rounded down)
        # After pool2: 16/2 = 8 (rounded down)
        # Final size: 128 channels * 8 features = 1024
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.res_block3(x)
        x = self.classifier(x)
        return x

# Temporal smoothing for predictions
class TemporalSmoother:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)
    def smooth(self, pred):
        self.history.append(pred)
        return np.mean(self.history, axis=0)

smoother = TemporalSmoother(window_size=3)

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

perf_monitor = PerformanceMonitor()

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

# PyTorch device setup
device = torch_directml.device()
print(f"Using device: {device}")

# Initialize video capture
# cap = video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Uncomment the line below to use a video file instead of webcam
cap = cv2.VideoCapture('pushups.mp4')

# Optional: Set OpenCV to use GPU for video decoding if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("OpenCV CUDA backend available")
    # No direct video decoding GPU API in Python OpenCV binding

# Load the trained CNN model using Keras
num_classes = 8  # Set this to your actual number of classes
model = ExerciseCNN(num_classes).to(device)
model.load_state_dict(torch.load('exercise_cnn_model_state.pth'))
model.eval()

# Load the scaler used during training
with open('cnn_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define class names based on the model's training
class_names = ["Unknown", "Situps Down", "Situps UP", "Pushups Down", "Pushups UP", "Squat Up", "Squat Down", "Jump Jack Up", "Jump Jack Down"]

# Batch prediction for better GPU utilization
def batch_predict(frames, batch_size=4):
    if len(frames) == 0:
        return []
    
    frames_np = np.array(frames, dtype=np.float32)
    frames_tensor = torch.from_numpy(frames_np).to(device)
    with torch.no_grad():
        outputs = model(frames_tensor)
        # If model outputs logits, apply softmax
        if outputs.shape[-1] == len(class_names) - 1 or outputs.shape[-1] == len(class_names):
            probs = torch.softmax(outputs, dim=-1)
        else:
            probs = outputs
        return probs.cpu().numpy()

# Create a buffer for frame processing
frame_buffer = []
prediction_buffer = []

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
        
        try:
            pred_start = time.time()
            # Extract pose landmarks
            row = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
            
            # Scale the input data using the same scaler used during training
            X_scaled = scaler.transform([row])
            
            # Reshape for CNN (samples, timesteps, features)
            X_reshaped = X_scaled.reshape(1, 33, 2)
            X_reshaped = X_reshaped.transpose(0, 2, 1)  # Now shape is (1, 2, 33)
            frame_buffer.append(X_reshaped[0])

            # Add to buffer for batch processing
            frame_buffer.append(X_reshaped[0])
            
            # Process in batches when buffer is full
            batch_size = 8
            if len(frame_buffer) >= 4:  # Adjust batch size based on your GPU memory
                batch_frames = np.array(frame_buffer)
                prediction_buffer = batch_predict(batch_frames, batch_size=batch_size)
                frame_buffer = []
            
            # Use the current prediction if available
            if len(prediction_buffer) > 0:
                y_pred_prob = prediction_buffer[0]
                prediction_buffer = prediction_buffer[1:]
            else:
                # Or make a single prediction if buffer is empty
                y_pred_prob = model.predict(X_reshaped, verbose=0)[0]

            pred_end = time.time()
            perf_monitor.update_prediction_time(pred_end - pred_start)

            class_idx = np.argmax(y_pred_prob)
            confidence = y_pred_prob[class_idx]
            
            # Only proceed if confidence is high enough
            if confidence >= 0.99:
                # Get the class name (+1 because class indices in training started from 1)
                class_id = class_idx + 1
                current_pos = class_names[class_id]
                
                # Count reps based on positions
                if class_id == 1:  # Situps Down
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Situps UP" and not count_reset:
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 4:  # Pushups UP
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Pushups Down":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 5:  # Squat Up
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Squat Down":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 8:  # Jump Jack Down
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Jump Jack Up":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
            
            reps_duration = round(reps_duration, 2)
            
            # Display information on the frame
            cv2.rectangle(image, (0,0), (250, 40), (245, 117, 16), -1)
            cv2.putText(image, current_pos, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Reps: {reps_counter}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Duration: {reps_duration}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {confidence:.2f}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            
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
        if cv2.waitKey(5) & 0xFF == 27:
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