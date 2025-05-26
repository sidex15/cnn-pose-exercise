import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import csv
import keyboard
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = 'situps2.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print(f"Video: {os.path.basename(video_path)}")
print(f"FPS: {fps}, Total Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# Counter for frames
frame_number = 0

#array of variables containing file urls, labels etc.
vars = {
    "label": 2,
    "recordID": 0,
    "csvFile": "posedataset.csv",
    "mediaURL": ""
}

#Generate landmarks head row for CSV
def firstRow():
    landmarks = ['class']
    for val in range(1, 33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val)]
    print(landmarks[1:])

#Function for appending data in csv
def writeCSV(csvFile, list):
    try:
        with open(csvFile, mode="a", newline='') as new_file:
                write_content = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                write_content.writerow(list)
                print("Landmark Recorded Successfully.")
    except Exception as e:
        print(e)
        pass

def export_landmark(result, label, csvFilePath):
    try:
        keypoints = np.array([[res.x,res.y] for res in result.pose_landmarks.landmark]).flatten()
        keypoints = np.insert(keypoints, 0, label)
        print(keypoints)
        writeCSV(csvFilePath, keypoints)
    except Exception as e:
        print(e)
        pass
    
    

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (960, 540))
    if not success:
      print("End of video reached.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Increment frame counter and calculate timestamp
    frame_number += 1
    current_time = frame_number / fps
    
    image = cv2.resize(image, (960, 540))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Add timestamp to frame
    cv2.putText(image, f"Time: {current_time:.2f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Frame: {frame_number}/{frame_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=0, circle_radius=0),
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=1))
    else:
        continue  
    
    # If condition for triggering dataset capture
    if keyboard.is_pressed('r'):
        print("Landmarks Saved.")
        print(vars["label"])
        export_landmark(results, vars["label"], vars["csvFile"])
    
    # print("X:",round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, 4), "Y:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y, 4), "Z:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z, 4), "V:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility,4), "Ankle:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility, 4))
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()