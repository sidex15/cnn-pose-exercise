import tkinter as tk 
from tkinter import *
from tkinter import ttk, filedialog
import os
import datetime

import cv2
import mediapipe as mp
import numpy as np
import csv

from PIL import Image, ImageTk 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.9, min_detection_confidence=0.9)

      
class datasetGUI:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("720x540")
        self.root.title("Dataset Recorder - Exercise Classification")
        
        self.csv_filepath = "newcnndataset.csv"
        self.dataset_label = ""
        self.cap = None
        self.is_paused = False
        self.results = None
        self.frame_number = 0  # Initialize frame_number
        self.fps = 0           # Initialize fps
        self.frame_count = 0   # Initialize frame_count
        
        self.frame = tk.Frame(height=380, width=720)
        self.frame.place(x=0, y=0)
        self.lmain = tk.Label(self.frame) 
        self.lmain.place(x=0, y=0)
        
        # Add timestamp label
        self.timestamp_label = tk.Label(self.root, text="", font=('Arial', 10), fg='blue')
        self.timestamp_label.place(x=550, y=390)
        
        self.utilityComponents()
        self.update_timestamp()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.update_idletasks()
        
        self.root.mainloop() 
    
    def update_timestamp(self):
        """Update timestamp display every second"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.timestamp_label.config(text=f"Time: {current_time}")
        self.root.after(1000, self.update_timestamp)
        
    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()
        
    def open_file(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv'), ('All files', '*.*')])
        if file:
            filepath = os.path.relpath(file.name)
            self.openVideo(filepath)
            
    def open_csv(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if file:
            filepath = os.path.relpath(file.name)
            self.csv_filepath = filepath
            Label(self.root, text="File selected : " + str(filepath), font=('Arial 11')).place(x=220, y=430)
            
    def record_landmarks(self, result, label, csvFilePath):
        print("Recording Landmarks...")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            if result and result.pose_landmarks:
                label = float(label)
                self.export_landmark(result, label, csvFilePath)
                print(f"Landmarks recorded with label: {label} at {timestamp}")
                # Show confirmation message with timestamp
                self.show_record_confirmation(timestamp)
            else:
                print("No pose landmarks detected")
        except ValueError:
            print("Invalid label - must be a number")
        except Exception as e:
            print(f"Error recording landmarks: {e}")
    
    def show_record_confirmation(self, timestamp):
        """Show confirmation message when data is recorded"""
        confirmation_label = Label(self.root, text=f"Data recorded at {timestamp}", 
                                 font=('Arial', 9), fg='green')
        confirmation_label.place(x=200, y=490)
        # Remove the confirmation message after 3 seconds
        self.root.after(3000, confirmation_label.destroy)
        
    def utilityComponents(self):
        ttk.Button(self.root, text="Browse", command=self.open_file).place(x=120, y=390)
        label = Label(self.root, text="Open Video File:", font=('Arial 11'))
        label.place(x=0, y=390)
        
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("Video paused")
        else:
            print("Video resumed")

    #Function for appending data in csv
    def writeCSV(self,csvFile, list):
        try:
            with open(csvFile, mode="a", newline='') as new_file:
                    write_content = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    write_content.writerow(list)
                    print("Landmark Recorded Successfully.")
        except Exception as e:
            print(e)
            pass

    def export_landmark(self,result, label, csvFilePath):
        try:
            keypoints = np.array([[res.x,res.y] for res in result.pose_landmarks.landmark]).flatten()
            keypoints = np.insert(keypoints, 0, label)
            print(keypoints)
            self.writeCSV(csvFilePath, keypoints)
        except Exception as e:
            print(e)
            pass
              
    def openVideo(self, vidpath):
        # Release previous capture if exists
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(vidpath)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0  # Reset frame_number for new video
        
        if self.fps == 0:
            print("Warning: Video FPS is 0. Timestamp calculation may be incorrect.")
            # Optionally, handle this case, e.g., by setting a default FPS or showing an error
            # self.fps = 30 # Example default if FPS is 0

        duration = self.frame_count / self.fps if self.fps > 0 else 0
        print(f"FPS: {self.fps}, Total Frames: {self.frame_count}")
        print(f"Duration: {duration:.2f} seconds")

        
        if not self.cap.isOpened():
            Label(self.root, text="Error: Could not open video file", font=('Arial 11'), fg='red').place(x=0, y=350)
            return
        
        label = Label(self.root, text="Choose CSV File:", font=('Arial 11'))
        label.place(x=0, y=430)
        
        ttk.Button(self.root, text="Pause/Resume", command=self.toggle_pause).place(x=330, y=470)
        
        Label(self.root, text="Row Label: ", font=('Arial 11')).place(x=0, y=470)
        self.dataset_label = tk.Entry(self.root, width=10)
        self.dataset_label.place(x=100, y=470)
        
        def detect():
            if not self.cap or not self.cap.isOpened():
                return
                
            if not self.is_paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    # Video ended or error reading frame
                    print("Video ended or error reading frame")
                    # self.cap.release() # Consider implications of releasing here
                    return

                # Increment frame counter and calculate timestamp
                self.frame_number += 1
                current_time = self.frame_number / self.fps if self.fps > 0 else 0   

                # Convert frame to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                image_rgb = cv2.resize(image_rgb, (720, 380))

                # Process with MediaPipe
                image_rgb.flags.writeable = False # MediaPipe prefers read-only
                self.results = pose.process(image_rgb)
                image_rgb.flags.writeable = True # Make writeable again for drawing

                # Convert to BGR for OpenCV drawing functions (putText, draw_landmarks)
                image_bgr_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Add timestamp to frame (on BGR image, color is BGR: Red)
                cv2.putText(image_bgr_draw, f"Time: {current_time:.2f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image_bgr_draw, f"Frame: {self.frame_number}/{self.frame_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if self.results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_bgr_draw, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=0, circle_radius=0), # BGR: Cyan
                        mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=1)) # BGR: Yellow
                
                # Convert final image (with drawings) from BGR to RGB for PIL/Tkinter
                final_image_rgb_display = cv2.cvtColor(image_bgr_draw, cv2.COLOR_BGR2RGB)
                
                imgarr = Image.fromarray(final_image_rgb_display) 
                imgtk = ImageTk.PhotoImage(imgarr) 
                self.lmain.imgtk = imgtk 
                self.lmain.configure(image=imgtk)
            
            # Continue the loop
            self.lmain.after(10, detect)
        
        ttk.Button(self.root, text="Record Data", command=lambda: self.record_landmarks(result=self.results, label=self.dataset_label.get(), csvFilePath=self.csv_filepath)).place(x=0, y=490)    
        
        ttk.Button(self.root, text="CSV Location", command=self.open_csv).place(x=130, y=430)
        
        detect()        
                
                
if __name__ == "__main__":
    datasetGUI()