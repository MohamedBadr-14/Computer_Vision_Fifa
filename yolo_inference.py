from ultralytics import YOLO 
from PIL import Image, ImageDraw
import cv2
import os

model = YOLO('models/ball_e7na.pt')

def read_frames(frames_path):
        # Read the frames from the specified path
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                # frame = cv2.resize(frame, (640, 640))  
                frames.append(frame)
                if count == 60:
                    break
                count += 1
        return frames

# Perform prediction
frames = read_frames('./badr/train/images')
frames = frames[0:50]
results = model.predict(frames, save=True , conf=0.25)
