from ultralytics import YOLO 
from PIL import Image, ImageDraw
import cv2
import os

model = YOLO('models/keypoints.pt')

def read_frames(frames_path):
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, (640, 640))  
                frames.append(frame)
                if count == 60:
                    break
                count += 1
        return frames

frames = read_frames('./frames')
results = model(frames , save=True )
