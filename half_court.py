from utils import read_video, save_video
import cv2
import numpy as np


frames = read_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/test_soghayar.mp4')

output = []
for frame in frames:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    output.append(frame)