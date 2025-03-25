from utils import read_video, save_video
import cv2
import numpy as np

video_frames = read_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/fouda.mp4')
# np.save('fouda.npy', video_frames)
# video_frames = np.load('fouda.npy', allow_pickle=True)

out_frames = []
# for frame in out_frames:
gray_frame = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# output_frame = video_frames[0].copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(x2 - x1) < 10: 
            cv2.line(video_frames[0], (x1, y1), (x2, y2), (120, 120, 0), 2)

# save_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/output_videos/fouda.mp4', video_frames)
cv2.imwrite('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/output_videos/line_test.png', video_frames[0])



