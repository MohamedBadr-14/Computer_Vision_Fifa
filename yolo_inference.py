from ultralytics import YOLO 
import os
import cv2

model = YOLO('models/best_new.pt')

def read_frames(frames_path):
    # Read the frames from the specified path
    frames = []
    count = 0
    for filename in sorted(os.listdir(frames_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frame_path = os.path.join(frames_path, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
            if count == 10:
                break
            count += 1
    return frames


# frames = read_frames('./badr')
results = model.predict('./badr/frame_0008.png',save=True)
#print(results[0])
#print('=====================================')
for box in results[0].boxes:
    print(box)
