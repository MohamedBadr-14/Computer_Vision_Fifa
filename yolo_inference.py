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
            if count == 100:
                break
            count += 1
    return frames


frames = read_frames('./badr/all')
results = model.predict(frames[94:])

#draw yolo results on the image
for i, frame in enumerate(frames[94:]):
    #print(results[i])
    #results[i].plot()
    #cv2.imshow('YOLO Results', results[i].plot())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(f'./badr/yolo_results/frame_{i}.jpg')
    cv2.imwrite(f'./badr/yolo_results/frame_{i}.jpg', results[i].plot())

#print(results[0])
#print('=====================================')
for box in results[0].boxes:
    print(box)
