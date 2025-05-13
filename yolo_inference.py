from ultralytics import YOLO 
from PIL import Image, ImageDraw
import cv2
import os

model = YOLO('models/ball_best.pt')

def read_frames(frames_path):
        # Read the frames from the specified path
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                frames.append(frame)
                if count == 50:
                    break
                count += 1
        return frames

# Perform prediction
frames = read_frames('./badr/all')
results = model.predict(frames, save=True , conf = 0.1)

# Load the image
# image_path = 'frame_buffer
# _000175.jpg'
# image = cv2.imread(image_path)

# Draw the boxes on the image
# for box in results[0].boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract and convert box coordinates to integers
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw rectangle with red color (BGR)
    
#     # Calculate the center coordinates of the bounding box
#     center_x = (x1 + x2) // 2
#     center_y = (y1 + y2) // 2
    
#     # Write the center coordinates on the image
#     text = f"({center_x}, {center_y})"
#     cv2.putText(image, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Write coordinates in white

# Save or display the image with boxes
# image_with_boxes_path = 'image_after_inf_3.jpg'
# cv2.imwrite(image_with_boxes_path, image)
# print(f"Image with boxes saved to {image_with_boxes_path}")
