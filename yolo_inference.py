from ultralytics import YOLO 
from PIL import Image, ImageDraw
import cv2

model = YOLO('models/best_new.pt')

# Perform prediction
results = model.predict('frame_buffer_000175.jpg', save=True)

# Load the image
image_path = 'frame_buffer_000175.jpg'
image = cv2.imread(image_path)

# Draw the boxes on the image
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract and convert box coordinates to integers
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw rectangle with red color (BGR)
    
    # Calculate the center coordinates of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Write the center coordinates on the image
    text = f"({center_x}, {center_y})"
    cv2.putText(image, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Write coordinates in white

# Save or display the image with boxes
image_with_boxes_path = 'image_after_inf_3.jpg'
cv2.imwrite(image_with_boxes_path, image)
print(f"Image with boxes saved to {image_with_boxes_path}")
