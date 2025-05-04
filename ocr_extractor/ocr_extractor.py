import cv2
import numpy as np
import re
from easyocr import Reader


class OCRExtractor:
    def __init__(self):
        self.reader = Reader(['en'], gpu=True,quantize=True) #BADR OUDA
        
    def enhance_image(self, img ):
            # Convert to grayscale if not already
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply thresholding
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Denoise
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Invert back for OCR
            binary = cv2.bitwise_not(binary)
            
            return binary
        
    def extract_points_and_attempts(self,image , save=False):    
        # Crop the relevant parts of the image
        height, width, _ = image.shape
        cropped_points = image[int(height*0.13):int(height*0.165), int(width*0.879):int(width*0.923)]
        cropped_attempts = image[int(height*0.175):int(height*0.2), int(width*0.79):int(width*0.805)]
        
        # if save:
        #     cv2.imwrite("cropped_points.png" , cropped_points)
        #     cv2.imwrite("cropped_attempts.png" , cropped_attempts)
        
        
        
        # Process points image
        enhanced_points = self.enhance_image(cropped_points)
        enhanced_points = cv2.resize(enhanced_points, (0, 0), fx=2, fy=2)
        result_points = self.reader.readtext(enhanced_points)
        
        # Process attempts image
        enhanced_attempts = self.enhance_image(cropped_attempts)
        enhanced_attempts = cv2.resize(enhanced_attempts, (0, 0), fx=2, fy=2)
        result_attempts = self.reader.readtext(enhanced_attempts)
        
        # Extract points
        points = None
        if len(result_points) > 0:
            points_text = result_points[0][1]
            points_numbers = re.findall(r'\d+', points_text)
            points = ''.join(points_numbers) if points_numbers else None
        
        # Extract attempts
        attempts = None
        if len(result_attempts) > 0:
            attempts_text = result_attempts[0][1]
            attempts_numbers = re.findall(r'\d+', attempts_text)
            attempts = ''.join(attempts_numbers) if attempts_numbers else None
        
        return {
            'points': points,
            'attempts': attempts,
            'raw_points': result_points[0][1] if len(result_points) > 0 else None,
            'raw_attempts': result_attempts[0][1] if len(result_attempts) > 0 else None
        }



# image = cv2.imread('input_videos/frame_0074.png')

# print("Image shape:", image.shape)
# ocr = OCRExtractor()
# # Call the function
# results = ocr.extract_points_and_attempts(image , True)

# # Access the results
# print(f"Points: {results['points']}")
# print(f"Attempts: {results['attempts']}")


