import cv2
import numpy as np
import re
from easyocr import Reader
from PIL import Image
import re

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
        
    def extract_points_and_attempts(self,image,num ,save=True):    
        # Crop the relevant parts of the image
        height, width, _ = image.shape
        cropped_points = image[int(height*0.05):int(height*0.18), int(width*0.865):int(width*0.95)]
        cropped_attempts = image[int(height*0.13):int(height*0.17), int(width*0.73):int(width*0.8)]
        # cropped_points = image[int(height*0.089):int(height*0.13), int(width*0.879):int(width*0.923)]
        # cropped_attempts = image[int(height*0.14):int(height*0.17), int(width*0.78):int(width*0.8)]
        
        if save:
            cv2.imwrite("cropped_points_{}.png".format(num) , cropped_points)
            cv2.imwrite("cropped_attempts_{}.png".format(num) , cropped_attempts)
            # cv2.imwrite("cropped_points_{}.png" , cropped_points)
            # cv2.imwrite("cropped_attempts_{}.png" , cropped_attempts)
        
        
        
        # Process points image
        enhanced_points = self.enhance_image(cropped_points)
        enhanced_points = cv2.resize(enhanced_points, (0, 0), fx=2, fy=2)
        result_points = self.reader.readtext(cropped_points)\
            
        points = None
        if result_points:
            for result in result_points:
                text = result[1]
                if re.fullmatch(r'\d+', text):  # Check if the text contains only digits
                    points = text
                    break
        
        # # Process attempts image
        enhanced_attempts = self.enhance_image(cropped_attempts)
        enhanced_attempts = cv2.resize(enhanced_attempts, (0, 0), fx=2, fy=2)
        result_attempts = self.reader.readtext(cropped_attempts)
        
        attempts = None
        if result_attempts:
            for result in result_attempts:
                text = result[1]
                if re.fullmatch(r'\d+', text):  # Check if the text contains only digits
                    attempts = text
                    break
            
        
        # result_points = pytesseract.image_to_string(enhanced_points, config=self.custom_config)
        # result_attempts = pytesseract.image_to_string(enhanced_attempts, config=self.custom_config)

        # # Extract numbers
        # points = re.findall(r'\d+', result_points)
        # attempts = re.findall(r'\d+', result_attempts)
        # if points:
        #     points = ''.join(points)
        # else:
        #     points = None
        # if attempts:
        #     attempts = ''.join(attempts)
        # else:
        #     attempts = None
        # Extract points
        # points = None
        # if len(result_points) > 0:
        #     points_text = result_points[0][1]
        #     points_numbers = re.findall(r'\d+', points_text)
        #     points = ''.join(points_numbers) if points_numbers else None
        
        # # Extract attempts
        # attempts = None
        # if len(result_attempts) > 0:
        #     attempts_text = result_attempts[0][1]
        #     attempts_numbers = re.findall(r'\d+', attempts_text)
        #     attempts = ''.join(attempts_numbers) if attempts_numbers else None
        
        return {
            'points': points,
            'attempts': attempts,
            'raw_points': result_points,
            'raw_attempts': result_attempts
        }



# image = cv2.imread('input_videos/frame_0074.png')

# print("Image shape:", image.shape)
# ocr = OCRExtractor()
# # Call the function
# results = ocr.extract_points_and_attempts(image , True)

# # Access the results
# print(f"Points: {results['points']}")
# print(f"Attempts: {results['attempts']}")


