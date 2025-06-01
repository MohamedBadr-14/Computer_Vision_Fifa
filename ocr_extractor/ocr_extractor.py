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
        
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            binary = cv2.bitwise_not(binary)
            
            return binary
        
    def extract_points_and_attempts(self,image,num ,save=True):    
        height, width, _ = image.shape
        cropped_points = image[int(height*0.05):int(height*0.18), int(width*0.865):int(width*0.95)]
        cropped_attempts = image[int(height*0.13):int(height*0.17), int(width*0.73):int(width*0.8)]
        
        enhanced_points = self.enhance_image(cropped_points)
        enhanced_points = cv2.resize(enhanced_points, (0, 0), fx=2, fy=2)
        result_points = self.reader.readtext(cropped_points)\
            
        points = None
        if result_points:
            for result in result_points:
                text = result[1]
                if re.fullmatch(r'\d+', text):
                    points = text
                    break
        
        enhanced_attempts = self.enhance_image(cropped_attempts)
        enhanced_attempts = cv2.resize(enhanced_attempts, (0, 0), fx=2, fy=2)
        result_attempts = self.reader.readtext(cropped_attempts)
        
        attempts = None
        if result_attempts:
            for result in result_attempts:
                text = result[1]
                if re.fullmatch(r'\d+', text):
                    attempts = text
                    break

        
        return {
            'points': points,
            'attempts': attempts,
            'raw_points': result_points,
            'raw_attempts': result_attempts
        }