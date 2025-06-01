import easyocr
import cv2
import numpy as np
import time
import threading
import keyboard

class OffsideDetector:
    def __init__(self, gpu=False):
        self.reader = easyocr.Reader(['en'], gpu=gpu, quantize=True)
        
    def detect_offside(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)    
        preprocessed = self.preprocess_image(frame)

        result = self.reader.readtext(
            thresh, 
            detail=0,
            paragraph=False,
            # allowlist='OFSIDEofside',
            batch_size=1
        )
        
        print(f"Detected text: {result}")
        # keyboard.wait('l')
        # print("ANA HENAAAAAAAAA")
        for text in result:
            text_cleaned = text.replace(" ", "").replace("0", "O")
            text_lower = text_cleaned.lower()
            if ('offside' in text_lower or
                'off'in text_lower or
                'side' in text_lower or
                'ofside' in text_lower or
                'offsice' in text_lower or
                'offcide' in text_lower or
                'ofcide' in text_lower or
                'offcide' in text_lower or
                'offcice' in text_lower 
                ):
                # print("Offside detected!")
                return True
            # if 'offside' in text.lower():
            #     print("Offside detected!")
            #     return True
                
        return False
