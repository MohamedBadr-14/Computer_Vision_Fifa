import easyocr
import cv2
import numpy as np
import time
import threading

class OffsideDetector:
    """
    A class to detect offside text in FIFA game frames using EasyOCR.
    Optimized for speed and minimal resource usage.
    """
    def __init__(self, gpu=False):
        """
        Initialize the OCR reader in a separate thread to avoid blocking.
        
        Args:
            gpu (bool): Whether to use GPU acceleration
        """
        self.reader = None
        self.reader_ready = False
        # self.offside_region = [50, 100, 250, 200]  # [x1, y1, x2, y2] region where offside text appears
        self.last_detection_time = 0
        self.cooldown_period = 2.0  # seconds between detections
        self.initialize_reader(gpu)
    
    def initialize_reader(self, gpu):
        """Initialize the EasyOCR reader in a background thread"""
        def init_thread():
            self.reader = easyocr.Reader(['en'], gpu=gpu, quantize=True)
            self.reader_ready = True
            print("OCR reader initialized and ready")
            
        # Start initialization in background
        threading.Thread(target=init_thread).start()
    
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy and speed.
        - Extract only the region of interest
        - Convert to grayscale
        - Apply thresholding to make text clearer
        """
        # Extract the region where offside text appears
        # x1, y1, x2, y2 = self.offside_region
        # roi = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to make text stand out
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        return thresh
    
    def detect_offside(self, frame):
        """
        Check if offside text is present in the frame.
        
        Args:
            frame: The input frame to check
            
        Returns:
            bool: True if offside is detected, False otherwise
        """
        # Skip if reader isn't ready or if we've checked recently
        current_time = time.time()
        if not self.reader_ready or (current_time - self.last_detection_time < self.cooldown_period):
            return False
            
        # Preprocess the image
        preprocessed = self.preprocess_image(frame)
        # preprocessed = frame.copy()
        
        # Update last detection time
        self.last_detection_time = current_time
        
        try:
            # Only look for the word "Offside" with limited detection
            result = self.reader.readtext(
                preprocessed, 
                detail=0,
                paragraph=False,
                allowlist='OFSIDEofside',
                batch_size=1
            )
            print(f"Detected text: {result}")
            print("ANA HENAAAAAAAAA")
            # Check if "Offside" is in the results (case insensitive)
            for text in result:
                if 'offside' in text.lower():
                    print("Offside detected!")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return False
            
    def warmup(self):
        """
        Force the model to load and process a dummy image to warm up.
        Call this method once after initialization to reduce first-detection delays.
        """
        if not self.reader_ready:
            print("Reader not ready yet, waiting...")
            while not self.reader_ready:
                time.sleep(0.1)
                
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        _ = self.reader.readtext(dummy, detail=0, batch_size=1)
        print("OCR warm-up complete")

# Usage example
if __name__ == "__main__":
    detector = OffsideDetector(gpu=True)
    detector.warmup()  # Optional: pre-warm the model
    
    # Test on sample image
    image = cv2.imread("off2.png")
    if image is not None:
        start_time = time.time()
        result = detector.detect_offside(image)
        elapsed = time.time() - start_time
        print(f"Detection result: {result} (took {elapsed:.3f} seconds)")
    else:
        print("Failed to load test image")
