import cv2
import os
import numpy as np
import pickle
import cupy as cp  # For GPU acceleration
from numba import jit, cuda  # For GPU acceleration
import torch  # For GPU operations
from sklearn.preprocessing import normalize


class BallDetect:
    def __init__(self, use_gpu=True):
        # Load saved models
        with open('./models/svm_model.pkl', 'rb') as f:
            self.svm = pickle.load(f)
        with open('./models/vocabulary.pkl', 'rb') as f:
            self.vocabulary = pickle.load(f)
        with open('./models/kmeans_model.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)
            
        self.num_clusters = self.vocabulary.shape[0]
        self.sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, 
                                   contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        self.last_ball_position = None
        self.use_gpu = use_gpu
        
        # Initialize CUDA device if available and requested
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.torch_enabled = True
                print("GPU acceleration enabled using PyTorch")
            else:
                self.torch_enabled = False
                print("GPU requested but not available, falling back to CPU")
                self.use_gpu = False
        
    # Function to extract SIFT descriptors
    def extract_sift_descriptors(self, img):
        # SIFT extraction is usually CPU-bound as OpenCV doesn't have GPU implementation
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return descriptors

    # Function to convert descriptors to BoVW histogram with GPU acceleration
    def compute_bovw_feature(self, descriptors):
        if descriptors is None:
            return None
            
        if self.use_gpu and self.torch_enabled:
            # Move data to GPU
            descriptors_gpu = torch.tensor(descriptors, device=self.device, dtype=torch.float32)
            centroids_gpu = torch.tensor(self.kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
            
            # Compute distances to all centroids
            distances = torch.cdist(descriptors_gpu, centroids_gpu)
            
            # Find nearest centroid for each descriptor
            words = torch.argmin(distances, dim=1).cpu().numpy()
            
            # Create histogram
            histogram = np.zeros(self.num_clusters)
            for word in words:
                histogram[word] += 1
                
            # Normalize
            if np.sum(histogram) > 0:
                histogram = histogram / np.sum(histogram)
                
            return histogram
        else:
            # CPU fallback
            words = self.kmeans.predict(descriptors)
            histogram = np.zeros(self.num_clusters)
            for word in words:
                histogram[word] += 1
            
            if np.sum(histogram) > 0:
                histogram = histogram / np.sum(histogram)
                
            return histogram
    
    def svm_features(self, cropped_img):
        descriptors = self.extract_sift_descriptors(cropped_img)
        if descriptors is not None:
            bovw_feature = self.compute_bovw_feature(descriptors)
            bovw_feature = bovw_feature.reshape(1, -1)
            return bovw_feature
        return None
    
    def adjust_svm_input(self, frame, bbox):
        svm_input = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Desired size
        target_size = (128, 128)

        # Resize the cropped image
        resized_input = cv2.resize(svm_input, target_size, interpolation=cv2.INTER_CUBIC)
        
        return resized_input
    
    @jit(nopython=True, target="cuda")
    def _process_mask(self, hsv_array, lower_bound, upper_bound):
        """GPU-accelerated mask processing using Numba CUDA"""
        height, width, _ = hsv_array.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                h, s, v = hsv_array[y, x]
                if (lower_bound[0] <= h <= upper_bound[0] and 
                    lower_bound[1] <= s <= upper_bound[1] and 
                    lower_bound[2] <= v <= upper_bound[2]):
                    mask[y, x] = 255
                    
        return mask
    
    def mask_on_white(self, img):
        height, width = img.shape[:2]

        # Crop 20% of the height from the top
        crop_height = int(height * 0.2)
        cropped_img = img.copy()
        cropped_img[0:crop_height] = 0  # Crop from top

        # Your target BGR color
        bgr_color = np.uint8([[[181, 187, 173]]])  # BGR format
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

        # Define a tolerance range
        lower_bound = np.array([0, 0, 200])  # Lower bound for HSV
        upper_bound = np.array([179, 50, 255])  # Upper bound for HSV

        # Convert input image to HSV
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        
        if self.use_gpu and self.torch_enabled:
            # GPU-accelerated mask creation
            hsv_tensor = torch.from_numpy(hsv).to(self.device)
            lower_bound_tensor = torch.from_numpy(lower_bound).to(self.device)
            upper_bound_tensor = torch.from_numpy(upper_bound).to(self.device)
            
            # Create mask using GPU tensor operations
            mask_tensor = torch.logical_and(
                torch.logical_and(
                    hsv_tensor[:,:,0] >= lower_bound_tensor[0],
                    hsv_tensor[:,:,0] <= upper_bound_tensor[0]
                ),
                torch.logical_and(
                    torch.logical_and(
                        hsv_tensor[:,:,1] >= lower_bound_tensor[1],
                        hsv_tensor[:,:,1] <= upper_bound_tensor[1]
                    ),
                    torch.logical_and(
                        hsv_tensor[:,:,2] >= lower_bound_tensor[2],
                        hsv_tensor[:,:,2] <= upper_bound_tensor[2]
                    )
                )
            ).type(torch.uint8) * 255
            
            mask = mask_tensor.cpu().numpy()
        else:
            # CPU fallback
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        kernel = np.ones((2, 2), np.uint8)
        closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed_image
    
    def find_ball(self, mask, frame):
        bounding_boxes = []
        
        y_min = 0  # Top of the image (farthest from camera)
        y_max = frame.shape[0]  # Bottom of the image (closest to camera)
        area_min_at_y_min = 20  # Smallest area when the ball is farthest
        area_max_at_y_min = 120
        area_min_at_y_max = 70  # Largest area when the ball is closest
        area_max_at_y_max = 600
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare batch for faster processing
        candidates = []
        
        for cnt in contours:
            coor_x, coor_y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Linearly interpolate the area bounds based on y
            y_fraction = max(0, min(1, (coor_y - y_min) / (y_max - y_min)))
            
            # Interpolate area bounds
            area_min = area_min_at_y_min + y_fraction * (area_min_at_y_max - area_min_at_y_min)
            area_max = area_max_at_y_min + y_fraction * (area_max_at_y_max - area_max_at_y_min)

            if area_min < area < area_max:
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                
                if circularity > 0.45:  # adjust based on your needs
                    bbox = [coor_x, coor_y, coor_x + w, coor_y + h]
                    candidates.append((bbox, frame[coor_y:coor_y+h, coor_x:coor_x+w]))
        
        # Batch process candidates if possible
        if self.use_gpu and self.torch_enabled and len(candidates) > 0:
            batch_features = []
            batch_bboxes = []
            
            for bbox, crop in candidates:
                svm_input = self.adjust_svm_input(frame, bbox)
                features = self.svm_features(svm_input)
                if features is not None:
                    batch_features.append(features)
                    batch_bboxes.append(bbox)
            
            if batch_features:
                batch_features = np.vstack(batch_features)
                predictions = self.svm.predict(batch_features)
                
                for i, (bbox, prediction) in enumerate(zip(batch_bboxes, predictions)):
                    if prediction == 1:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        bounding_boxes.append(bbox)
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        else:
            # Process one by one (CPU fallback)
            for bbox, crop in candidates:
                svm_input = self.adjust_svm_input(frame, bbox)
                features = self.svm_features(svm_input)
                if features is not None:
                    prediction = self.svm.predict(features)[0]
                    if prediction == 1:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        bounding_boxes.append(bbox)
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        return bounding_boxes

    def svm_predict(self, frame):
        mask = self.mask_on_white(frame)
        bboxes = self.find_ball(mask, frame)
        if len(bboxes) > 0:
            if self.last_ball_position is None:
                self.last_ball_position = self.get_center_of_bbox(bboxes[0])
                return bboxes[0]
            
            closest_bbox = self.get_closest_bbox(bboxes)
            self.last_ball_position = self.get_center_of_bbox(closest_bbox)
            return closest_bbox
        return None
    
    def get_center_of_bbox(self, bbox):
        if bbox is None:
            return None
        if len(bbox) != 4:
            print(f"Invalid bbox format: {bbox}")
            return None
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def measure_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def get_closest_bbox(self, bboxes):
        if len(bboxes) == 0:
            return None
        min_distance = float('inf')
        closest_bbox = None
        for bbox in bboxes:
            current_center = self.get_center_of_bbox(bbox)
            distance = self.measure_distance(current_center, self.last_ball_position)
            if distance < min_distance:
                min_distance = distance
                closest_bbox = bbox
        return closest_bbox