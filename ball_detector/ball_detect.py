import cv2
import os
import numpy as np
import pickle


class BallDetect:
    def __init__(self):
        # Load saved models
        with open('./models/svm_model.pkl', 'rb') as f:
            self.svm = pickle.load(f)
        with open('./models/vocabulary.pkl', 'rb') as f:
            self.vocabulary = pickle.load(f)
        with open('./models/kmeans_model.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)
        self.num_clusters = self.vocabulary.shape[0]
        self.sift = cv2.SIFT_create()
        self.last_ball_position = None        
        
    # Function to extract SIFT descriptors
    def extract_sift_descriptors(self,img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return descriptors

    # Function to convert descriptors to BoVW histogram
    def compute_bovw_feature(self,descriptors):
        words = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.num_clusters)
        for word in words:
            histogram[word] += 1
        histogram = histogram / np.sum(histogram)
        return histogram

    
    def extract_sift_descriptors(self,img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return descriptors


    def compute_bovw_feature(self,descriptors):
        words = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.num_clusters)
        for word in words:
            histogram[word] += 1
        histogram = histogram / np.sum(histogram)
        
        return histogram
    
    def svm_features(self,cropped_img):
        descriptors = self.extract_sift_descriptors(cropped_img)
        if descriptors is not None:
            bovw_feature = self.compute_bovw_feature(descriptors)
            bovw_feature = bovw_feature.reshape(1, -1)
            return bovw_feature
        return None
    
    def adjust_svm_input(self , frame , bbox):

        svm_input = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Desired size
        target_size = (128, 128)

        # Resize the cropped image
        resized_input = cv2.resize(svm_input, target_size, interpolation=cv2.INTER_CUBIC)

        
        return resized_input
    
    
    def mask_on_white(self,img):
        height, width = img.shape[:2]

        # Crop 20% of the height from the top
        crop_height = int(height * 0.2)
        # crop_width = int(width * 0.1)
        cropped_img = img.copy()
        cropped_img[0:crop_height] = 0 # Crop from top and left


    # Your target BGR color
        bgr_color = np.uint8([[[181, 187, 173]]])  # BGR format
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        # print("HSV color:", hsv_color)  # Just to check

        # Define a tolerance range
        h, s, v = hsv_color
        lower_bound = np.array([0,0, 200])  # Lower bound for hue, saturation, and value
        upper_bound = np.array([179, 50, 255])  # Upper bound for hue, saturation, and value

        # Convert input image to HSV
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Create the mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        kernel = np.ones((3, 3), np.uint8)  # Adjust size (e.g., 3x3 or 7x7) based on stripe width
        closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # plt.imshow(mask, cmap='gray')
        return closed_image
    
    def find_ball(self,mask,frame):
        # test_frame = frame.copy()
        # print("TEST FRAME SHAPE:", frame.shape) 
        # found = False
        # count = 0
        bounding_boxes = []
        
        y_min = 0  # Top of the image (farthest from camera)
        y_max = frame.shape[0]  # Bottom of the image (closest to camera)
        area_min_at_y_min = 20  # Smallest area when the ball is farthest
        area_max_at_y_min = 120
        area_min_at_y_max = 70  # Largest area when the ball is closest
        area_max_at_y_max = 600
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            coor_x, coor_y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Linearly interpolate the area bounds based on y
            # Fraction of y between y_min and y_max
            y_fraction = (coor_y - y_min) / (y_max - y_min)
            
            # Interpolate area bounds
            area_min = area_min_at_y_min + y_fraction * (area_min_at_y_max - area_min_at_y_min)
            area_max = area_max_at_y_min + y_fraction * (area_max_at_y_max - area_max_at_y_min)

            # # Adjust the condition dynamically
            # print("\nArea:", area , "Area_min:", area_min, "Area_max:", area_max )
            # print("coor_x:", coor_x , "coor_y:", coor_y )
            if area_min < area < area_max:
                # (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                
                if circularity > 0.45:  # adjust based on your needs
                    # extract_crops([coor_x, coor_y, coor_x + w, coor_y + h], frame, cont_batch , count)
                    bbox = [coor_x, coor_y, coor_x + w, coor_y + h]
                    svm_input = self.adjust_svm_input(frame, bbox)
                    features = self.svm_features(svm_input)
                    prediction = self.svm.predict(features)[0] if features is not None else 0
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 1
                    # thickness = 2
                    # label = f"A:{area:.2f},C:{circularity:.2f},P:{prediction}"
                    # text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                    # text_x = int(x) + int(radius) + 5  # Position text to the right of the circle
                    # text_y = int(y) + int(text_size[1] // 2)  # Center text vertically with circle
                    # cv2.putText(test_frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                    # cv2.imwrite(f'badr/output_koora/koora_{x}_{y}.jpg', test_frame)
                    if prediction == 1:
                        # cv2.rectangle(frame, (coor_x, coor_y), (coor_x + w, coor_y + h), (255, 0, 0), 2)
                        bounding_boxes.append([coor_x, coor_y, coor_x + w, coor_y + h])
                    # else:
                        # cv2.rectangle(frame, (coor_x, coor_y), (coor_x + w, coor_y + h), (0, 0, 255), 2)
                       
                    # if features is not None and not np.isnan(features).any():
                    #     probs = self.svm.predict_proba(np.array(features).reshape(1, -1))[0]
                    #     confidence = max(probs)
                    # else:
                    #     print("FEATURESS " ,(features is not None))
                    #     print("NAN " , (features is not None and not np.isnan(features).any()))

                    #     confidence = 0.0
        return bounding_boxes

    
    def svm_predict(self, frame ):
        mask = self.mask_on_white(frame)
        bboxes = self.find_ball(mask,frame)
        if len(bboxes) > 0:
            if self.last_ball_position is None:
                return bboxes[0]
            return self.get_closest_bbox(bboxes)
        return None
    
    def get_center_of_bbox(self,bbox):
        if bbox is None:
            return None #OUDA
        if len(bbox) != 4:
            print(f"Invalid bbox format: {bbox}")
            return None #OUDA
        x1,y1,x2,y2 = bbox
        return int((x1+x2)/2),int((y1+y2)/2)

    def measure_distance(self,p1,p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    
    def get_closest_bbox(self, bboxes):
        if len(bboxes) == 0:
            return None
        min_distance = float('inf')
        closest_bbox = None
        for bbox in bboxes:
            distance = self.measure_distance(self.get_center_of_bbox(bbox), self.last_ball_position)
            if distance < min_distance:
                min_distance = distance
                closest_bbox = bbox
        return closest_bbox