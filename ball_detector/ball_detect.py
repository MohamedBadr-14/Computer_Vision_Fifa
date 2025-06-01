import cv2
import os
import numpy as np
import pickle
from skimage.feature import hog



class BallDetect:
    def __init__(self):
        with open('./models/ball_hog_model.pkl', 'rb') as f:
            self.hog_model = pickle.load(f)
        self.last_ball_position = None        
    
    def extract_hog_features(self,img):
        img = cv2.resize(img, (128, 256))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = hog(img_gray, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        return features
    
    
    def mask_on_white(self,img):
        
        cropped = img.copy()
        height, width = img.shape[:2]

        crop_height = int(height * 0.22)
        # crop_width = int(width * 0.1)
        cropped[0:crop_height] = 0
        

        lower_bound = np.array([0,  0, 120])
        upper_bound = np.array([179, 80, 255]) 

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        kernel_open =  np.array([[1, 0],[0, 1]], dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # kernel_close = np.array([[1, 0],[0, 1]], dtype=np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_image = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        

        # plt.imshow(mask, cmap='gray')
        return closed_image
    
    
    def find_ball(self , mask , frame , cont_batch):
        test_frame = frame.copy()
        # print("TEST FRAME SHAPE:", frame.shape) 
        # found = False
        count = 0
        bounding_boxes = []
        
        # y el min foo2 el max taht 
        y_min = 0 
        y_max = frame.shape[0]  
        area_min_at_y_min = 10
        area_max_at_y_min = 90
        area_min_at_y_max = 60 
        area_max_at_y_max = 400
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            coor_x, coor_y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            
            normalized = (coor_y - y_min) / (y_max - y_min)
            area_min = area_min_at_y_min + normalized * (area_min_at_y_max - area_min_at_y_min)
            area_max = area_max_at_y_min + normalized * (area_max_at_y_max - area_max_at_y_min)

                
            if area_min < area < area_max:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                
                if circularity > 0.45:
                    # extract_crops([coor_x, coor_y, coor_x + w, coor_y + h], frame, cont_batch , count)
                    bbox = [coor_x, coor_y, coor_x + w, coor_y + h]
                    extrated_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    features = self.extract_hog_features(extrated_img)
                    prediction = self.hog_model.predict(features.reshape(1, -1))[0] if features is not None else 0
                    txt = f"A:{area:.2f},C:{circularity:.2f},P:{prediction},MIN:{area_min:.2f},MAX:{area_max:.2f}"
                    text_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = int(x) + int(radius) + 5  # Position text to the right of the circle
                    text_y = int(y) + int(text_size[1] // 2)  # Center text vertically with circle
                    if prediction == 1:
                        count += 1
                        # cv2.rectangle(test_frame, (coor_x, coor_y), (coor_x + w, coor_y + h), (255, 0, 0), 2)
                        bounding_boxes.append([coor_x, coor_y, coor_x + w, coor_y + h])
                    # else:
                    #     cv2.rectangle(test_frame, (coor_x, coor_y), (coor_x + w, coor_y + h), (0, 0, 255), 2)
                       
                    # cv2.putText(test_frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                    # cv2.imwrite(f'../output_koora/a_{cont_batch}_out.jpg', test_frame)
                    # if features is not None and not np.isnan(features).any():
                    #     probs = self.svm.predict_proba(np.array(features).reshape(1, -1))[0]
                    #     confidence = max(probs)
                    # else:
                    #     print("FEATURESS " ,(features is not None))
                    #     print("NAN " , (features is not None and not np.isnan(features).any()))
                    #     confidence = 0.0
        return bounding_boxes 

    
    def svm_predict(self, frame ,i):
        mask = self.mask_on_white(frame)
        bboxes = self.find_ball(mask,frame,i)
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


def read_frames(frames_path):
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                # frame = cv2.resize(frame, (640, 640))  
                frames.append(frame)
                count += 1
                # if count == 100:
                #     break
                
        return frames
