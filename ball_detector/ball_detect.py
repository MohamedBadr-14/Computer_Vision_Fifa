import cv2
import os
import numpy as np
import pickle
from skimage.feature import hog



class BallDetect:
    def __init__(self):
        # Load saved models
        # with open('./models/svm_model.pkl', 'rb') as f:
        #     self.svm = pickle.load(f)
        # with open('./models/vocabulary.pkl', 'rb') as f:
        #     self.vocabulary = pickle.load(f)
        # with open('./models/kmeans_model.pkl', 'rb') as f:
        #     self.kmeans = pickle.load(f)
        # self.num_clusters = self.vocabulary.shape[0]
        # self.sift = cv2.SIFT_create()
        with open('./models/ball_hog_model.pkl', 'rb') as f:
            self.hog_model = pickle.load(f)
        self.last_ball_position = None        
    
    def extract_hog_features(self,img):
        
        img = cv2.resize(img, (128, 256))  # Resize to fixed size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        return features
    
    
    def mask_on_white(self,img):
        cropped_img = img.copy()
        height, width = img.shape[:2]

        # Crop 20% of the height from the top
        crop_height = int(height * 0.22)
        # crop_width = int(width * 0.1)
        cropped_img[0:crop_height] = 0 # Crop from top and left
        

        lower_bound = np.array([0,  0, 120])   # H=0–179, S=0–X, V≥150
        upper_bound = np.array([179, 80, 255]) 

        # Convert input image to HSV
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Create the mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        kernel_open =  np.array([[1, 0],
                   [0, 1]], dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # kernel_close = np.array([[1, 0],
        #            [0, 1]], dtype=np.uint8)
        #get structuring element
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
        
        y_min = 0  # Top of the image (farthest from camera)
        y_max = frame.shape[0]  # Bottom of the image (closest to camera)
        area_min_at_y_min = 10  # Smallest area when the ball is farthest
        area_max_at_y_min = 90
        area_min_at_y_max = 60  # Largest area when the ball is closest
        area_max_at_y_max = 400
        
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

                
            if area_min < area < area_max:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                
                if circularity > 0.45:  # adjust based on your needs
                    # extract_crops([coor_x, coor_y, coor_x + w, coor_y + h], frame, cont_batch , count)
                    bbox = [coor_x, coor_y, coor_x + w, coor_y + h]
                    extrated_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    features = self.extract_hog_features(extrated_img)
                    prediction = self.hog_model.predict(features.reshape(1, -1))[0] if features is not None else 0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    label = f"A:{area:.2f},C:{circularity:.2f},P:{prediction},MIN:{area_min:.2f},MAX:{area_max:.2f}"
                    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
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
        # Read the frames from the specified path
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

#main
# if __name__ == "__main__":
    
#     # Create an instance of the BallDetect class
#     ball_detector = BallDetect()
#     # Load a test image
#     frames = read_frames('../screenshots')

#     counter = 0
#     for i, frame in enumerate(frames):
#         # Detect the ball in the frame
#         bboxes, count = ball_detector.svm_predict(frame,i) 
#         counter += count if count else 0


#     print("Total number of balls detected:", counter)

        

 
    # # Function to extract SIFT descriptors
    # def extract_sift_descriptors(self,img):
    #     keypoints, descriptors = self.sift.detectAndCompute(img, None)
    #     return descriptors

    # # Function to convert descriptors to BoVW histogram
    # def compute_bovw_feature(self,descriptors):
    #     words = self.kmeans.predict(descriptors)
    #     histogram = np.zeros(self.num_clusters)
    #     for word in words:
    #         histogram[word] += 1
    #     histogram = histogram / np.sum(histogram)
    #     return histogram

    
    # def extract_sift_descriptors(self,img):
    #     keypoints, descriptors = self.sift.detectAndCompute(img, None)
    #     return descriptors


    # def compute_bovw_feature(self,descriptors):
    #     words = self.kmeans.predict(descriptors)
    #     histogram = np.zeros(self.num_clusters)
    #     for word in words:
    #         histogram[word] += 1
    #     histogram = histogram / np.sum(histogram)
        
    #     return histogram
    
    # def svm_features(self,cropped_img):
    #     descriptors = self.extract_sift_descriptors(cropped_img)
    #     if descriptors is not None:
    #         bovw_feature = self.compute_bovw_feature(descriptors)
    #         bovw_feature = bovw_feature.reshape(1, -1)
    #         return bovw_feature
    #     return None
    
    # def adjust_svm_input(self , frame , bbox):

    #     svm_input = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

    #     # Desired size
    #     target_size = (128, 128)

    #     # Resize the cropped image
    #     resized_input = cv2.resize(svm_input, target_size, interpolation=cv2.INTER_CUBIC)

        
    #     return resized_input
