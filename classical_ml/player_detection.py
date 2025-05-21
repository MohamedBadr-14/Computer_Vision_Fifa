# predict_player_class.py
import cv2
import numpy as np
import pickle
import sys
from skimage.feature import hog
from .players_contour import get_players_contours,read_frames,decide_contours
import matplotlib.pyplot as plt
import os

class ClassicalML:
    def __init__(self):
        with open('./models/svm_hog_model.pkl', 'rb') as f:
            self.hog_svm = pickle.load(f)    



    def extract_hog_features(self,img):

        img = cv2.resize(img, (128, 256))  # Resize to fixed size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        return features
    
    def run(self,frame):
        
        features= []
        player_tracks = {}
        id_counter = 0
        image_cropped_new, contours_pic,contours = get_players_contours(frame)
        for ids,contour in enumerate(contours_pic):
       
            feature_vector = self.extract_hog_features(contour)
         
            if feature_vector is not None:
                    features.append(feature_vector)

        features = np.array(features)

        prediction = self.hog_svm.predict(features)
        for pred, contour, contour_pic in zip(prediction, contours, contours_pic):
        
            if pred == 1:
                
                x, y, w, h = cv2.boundingRect(contour)
        
                player_assign,contour_blue,contour_yellow = decide_contours(contour_pic)
          
                if player_assign == 1:
                    player_tracks[id_counter] = {"bbox": (x, y, x + w, y + h), "team": 1}
                    id_counter += 1
                    cv2.rectangle(image_cropped_new, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif player_assign == 0:
                    player_tracks[id_counter] = {"bbox": (x, y, x + w, y + h), "team": 0}
                    id_counter += 1
                    cv2.rectangle(image_cropped_new, (x, y), (x + w, y + h), (0, 0, 255), 2)
                elif player_assign == 2:
                    x_initial, y_initial, w, h = cv2.boundingRect(contour)
                    if  type(contour_blue) == np.ndarray:
                        x, y, w, h = cv2.boundingRect(contour_blue)
                        x=x-7+x_initial
                        y=y-15+y_initial
                        cv2.rectangle(image_cropped_new, (x, y), (x + w+10, y + h+70), (0, 0, 255), 2)
                        player_tracks[id_counter] = {"bbox": (x, y, x + w + 10, y + h + 70), "team": 0}
                        id_counter += 1


                    if type(contour_yellow) == np.ndarray:    
                        x, y, w, h = cv2.boundingRect(contour_yellow)
                        x=x-7+x_initial
                        y=y-15+y_initial
                        cv2.rectangle(image_cropped_new, (x, y), (x + w+10, y + h+70), (0,255,0), 2)
                        player_tracks[id_counter] = {"bbox": (x, y, x + w + 10, y + h+70), "team": 1}
                        id_counter += 1
                
        return player_tracks
        




        

  
                
            

