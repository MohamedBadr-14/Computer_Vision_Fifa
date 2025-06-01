# predict_player_class.py
import cv2
import numpy as np
import pickle
import sys
from skimage.feature import hog
from classical_ml.players_contour import get_players_contours, decide_contours
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import time
import glob

class ClassicalML:
    def __init__(self):
        with open('./models/svm_hog_model.pkl', 'rb') as f:
            self.hog_svm = pickle.load(f)    

    def extract_hog_features(self,img):

        img = cv2.resize(img, (128, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray, cmap='gray')
        # plt.title("Gray Image")
        # plt.axis('off')
        # plt.show()
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
      # print(features,'midoooo')
        return features
    
    def run(self,frame):
        features= []
        player_tracks = {}
        id_counter = 0
        before_getting_contours = time.time()
        image_cropped_new, contours_pic,contours = get_players_contours(frame)
        after_getting_contours = time.time()
        # print("GETTING CONTOURS " , after_getting_contours - before_getting_contours)
        
        # cv2.imshow("Cropped Image", image_cropped_new)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print("Cropped Image", len(contours_pic))
        # print('contours',contours)
        # print('contours_pic',contours_pic)
        before_hog = time.time()
        if len(contours_pic) == 0:
            # print("No contours found in the image.")
            return player_tracks
        for ids,contour in enumerate(contours_pic):
       
            feature_vector = self.extract_hog_features(contour)
            if feature_vector is not None:
                    features.append(feature_vector)

        after_hog = time.time()
        # print("HOG TIME " , after_hog - before_hog)
        # print('CONTOURS PIC', len(contours_pic))
        features = np.array(features)
        before_prediction = time.time()
        # print(features,features.shape,'mido')
        prediction = self.hog_svm.predict(features) 
        after_prediction = time.time()
        # print("PREDICTION TIME " , after_prediction - before_prediction)
        # print('predictions',prediction)
        new_pic = frame.copy()
        
        for pred, contour, contour_pic in zip(prediction, contours, contours_pic):
        
            if pred == 1:  # Team 1 (Blue)
                
                x, y, w, h = cv2.boundingRect(contour)
                player_assign,contour_blue,contour_yellow = decide_contours(contour_pic,frame, x, y)
              # print(player_assign, 'player_assign')
                # print(player_assign, 'player_assign')               
                # cv2.imshow("Contour pic", contour_pic)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
          
                if player_assign == 1:
                    player_tracks[id_counter] = {"bbox": (x, y+162, x + w, y + h+162 ), "team": 1}
                    id_counter += 1
                    cv2.rectangle(new_pic, (x, y +162), (x + w, y + h +162), (0, 255, 0), 2)
                elif player_assign == 0:
                    player_tracks[id_counter] = {"bbox": (x, y+162 , x + w, y + h+162 ), "team": 0}
                    id_counter += 1
                    cv2.rectangle(new_pic, (x, y+162), (x + w, y + h+162), (0, 0, 255), 2)
                elif player_assign == 2:
                    x_initial, y_initial, w, h = cv2.boundingRect(contour)
                    if  type(contour_blue) == np.ndarray:
                        x, y, w, h = cv2.boundingRect(contour_blue)
                        x=x-10+x_initial
                        y=y-15+y_initial
                        # cv2.rectangle(new_pic, (x, y +162), (x + w+10, y + h+162+50), (0, 0, 255), 2)
                        player_tracks[id_counter] = {"bbox": (x-10, y+162, x + w+15, y + h+162+50), "team": 0}
                        id_counter += 1


                    if type(contour_yellow) == np.ndarray:    
                        x, y, w, h = cv2.boundingRect(contour_yellow)
                        x=x-10+x_initial
                        y=y-15+y_initial
                        # cv2.rectangle(new_pic, (x, y+162), (x + w+5, y + h+162+50), (0,255,0), 2)
                        player_tracks[id_counter] = {"bbox": (x-10, y+162, x + w+15 , y + h+162+50), "team": 1}
                        id_counter += 1
        # cv2.imshow("Detection", new_pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()     
        return player_tracks
    
    
def main(images_folder):

    # Initialize the model
    model = ClassicalML()

    # Get all image paths (jpg and png)
    image_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")) + glob.glob(os.path.join(images_folder, "*.png")))
    i=0
    for img_path in image_paths:
        # if i <100 :
        #     i+=1
        #     continue
        img = cv2.imread(img_path)
        if img is None:
          # print(f"Failed to read {img_path}")
            continue
        
        player_tracks = model.run(img)
      # print(f"Results for {img_path}:")
        # for pid, info in player_tracks.items():
          # print(f"  Player {pid}: bbox={info['bbox']}, team={info['team']}")

        # Optionally, show the image with detections
        # cv2.imshow("Detection mido", img)
        # cv2.waitKey(0)
        # break
        
       

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main("C:/Users/DELL/Downloads/wetransfer_frames_20250524_181705_2025-05-24_1521/frames_20250524_181705")