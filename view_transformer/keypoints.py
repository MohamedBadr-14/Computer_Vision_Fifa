import cv2
import numpy as np
from ultralytics import YOLO
import torch
# Define the pixel coordinates of the 4 keypoints (from the image)

class KeypointsTransformer():
    
    def __init__(self):
        self.real_world_points = np.array([
        [49, 45],      # PEN_DOT
        [43.5,37.6875],    # ARC_UP
        [39.85, 45],       # ARC_CURVE
        [43.5,52.31]      # ARC_DOWN
        ], dtype=np.float32)
        self.keypoint_detector = YOLO('./models/keypoints.pt')
        self.keypoint_detector.to('cuda') 
        
        self.last_valid_keypoints = np.array([
        [1991, 1013],  #PEN_DOT #BADR 
        [1486, 799],  #ARC_UP
        [1223, 1017], #ARC_CURVE
        [1587, 1324] #ARC_DOWN
        ],dtype=np.float32)



    def detect_keypoints(self, image):
        results = self.keypoint_detector(image, conf=0.5, show=False,verbose=False)
        keypoints = []
        results = results[0].keypoints.xy[0]

        for result in results:
            if torch.all(result == 0):
                continue
            keypoints.append(result.cpu().numpy())
            

        if len(keypoints) < 4:            
            
            return self.last_valid_keypoints
        
        keypoints = np.array(keypoints, dtype=np.float32)
        self.last_valid_keypoints = keypoints
        keypoints = self.last_valid_keypoints

        return keypoints
    
    def get_matrix_transform(self, frame):
        keypoints = self.detect_keypoints(frame)
        # print("Keypoints detected in mt:", keypoints)
        
        
        pixel_points = np.array(keypoints, dtype=np.float32)
        real_world_points = np.array(self.real_world_points, dtype=np.float32)

        pixel_points = pixel_points.reshape(-1, 1, 2)
        real_world_points = real_world_points.reshape(-1, 1, 2)

        self.matrix = cv2.getPerspectiveTransform(pixel_points, real_world_points)
        return self.matrix


    def pixel_to_real_world(self,pixel_pos):
        px = np.array([[list(pixel_pos)]], dtype=np.float32)
        real_world = cv2.perspectiveTransform(px, self.matrix)
        return real_world[0][0]


    def add_transformed_position_to_tracks(self,frame_tracks):
        for object_type, object_tracks in frame_tracks.items():
            if object_type == "action":
                continue
            for track_id, track_info in object_tracks.items():
                position = track_info['position']
                position = np.array(position)
                position_trasnformed = self.pixel_to_real_world(position)
                if position_trasnformed is not None:
                    position_trasnformed = position_trasnformed.squeeze().tolist()
                frame_tracks[object_type][track_id]['position_transformed'] = position_trasnformed
    
    def real_world_to_pixel(self, real_world_pos):
        real_world_pos = np.array([[list(real_world_pos)]], dtype=np.float32)
        pixel_pos = cv2.perspectiveTransform(real_world_pos, np.linalg.inv(self.matrix))
        return pixel_pos[0][0]

