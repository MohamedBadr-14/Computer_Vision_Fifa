import cv2
import numpy as np
import pickle
import os
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

        # Initialize old_features in the constructor
        self.old_gray = first_frame_grayscale
        self.old_features = cv2.goodFeaturesToTrack(self.old_gray, **self.features)


    def get_camera_movement(self, current_frame, read_from_stub=False, stub_path=None):
        # Read the stub (if applicable)
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize camera movement for the current frame
        camera_movement = [0, 0]

        # Convert the current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


        # Calculate optical flow between the previous and current frame
        current_features, _, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, current_gray, self.old_features, None, **self.lk_params)

        # Estimate camera movement based on the optical flow
        max_distance = 0
        camera_movement_x, camera_movement_y = 0, 0

        for i, (new, old) in enumerate(zip(current_features, self.old_features)):
            new_point = new.ravel()
            old_point = old.ravel()

            distance = measure_distance(new_point, old_point)
            if distance > max_distance:
                max_distance = distance
                camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

        # If the movement exceeds the minimum distance, update the camera movement
        if max_distance > self.minimum_distance:
            camera_movement = [camera_movement_x, camera_movement_y]

        # Update the previous frame's grayscale and features for the next call
        self.old_gray = current_gray
        self.old_features = cv2.goodFeaturesToTrack(current_gray, **self.features)

        # Save to stub (if applicable)
        # if stub_path is not None:
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(camera_movement, f)

        return camera_movement
    
    def add_adjust_positions_to_tracks(self,frame_tracks, camera_movement_per_frame):

        for object_type , object_tracks in frame_tracks.items():
            for track_id,track_info in object_tracks.items():
                position = track_info['position']
                position_adjusted = (position[0]-camera_movement_per_frame[0],position[1]-camera_movement_per_frame[1])
                frame_tracks[object_type][track_id]['position_adjusted'] = position_adjusted

    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames