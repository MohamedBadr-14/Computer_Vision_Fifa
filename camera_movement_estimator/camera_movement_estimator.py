import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append("../")
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
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
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read from stub if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        # Convert first frame to grayscale
        if frames[0] is None:
            raise ValueError("First frame is empty or None.")
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # Detect initial features
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        if old_features is None or len(old_features) == 0:
            # print("No features found in the first frame.")
            return camera_movement

        # Ensure correct format
        old_features = np.array(old_features, dtype=np.float32).reshape(-1, 1, 2)

        for frame_num in range(1, len(frames)):
            if frames[frame_num] is None:
                # print(f"Skipping frame {frame_num} as it is empty.")
                continue

            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Compute optical flow
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or status is None:
                # print(f"Optical flow failed at frame {frame_num}.")
                continue

            # Filter only good points
            good_old = old_features[status.flatten() == 1]
            good_new = new_features[status.flatten() == 1]

            if len(good_old) == 0 or len(good_new) == 0:
                # print(f"No good features to track at frame {frame_num}.")
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old in zip(good_new, good_old):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                new_detected_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features
                )
                if new_detected_features is not None:
                    old_features = np.array(
                        new_detected_features, dtype=np.float32
                    ).reshape(-1, 1, 2)

            old_gray = frame_gray.copy()

        # Save to stub if required
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(frame)

        return output_frames
