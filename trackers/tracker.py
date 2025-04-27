from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position,measure_distance


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.ball_positions_history = []
        self.last_valid_position = None
        self.frame_num = 0
        self.max_player_post_distance = 1000
        self.goal_position = (60,42)
        # self.previously_interpolated_ids = []

    def add_position_to_tracks(self, frame_tracks):
        for object_type, object_tracks in frame_tracks.items():
            for track_id, track_info in object_tracks.items():
                bbox = track_info["bbox"]
                if object_type == "ball":
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                frame_tracks[object_type][track_id]["position"] = position
    

    def adjust_goalkeeper_detection(self, frame_tracks):
        # if frame_tracks["goalpost"][1]["position"][0] < 2560:
            # print("DKHALT")
            
        post_position = frame_tracks["goalpost"][1]["position"]

        miniumum_distance = 99999
        goalkeeper=-1

        for player_id, player in frame_tracks["players"].items():
            player_pos = player['position']

            distance = measure_distance(player_pos,post_position)

            if distance < miniumum_distance:
                if distance < self.max_player_post_distance:
                    miniumum_distance = distance
                    goalkeeper = player_id
                    # print("FRAME_TRACKS" , frame_tracks["players"])
        if goalkeeper != -1:
            frame_tracks["goalkeeper"][goalkeeper] = {
                "bbox": frame_tracks["players"][goalkeeper]["bbox"],
                "conf": None, #BADRR
                "position": frame_tracks["players"][goalkeeper]["position"],
                "position_transformed": frame_tracks["players"][goalkeeper]["position_transformed"],
            }
            del frame_tracks["players"][goalkeeper] 
        return frame_tracks
    
    def remove_out_of_pitch_detections(self,player_tracks , height):
        to_be_deleted = []
        for player_id, player in player_tracks.items():
            bbox = player["bbox"]
            y = bbox[1]
            if y < 0.1*height:
                # Remove the player from the track if they are out of the pitch
                to_be_deleted.append(player_id)
        for player_id in to_be_deleted:
            del player_tracks[player_id]

    def interpolate_ball_positions(self, ball_position):
        self.ball_positions_history.append(ball_position.get(1, {}).get("bbox", []))

        if not self.ball_positions_history or all(
            len(bbox) == 0 for bbox in self.ball_positions_history
        ):
            return ball_position

        # Convert the ball positions history to a DataFrame
        df_ball_positions = pd.DataFrame(
            self.ball_positions_history, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = (
            df_ball_positions.bfill()
        )  # Backward fill for any remaining NaNs

        # Get the interpolated ball position for the current frame
        interpolated_bbox = df_ball_positions.iloc[-1].tolist()

        # Format the result
        # interpolated_ball_position = {1: {"bbox": interpolated_bbox}}
        self.last_valid_position = interpolated_bbox[0:2]
        return interpolated_bbox

    def detect_frame(self, frame):
        # Process a single frame
        detection = self.model.predict(frame) #BADR MIDO
        return detection

    def get_object_track(self, frame):
        # if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        #     with open(stub_path, 'rb') as f:
        #         tracks = pickle.load(f)
        #     return tracks

        # Detect objects in the current frame+
        detection = self.detect_frame(frame)
        ball_count = 0
        goal_count = 0

        # Initialize tracks for the current frame
        frame_tracks = {"players": {},"ball": {} , "goalkeeper": {} , "goalpost": {}}

        cls_names = detection[0].names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # Convert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection[0])
        detections_nms = detection_supervision.with_nms(threshold=0.5, class_agnostic=True)


        # Track objects in the current frame
        # detection_with_tracks = self.tracker.update_with_detections(
        #     detections_nms
        # )

            

        for i,frame_detection in enumerate(detections_nms):
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = i
            conf = frame_detection[2]

            # if cls_id == cls_names_inv["goalkeeper"]:
            #     print("SHAYEFFF GOALLL")
            

            if cls_id == cls_names_inv["player"] or cls_id == cls_names_inv["referee"] or cls_id == cls_names_inv["goalkeeper"]:
                frame_tracks["players"][track_id] = {"bbox": bbox, "conf": conf}

            # if cls_id == cls_names_inv["goalpost"]:
            #     if (
            #         goal_count > 0
            #         and frame_detection[2] < frame_tracks["goalpost"][1]["conf"]
            #     ):
            #         continue
            #     frame_tracks["goalpost"][1] = {"conf": frame_detection[2], "bbox": bbox}
            #     goal_count += 1
                
            if cls_id == cls_names_inv["ball"]:
                if (
                    ball_count > 0
                    and frame_detection[2] < frame_tracks["ball"][1]["conf"]
                ):
                    # If the ball is already detected, skip adding it again
                    # print("Ball already detected, skipping..." , frame_detection[2] , frame_tracks["ball"][1]["conf"])
                    continue
                frame_tracks["ball"][1] = {"conf": frame_detection[2], "bbox": bbox}
                # print("BALLLL CONFFF" , frame_detection[2])
                ball_count += 1

           
        return frame_tracks
    
                    
    def replace_outliers_based_on_distance(self, ball_track, distance_threshold: float):
        position = ball_track.get(1, {}).get("bbox", [])
        if not position:
            # If the current position is already empty, just add it to the cleaned positions
            # print("Empty ball")
            return {}
        else:
            if self.last_valid_position is None:
                # If there's no valid last position, accept the first valid one
                self.last_valid_position = position[0:2]
                return ball_track
            else:
                # Calculate the distance from the last valid position
                # #print("position", position)
                # #print("last_valid_position", last_valid_position)
                distance = np.linalg.norm(
                    np.array(position[0:2]) - np.array(self.last_valid_position)
                )
                # #print("distance", distance)
                if distance > distance_threshold:
                    # Replace with empty array if the distance exceeds the threshold
                    print("Replaced balll" , distance ,  " prev" , self.last_valid_position, "curr" , position[0:2])
                    return {}
                else:
                    self.last_valid_position = position[0:2]
                    return ball_track
        

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, center, color):
        # y = int(bbox[1])
        # x, _ = get_center_of_bbox(bbox)
        x,y = center[0], center[1]
        if x > 2581:
            return frame

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ],
            dtype=np.int32  # Ensure the points are integers

        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1*100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2*100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        # #print("len(video_frames)", len(video_frames))
        # print("len(tracks['players'])", len(tracks['players']))
        # print("len(tracks['ball'])", len(tracks['ball']))
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = (
                tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            )
            goal_keeper_dict = tracks["goalkeeper"][frame_num] if frame_num < len(tracks["goalkeeper"]) else {}
            goalpost_dict = tracks["goalpost"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                team = player.get("team")
                color = (0, 255, 0) if team == 1 else (255, 0, 0)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["position"], (0, 0, 255))
            
            for _, goalkeeper in goal_keeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 0, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["position"], (0, 255, 0))

            # Draw Goalpost
            for _, goalpost in goalpost_dict.items():
                frame = self.draw_traingle(frame, goalpost["position"], (255, 0, 0))

            # Draw Team Ball Control
            # frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    # def get_dropped_player_ids(self, frame_tracks, all_tracks,frame_num):
    #     dropped_player_ids = []
    #     if frame_num > 0:
    #         for track_id in frame_tracks["players"].keys():
    #             if track_id not in all_tracks["players"][frame_num-1].keys():
    #                 dropped_player_ids.append(track_id)
    #         return dropped_player_ids
    #     else:
    #         return []
        
    # def get_all_past_bbox(self, all_tracks, track_id):
    #     bboxes = []
    #     for frame_num, frame_tracks in all_tracks.items():
    #         if track_id in frame_tracks["players"].keys():
    #             bbox = frame_tracks["players"][track_id]["bbox"]
    #             bboxes.append(bbox)
    #     return bboxes
    
    # def interpolate_player_positions(self, frame_tracks, all_tracks,frame_num):
    #     dropped_ids = self.get_dropped_player_ids(frame_tracks, all_tracks,frame_num)
    #     if dropped_ids:
    #         for track_id in dropped_ids:
    #             bboxes = self.get_all_past_bbox(all_tracks, track_id)
    #             if len(bboxes) > 0:
    #                 # Interpolate the missing positions using the past positions
    #                 df_bboxes = pd.DataFrame(bboxes, columns=["x1", "y1", "x2", "y2"])
    #                 df_bboxes = df_bboxes.interpolate()
    #                 df_bboxes = df_bboxes.bfill()
    #                 interpolated_bbox = df_bboxes.iloc[-1].tolist()
    #                 frame_tracks["players"][track_id] = {
    #                     "bbox": interpolated_bbox,
    #                     "conf": 0.1,
    #                 }