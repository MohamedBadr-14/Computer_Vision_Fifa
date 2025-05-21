from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from view_transformer import  KeypointsTransformer
import json
import cv2
import os
import time
from offside_detector import OffsideDetector
from classical_ml.player_detection import ClassicalML
import keyboard

class ML_ComputerVision:
    def __init__(self):
        
        self.tracker = Tracker('./models/best_new.pt')
        # self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = KeypointsTransformer()
        self.attempts = None
        self.new_attempt = False
        self.classical_ml = ClassicalML()
        # self.offside_detector = OffsideDetector(gpu=True)
        # self.offside_detector.warmup()
        
    
    def read_frames(self,frames_path):
        # Read the frames from the specified path
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                # new_frame = cv2.resize(frame, (640, 640))
                frames.append(frame)
                if count == 55:
                    break
                count += 1
        return frames
        

    def run(self,video_frames,attempts):

        # for i, frame in enumerate(video_frames):
        #             timestamp = time.time()
        #             cv2.imwrite(f"D:/fifa-vision/all/frame_{timestamp}_{i}.jpg", frame)
        if self.attempts is None:
            self.attempts = attempts
            self.new_attempt = True
        else:
            self.new_attempt = False

        
        if self.attempts != attempts:
            # print("New attempt detected, resetting tracker...")
            self.attempts = attempts
            self.tracker.reset_tracker()
            self.new_attempt = True
        else:
            self.new_attempt = False
            # for i, frame in enumerate(video_frames):
            #         cv2.imwrite(f"D:/fifa-vision/new_attempts/frame_{self.attempts}_{i}.jpg", frame)

        all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}

        for test,frame in enumerate(video_frames):
            start_time = time.time()
            height, width, _ = frame.shape

            frame_tracks = {"players": {}, "referees": {}, "ball": {}, "goalkeeper": {} , "goalpost" : {}}
            frame_tracks['ball'] = self.tracker.get_ball_tracks(frame)
            frame_tracks["players"] = self.classical_ml.run(frame)

            self.tracker.remove_out_of_pitch_detections(frame_tracks["players"], height)
            #adjust goalkeeper detections

            # Replace Outliers
            # if frame_tracks["ball"]:
            # print("Before outliers: ", frame_tracks["ball"][1]["conf"] if frame_tracks["ball"] else None)
            # if frame_tracks["ball"] and not self.new_attempt: 
            #     frame_tracks["ball"] = self.tracker.replace_outliers_based_on_distance(
            #         frame_tracks["ball"], 170
            #     )

            # interpolate again

            if frame_tracks["ball"]:
                frame_tracks["ball"][1]["bbox"] = self.tracker.interpolate_ball_positions(
                    frame_tracks["ball"]
                )
            else:
                bbox = self.tracker.interpolate_ball_positions(frame_tracks["ball"])
                if bbox:
                    frame_tracks["ball"] = {
                        1: {
                            "bbox": bbox,
                            "conf": 0.11
                        }
                    }
                else:
                    print("NO BALL DETECTED OR INTERPOLATED")
                    

            # Get object positions
            self.tracker.add_position_to_tracks(frame_tracks)

            # # View Trasnformer
            self.view_transformer.get_matrix_transform(frame)
            self.view_transformer.add_transformed_position_to_tracks(frame_tracks)
            goal_post = self.view_transformer.real_world_to_pixel(self.tracker.goal_position)
            frame_tracks["goalpost"] = {
                1:{
                    "position_transformed": self.tracker.goal_position,
                    "position": goal_post
                }
            }
            
            frame_tracks = self.tracker.adjust_goalkeeper_detection(frame_tracks)
            
            # handle penalty detections
            frame_tracks["ball"],changed = self.tracker.handle_penalty_detection(frame_tracks["ball"])
            if changed:
                position = np.array(frame_tracks["ball"][1]["position"])
                position_trasnformed = self.view_transformer.pixel_to_real_world(position)
                if position_trasnformed is not None:
                    position_trasnformed = position_trasnformed.squeeze().tolist()
                # print("HANDLED" , position_trasnformed)
                frame_tracks['ball'][1]['position_transformed'] = position_trasnformed    
            
            
        

            # Assign Ball to Player
            if frame_tracks["ball"]:
                    ball_position = frame_tracks["ball"][1]["bbox"]
                    assigned_player = self.player_assigner.assign_ball_to_player(
                        frame_tracks["players"], ball_position
                    )

                    if assigned_player != -1:
                        # print("frame_tracks:" , frame_tracks["players"])
                        # print("assigned player:" , assigned_player)
                        # print("ball position:" , ball_position)
                        frame_tracks["players"][assigned_player]["has_ball"] = True
                        # save the bounding box of the player having the ball save it as new image 

                        # Convert bbox coordinates to integers before using for image cropping
                        # Convert bbox coordinates to integers and expand by 300px in each direction
                        bbox = frame_tracks["players"][assigned_player]["bbox"]
                        height, width, _ = frame.shape

                        # Calculate expanded coordinates with bounds checking
                        x1 = max(0, int(bbox[0]) - 600)
                        y1 = max(0, int(bbox[1]) - 300)
                        x2 = min(width, int(bbox[2]) + 600)
                        y2 = min(height, int(bbox[3]) + 300)

                        # Save the expanded crop
                      
                        # isOffside = self.offside_detector.detect_offside(frame[y1:y2, x1:x2])
                        
                        # if isOffside:

                        #     return None, None
                        


            

            all_tracks["players"].append(frame_tracks["players"])
            all_tracks["ball"].append(frame_tracks["ball"])
            all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
            all_tracks["goalpost"].append(frame_tracks["goalpost"])
        print("DONEEEEEE" , time.time() - start_time)

        # Draw output
        output_video_frames = self.tracker.draw_annotations(
            video_frames, all_tracks
        )
        save_video(output_video_frames, "output_videos/test_alwan_fouda.mp4")

        # # Save video
        # save_video(output_video_frames, "output_videos/FINAL_CV_ISA.mp4")
                # Check if any player in team 1 has the ball
        hasBall = any(player.get("has_ball", False) and player.get("team") == 1 for player in frame_tracks["players"].values())
        # return also the biggest x for team 1 and biggest x for team 2 using frame_tracks["players"]

        # farthest player in team 1 without has_ball = true
        # farthest player in team 2 without has_ball = true

        # all_tracks["players"]

        

        return frame_tracks, hasBall



    # def voting_algo_frames (self,all_tracks):
    #     # all tracks is a dictionary of lists of dictionaries
    #     # all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}
    #     # i want to return single dictionary of players, referees, ball, goalkeeper, goalpost
    #     # the problem if i take the last frame only is that it might not be the most accurate i want to make like voting algo and take average

