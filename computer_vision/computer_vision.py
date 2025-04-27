from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from view_transformer import  KeypointsTransformer
import cv2
import os
import time

class ComputerVision:
    def __init__(self):
        self.tracker = Tracker("models/best_new.pt")
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = KeypointsTransformer()
        self.attempts = None
        
    
    def read_frames(self,frames_path):
        # Read the frames from the specified path
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                frames.append(frame)
                if count == 100:
                    break
                count += 1
        return frames
        

    def run(self,video_frames,attempts):

        # for i, frame in enumerate(video_frames):
        #             timestamp = time.time()
        #             cv2.imwrite(f"D:/fifa-vision/all/frame_{timestamp}_{i}.jpg", frame)
        if self.attempts is None:
            self.attempts = attempts

        print("Prev attempt: ", self.attempts)
        print("Cur attempt: ", attempts)
        if self.attempts != attempts:
            # print("New attempt detected, resetting tracker...")
            self.attempts = attempts
            self.tracker.reset_tracker()
            # for i, frame in enumerate(video_frames):
            #         cv2.imwrite(f"D:/fifa-vision/new_attempts/frame_{self.attempts}_{i}.jpg", frame)
        # Read Video
        # print("Reading video..." , track_stubs)
        # video_frames = read_video("input_videos/general_small.mp4")
        all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}
        to_be_sent = {}

        # if track_stubs and os.path.exists("stubs/tracks.pkl"):
        #     with open("stubs/tracks.pkl", "rb") as f:
        #         loaded_tracks = pickle.load(f)

        print(len(video_frames))
        for test,frame in enumerate(video_frames):
            start_time = time.time()
            frame = video_frames[test]
            height, width, _ = frame.shape

            
            frame_tracks = self.tracker.get_object_track(frame)
            test += 1  # test

            self.tracker.remove_out_of_pitch_detections(frame_tracks["players"], height)
            #adjust goalkeeper detections

            # Replace Outliers
            if frame_tracks["ball"] and frame_tracks["ball"][1]["conf"] < 0.2 :
                frame_tracks["ball"] = self.tracker.replace_outliers_based_on_distance(
                    frame_tracks["ball"], 120
                )

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
                    

            # Get object positions
            self.tracker.add_position_to_tracks(frame_tracks)

            # # View Trasnformer
            # view_transformer.add_transformed_position_to_tracks(frame_tracks)
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

            # Assign Player Teams
            crops,crop_ids = self.team_assigner.extract_crops(frame_tracks["players"], frame)
            embedings = self.team_assigner.extract_embeddings(crops)
            clusters = self.team_assigner.predict_kmeans(embedings)
            self.team_assigner.assign_clusters_to_players(clusters, crop_ids, frame_tracks["players"])
            # Merge players from the same team who are very close to each other
            merged_players = {}
            distance_threshold = 75  # Adjust this threshold based on your use case

            for player_id, track in list(frame_tracks["players"].items()):
                if player_id in merged_players:
                    continue  # Skip already merged players
                # print('one track: ', track)
                position1 = track["position"]
                team1 = track["team"]
                conf1 = track["conf"]  # Confidence of the first player

                for other_player_id, other_track in list(frame_tracks["players"].items()):
                    if player_id == other_player_id or other_player_id in merged_players:
                        continue

                    position2 = other_track["position"]
                    team2 = other_track["team"]
                    conf2 = other_track["conf"]  # Confidence of the second player

                    # Check if players are from the same team and very close
                    if team1 == team2:
                        distance = np.linalg.norm(np.array(position1) - np.array(position2))
                        if distance < distance_threshold:
                            # Keep the player with the highest confidence
                            if conf1 >= conf2:
                                merged_players[other_player_id] = True
                            else:
                                merged_players[player_id] = True
                                # Update the current player's data with the other player's data
                                # frame_tracks["players"][player_id] = other_track
                                break  # Stop checking further for this player

            # Remove duplicate players
            for player_id in merged_players:
                # print("Removing player ID:", player_id)
                del frame_tracks["players"][player_id]
            
        

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

            

            all_tracks["players"].append(frame_tracks["players"])
            all_tracks["ball"].append(frame_tracks["ball"])
            all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
            all_tracks["goalpost"].append(frame_tracks["goalpost"])
        print("DONEEEEEE" , time.time() - start_time)

        # Draw output
        output_video_frames = self.tracker.draw_annotations(
            video_frames, all_tracks
        )
        # save_video(output_video_frames, "output_videos/points.mp4")

        # # Save video
        save_video(output_video_frames, "output_videos/FINAL_CV_ISA.mp4")
                # Check if any player in team 1 has the ball
        hasBall = any(player.get("has_ball", False) and player.get("team") == 1 for player in frame_tracks["players"].values())
        return frame_tracks, hasBall



    # def voting_algo_frames (self,all_tracks):
    #     # all tracks is a dictionary of lists of dictionaries
    #     # all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}
    #     # i want to return single dictionary of players, referees, ball, goalkeeper, goalpost
    #     # the problem if i take the last frame only is that it might not be the most accurate i want to make like voting algo and take average

