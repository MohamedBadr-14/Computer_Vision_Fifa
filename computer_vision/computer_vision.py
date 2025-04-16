from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer, KeypointsTransformer
import cv2
import os
import pickle
import sys
import time

def adjust_all_tracks(all_tracks):
    loaded_tracks = {}
    for key in all_tracks.keys():
        loaded_tracks[key] = []
        for frame_num, frame_tracks in enumerate(all_tracks[key]):
            loaded_tracks[key].append({})
            for track_id, track in frame_tracks.items():
                #only load bounding box of each track
                # if key == "ball":
                #     print(track)
                if "bbox" in track.keys():
                    loaded_tracks[key][frame_num][track_id] = {
                        "bbox": track["bbox"]
                    }
                    if "conf" in track.keys():
                        loaded_tracks[key][frame_num][track_id]["conf"] = track["conf"]
                else:
                    loaded_tracks[key][frame_num][track_id] = {}
    return loaded_tracks

class ComputerVision:
    def __init__(self):
        self.tracker = Tracker("models/best_new.pt")
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = KeypointsTransformer()
        self.attempts = None
        

    def run(self,video_frames,attempts):

        if self.attempts is None:
            self.attempts = attempts

        if self.attempts != attempts:
            self.attempts = attempts
            self.tracker.reset_tracker()
        # Read Video
        # print("Reading video..." , track_stubs)
        # video_frames = read_video("input_videos/general_small.mp4")
        # all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}
        to_be_sent = {}

        # if track_stubs and os.path.exists("stubs/tracks.pkl"):
        #     with open("stubs/tracks.pkl", "rb") as f:
        #         loaded_tracks = pickle.load(f)

        print(len(video_frames))
        for test,frame in enumerate(video_frames):
            
            # if track_stubs and os.path.exists("stubs/tracks.pkl"):
            #     frame_tracks = {"players" : loaded_tracks["players"][test-1],
            #                     "referees" : loaded_tracks["referees"][test-1],
            #                     "ball" : loaded_tracks["ball"][test-1],
            #                     "goalkeeper" : loaded_tracks["goalkeeper"][test-1],
            #                     "goalpost" : loaded_tracks["goalpost"][test-1]}
            # else:

            frame_tracks = self.tracker.get_object_track(frame)

            frame_tracks = self.tracker.adjust_goalkeeper_detection(frame_tracks)

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
                # print("ball not detected")
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
            time_before = time.time()
            self.view_transformer.get_matrix_transform(frame)
            self.view_transformer.add_transformed_position_to_tracks(frame_tracks)
            print("\n  timeeeee -> " , time.time() - time_before)
            print("\nFRAMAATTTT" , frame_tracks["players"])

            # Assign Player Teams

            self.team_assigner.assign_team_color(frame, frame_tracks["players"])

            for player_id, track in frame_tracks["players"].items():
                
                team = self.team_assigner.get_player_team(frame, track["bbox"], player_id)
                # ##print(conf)
                frame_tracks["players"][player_id]["team"] = team
                frame_tracks["players"][player_id]["team_color"] = (
                    self.team_assigner.team_colors[team]
                )
            team_assigned = time.time()
            # Merge players from the same team who are very close to each other
            merged_players = {}
            distance_threshold = 40  # Adjust this threshold based on your use case

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
                del frame_tracks["players"][player_id]

            # Merge players from the same team who are very close to each other
            merged_players = {}
            distance_threshold = 40  # Adjust this threshold based on your use case

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
                del frame_tracks["players"][player_id]

            # print('frame trackss after:', frame_tracks["players"].items())
            # print('frame 222:', frame_tracks["players"][2])

            # Assign Ball to Player
            if frame_tracks["ball"]:
                ball_bbox = frame_tracks["ball"][1]["bbox"]
                assigned_player = self.player_assigner.assign_ball_to_player(
                    frame_tracks["players"], ball_bbox
                )

                if assigned_player != -1:
                    frame_tracks["players"][assigned_player]["has_ball"] = True

            ocr_dict = self.ocr_extractor.extract_points_and_attempts(frame , save=False)
            # print("end attempts and points" , "Points: ", ocr_dict)
            
            to_be_sent = {
                "players": frame_tracks["players"],
                "ball": frame_tracks["ball"],
                "goalkeeper": frame_tracks["goalkeeper"],
                "goalpost": frame_tracks["goalpost"],
                "points": ocr_dict['points'],
                "attempts": ocr_dict['attempts'],
                }
            
        

            # all_tracks["players"].append(frame_tracks["players"])
            # all_tracks["referees"].append(frame_tracks["referees"])
            # all_tracks["ball"].append(frame_tracks["ball"])
            # all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
            # all_tracks["goalpost"].append(frame_tracks["goalpost"])

        # to_bo_loaded_tracks = adjust_all_tracks(all_tracks)
        # Save all tracks to stub
        # if not os.path.exists("stubs"):
        #     os.makedirs("stubs")
        # with open("stubs/tracks.pkl", "wb") as f:
        #     pickle.dump(to_bo_loaded_tracks, f)

        # # Draw output
        # output_video_frames = self.tracker.draw_annotations(
        #     video_frames, all_tracks
        # )

        # # Save video
        # save_video(output_video_frames, "output_videos/points.mp4")
        return to_be_sent
