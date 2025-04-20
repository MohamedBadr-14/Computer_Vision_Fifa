from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from ocr_extractor import OCRExtractor
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



def main(track_stubs=False):
    # Read Video
    print("Reading video..." , track_stubs)
    video_frames = read_video("input_videos/goalkeeper.mp4")
    # video_frames = video_frames[:30]  # Downsample the video frames for testing
    all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}

    if track_stubs and os.path.exists("stubs/tracks.pkl"):
        with open("stubs/tracks.pkl", "rb") as f:
            loaded_tracks = pickle.load(f)

    # Initialize Tracker
    tracker = Tracker("models/best_new.pt")
    counter = 0
    test = 0
    save_count = 0
    team_ball_control = []
    view_transformer = ViewTransformer()
    team_assign_time = time.time()
    team_assigner = TeamAssigner()
    after_team_assign = time.time()
    print("team assign time", after_team_assign - team_assign_time)
    player_assigner = PlayerBallAssigner()
    team_assigner = TeamAssigner()
    # ocr_extractor = OCRExtractor()
    
    # video_frames = [video_frames[30]]  

    print(len(video_frames))
    while test < len(video_frames):  # just for testing
        frame = video_frames[test]
        

        if track_stubs and os.path.exists("stubs/tracks.pkl"):
            frame_tracks = {"players" : loaded_tracks["players"][test-1],
                            "referees" : loaded_tracks["referees"][test-1],
                            "ball" : loaded_tracks["ball"][test-1],
                            "goalkeeper" : loaded_tracks["goalkeeper"][test-1],
                            "goalpost" : loaded_tracks["goalpost"][test-1]}
        else:
            frame_tracks = tracker.get_object_track(frame)
        test += 1  # test

        #adjust goalkeeper detections
        # print("\nframe tracks before adjust: ", frame_tracks , "\n")
        frame_tracks = tracker.adjust_goalkeeper_detection(frame_tracks)

        # Replace Outliers
        if frame_tracks["ball"] and frame_tracks["ball"][1]["conf"] < 0.2 :
            # print("CONFF" , frame_tracks["ball"][1]["conf"])
            frame_tracks["ball"] = tracker.replace_outliers_based_on_distance(
                frame_tracks["ball"], 120
            )

        # interpolate again

        if frame_tracks["ball"]:
            frame_tracks["ball"][1]["bbox"] = tracker.interpolate_ball_positions(
                frame_tracks["ball"]
            )
        else:
            # print("ball not detected")
            bbox = tracker.interpolate_ball_positions(frame_tracks["ball"])
            if bbox:
                frame_tracks["ball"] = {
                    1: {
                        "bbox": bbox,
                        "conf": 0.11
                    }
                }

        # Get object positions
        tracker.add_position_to_tracks(frame_tracks)

        # camera movement estimator
        # if counter == 0:
        #     camera_movement_estimator = CameraMovementEstimator(frame)
        #     counter += 1

        # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        #     frame,
        #     read_from_stub=False,
        #     stub_path="camera_movement_estimator/stubs/camera_movement_stub.pkl",
        # )
        # camera_movement_estimator.add_adjust_positions_to_tracks(
        #     frame_tracks, camera_movement_per_frame
        # )

        # # View Trasnformer
        # view_transformer.add_transformed_position_to_tracks(frame_tracks)
        view_transformer.get_matrix_transform(frame)
        view_transformer.add_transformed_position_to_tracks(frame_tracks)

        # Assign Player Teams
        # detection_supervision = sv.Detections.from_ultralytics(result[0])
        crops,crop_ids = team_assigner.extract_crops(frame_tracks["players"], frame)
        embedings = team_assigner.extract_embeddings(crops)
        
        clusters = team_assigner.predict_kmeans(embedings)
        team_assigner.assign_clusters_to_players(clusters, crop_ids, frame_tracks["players"])
        # print(frame_tracks["players"])
        # #save crops to new directory named crops
        # if not os.path.exists("crops"):
        #     os.makedirs("crops")
        # for i, crop in zip(crop_ids, crops):
        #     cv2.imwrite(f"crops/crop_{i}.jpg", crop)
        
        
        
        

        # Remove duplicate players
        # for player_id in merged_players:
        #     del frame_tracks["players"][player_id]

        # print('frame trackss after:', frame_tracks["players"].items())
        # print('frame 222:', frame_tracks["players"][2])

        # Assign Ball to Player
        if frame_tracks["ball"]:
                ball_position = frame_tracks["ball"][1]["bbox"]
                assigned_player = player_assigner.assign_ball_to_player(
                    frame_tracks["players"], ball_position
                )

                if assigned_player != -1:
                    print("frame_tracks:" , frame_tracks["players"])
                    print("assigned player:" , assigned_player)
                    print("ball position:" , ball_position)
                    frame_tracks["players"][assigned_player]["has_ball"] = True

        # print(frame_tracks)
        # print("start attempts and points")
        # if save_count == 0:
        #     cv2.imwrite("frame_save.png" , frame)
        # ocr_dict = ocr_extractor.extract_points_and_attempts(frame , save=False if save_count > 0 else True)
        save_count+=1
        # print("end attempts and points" , "Points: ", ocr_dict)
        
        to_be_sent = {
            "players": frame_tracks["players"],
            "referees": frame_tracks["referees"],
            "ball": frame_tracks["ball"],
            "goalkeeper": frame_tracks["goalkeeper"],
            "goalpost": frame_tracks["goalpost"]
            # "points": ocr_dict['points'],
            # "attempts": ocr_dict['attempts'],
            }

        all_tracks["players"].append(frame_tracks["players"])
        all_tracks["referees"].append(frame_tracks["referees"])
        all_tracks["ball"].append(frame_tracks["ball"])
        all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
        all_tracks["goalpost"].append(frame_tracks["goalpost"])

    team_ball_control = np.array(team_ball_control)
    # to_bo_loaded_tracks = adjust_all_tracks(all_tracks)
    # # Save all tracks to stub
    # if not os.path.exists("stubs"):
    #     os.makedirs("stubs")
    # with open("stubs/tracks.pkl", "wb") as f:
    #     pickle.dump(to_bo_loaded_tracks, f)

    # # Draw output
    output_video_frames = tracker.draw_annotations(
        video_frames, all_tracks, team_ball_control
    )

    # # # Save video
    save_video(output_video_frames, "output_videos/FINALLLLLLLLLLLLLLLL.mp4")


if __name__ == "__main__":
    track_stubs = False
    if len(sys.argv) > 1 :
        track_stubs = sys.argv[1]
    main(track_stubs)
    
    
    
    
    
    
# team_assigner.assign_team_color(frame, frame_tracks["players"])

#         for player_id, track in frame_tracks["players"].items():
#             # ##print(tracks['players'][frame_num][player_id].keys())
#             # team_conf = tracks['players'][frame_num][player_id]['team_confidence'] if "team_confidence" in tracks['players'][frame_num][player_id].keys() else None
#             # team_assigner.assign_team_color(
#             #     frame, frame_tracks["players"]
#             # )
#             team = team_assigner.get_player_team(frame, track["bbox"], player_id)
#             # ##print(conf)
#             frame_tracks["players"][player_id]["team"] = team
#             frame_tracks["players"][player_id]["team_color"] = (
#                 team_assigner.team_colors[team]
#             )
#             # if count < 10:
#             #     ##print(track)
#             #     count+=1

#         team_assigned = time.time()
#         # print('frame trackss before:', frame_tracks["players"].items())
#         # Merge players from the same team who are very close to each other
#         merged_players = {}
#         distance_threshold = 40  # Adjust this threshold based on your use case

#         for player_id, track in list(frame_tracks["players"].items()):
#             if player_id in merged_players:
#                 continue  # Skip already merged players
#             # print('one track: ', track)
#             position1 = track["position"]
#             team1 = track["team"]
#             conf1 = track["conf"]  # Confidence of the first player

#             for other_player_id, other_track in list(frame_tracks["players"].items()):
#                 if player_id == other_player_id or other_player_id in merged_players:
#                     continue

#                 position2 = other_track["position"]
#                 team2 = other_track["team"]
#                 conf2 = other_track["conf"]  # Confidence of the second player

#                 # Check if players are from the same team and very close
#                 if team1 == team2:
#                     distance = np.linalg.norm(np.array(position1) - np.array(position2))
#                     if distance < distance_threshold:
#                         # Keep the player with the highest confidence
#                         if conf1 >= conf2:
#                             merged_players[other_player_id] = True
#                         else:
#                             merged_players[player_id] = True
#                             # Update the current player's data with the other player's data
#                             # frame_tracks["players"][player_id] = other_track
#                             break  # Stop checking further for this player

#         # Remove duplicate players
#         for player_id in merged_players:
#             del frame_tracks["players"][player_id]

#         # print('frame trackss after:', frame_tracks["players"].items())
#         # print('frame 222:', frame_tracks["players"][2])
#         # print('frame trackss before:', frame_tracks["players"].items())
#         # Merge players from the same team who are very close to each other
#         merged_players = {}
#         distance_threshold = 40  # Adjust this threshold based on your use case

#         for player_id, track in list(frame_tracks["players"].items()):
#             if player_id in merged_players:
#                 continue  # Skip already merged players
#             # print('one track: ', track)
#             position1 = track["position"]
#             team1 = track["team"]
#             conf1 = track["conf"]  # Confidence of the first player

#             for other_player_id, other_track in list(frame_tracks["players"].items()):
#                 if player_id == other_player_id or other_player_id in merged_players:
#                     continue

#                 position2 = other_track["position"]
#                 team2 = other_track["team"]
#                 conf2 = other_track["conf"]  # Confidence of the second player

#                 # Check if players are from the same team and very close
#                 if team1 == team2:
#                     distance = np.linalg.norm(np.array(position1) - np.array(position2))
#                     if distance < distance_threshold:
#                         # Keep the player with the highest confidence
#                         if conf1 >= conf2:
#                             merged_players[other_player_id] = True
#                         else:
#                             merged_players[player_id] = True
#                             # Update the current player's data with the other player's data
#                             # frame_tracks["players"][player_id] = other_track
#                             break  # Stop checking further for this player