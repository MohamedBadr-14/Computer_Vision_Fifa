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



def read_frames(frames_path):
    # Read the frames from the specified path
    frames = []
    count = 0
    for filename in sorted(os.listdir(frames_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frame_path = os.path.join(frames_path, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
            if count == 150:
                break
            count += 1
    return frames

def get_dominant_has_ball(team_ball_control):
    # Get the most common team that has the ball
    if len(team_ball_control) == 0:
        return None
    unique, counts = np.unique(team_ball_control, return_counts=True)
    dominant_team = unique[np.argmax(counts)]
    return dominant_team
    

def main(track_stubs=False):
    # Read Video
    print("Reading video..." , track_stubs)
    # video_frames = read_video("input_videos/training_final.mp4")
    # video_frames = video_frames[:30]  # Downsample the video frames for testing
    all_tracks = {"players": [], "ball": [] , "goalkeeper": [] , "goalpost" : []}

    if track_stubs and os.path.exists("stubs/tracks.pkl"):
        with open("stubs/tracks.pkl", "rb") as f:
            loaded_tracks = pickle.load(f)

    # Initialize Tracker
    tracker = Tracker("models/best_new.pt")
    test = 0
    team_ball_control = []
    view_transformer = ViewTransformer()
    team_assign_time = time.time()
    team_assigner = TeamAssigner()
    after_team_assign = time.time()
    print("team assign time", after_team_assign - team_assign_time)
    player_assigner = PlayerBallAssigner()
    # ocr_extractor = OCRExtractor()
    
    # video_frames = [video_frames[30]]  
    video_frames = read_frames("./badr/all")
    # video_frames = [cv2.imread("./goal1/frames/frame_0003.png")]
    # video_frames = video_frames[94:]
    print(video_frames[0].shape)

    print(len(video_frames))
    while test < len(video_frames):  # just for testing
        print("FRAME NUMBER: ", test)
        frame = video_frames[test]
        height, width, _ = frame.shape
        # frame[:int(0.1*height)] = 0  # Set the bottom 10% of the frame to black
        if test == 0:
            cv2.imwrite("frame_save.png" , frame)
            start_time = time.time()

        if track_stubs and os.path.exists("stubs/tracks.pkl"):
            frame_tracks = {"players" : loaded_tracks["players"][test-1],
                            "ball" : loaded_tracks["ball"][test-1],
                            "goalkeeper" : loaded_tracks["goalkeeper"][test-1],
                            "goalpost" : loaded_tracks["goalpost"][test-1]}
        else:
            frame_tracks = tracker.get_object_track(frame)
        test += 1  # test

        tracker.remove_out_of_pitch_detections(frame_tracks["players"], height)
        #adjust goalkeeper detections

        # Replace Outliers
        if frame_tracks["ball"] and frame_tracks["ball"][1]["conf"] < 0.2 :
            print("CONFF" , frame_tracks["ball"][1]["conf"])
            frame_tracks["ball"] = tracker.replace_outliers_based_on_distance(
                frame_tracks["ball"], 120
            )

        # interpolate again

        if frame_tracks["ball"]:
            print("ball detected" , frame_tracks["ball"][1]["bbox"] , frame_tracks["ball"][1]["conf"])
            frame_tracks["ball"][1]["bbox"] = tracker.interpolate_ball_positions(
                frame_tracks["ball"]
            )
        else:
            print("ball not detected")
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

        # # View Trasnformer
        # view_transformer.add_transformed_position_to_tracks(frame_tracks)
        view_transformer.get_matrix_transform(frame)
        view_transformer.add_transformed_position_to_tracks(frame_tracks)
        goal_post = view_transformer.real_world_to_pixel(tracker.goal_position)
        frame_tracks["goalpost"] = {
            1:{
                "position_transformed": tracker.goal_position,
                "position": goal_post
            }
        }
        
        frame_tracks = tracker.adjust_goalkeeper_detection(frame_tracks)

        # Assign Player Teams
        crops,crop_ids = team_assigner.extract_crops(frame_tracks["players"], frame)
        embedings = team_assigner.extract_embeddings(crops)
        clusters = team_assigner.predict_kmeans(embedings)
        team_assigner.assign_clusters_to_players(clusters, crop_ids, frame_tracks["players"])
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
        
        
        
    

        # print('frame trackss after:', frame_tracks["players"].items())
        # print('frame 222:', frame_tracks["players"][2])

        # Assign Ball to Player
        if frame_tracks["ball"]:
                ball_position = frame_tracks["ball"][1]["bbox"]
                assigned_player = player_assigner.assign_ball_to_player(
                    frame_tracks["players"], ball_position
                )

                if assigned_player != -1:
                    # print("frame_tracks:" , frame_tracks["players"])
                    # print("assigned player:" , assigned_player)
                    # print("ball position:" , ball_position)
                    frame_tracks["players"][assigned_player]["has_ball"] = True
                    team_ball_control.append(frame_tracks["players"][assigned_player]["team"])

        # print("start attempts and points")
        # if save_count == 0:
        #     cv2.imwrite("frame_save.png" , frame)
        # ocr_dict = ocr_extractor.extract_points_and_attempts(frame , save=False if save_count > 0 else True)
        # save_count+=1
        # print("end attempts and points" , "Points: ", ocr_dict)
        

        all_tracks["players"].append(frame_tracks["players"])
        all_tracks["ball"].append(frame_tracks["ball"])
        all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
        all_tracks["goalpost"].append(frame_tracks["goalpost"])
    print("DONEEEEEE" , time.time() - start_time)
    # print("\n\n ALL TRACKSS " , all_tracks["goalpost"])

    team_ball_control = np.array([0,0,1,1])
    dominant = get_dominant_has_ball(team_ball_control)
    # print("\n\n" , all_tracks)
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
    save_video(output_video_frames, "output_videos/CV_Final.mp4")
    # for frame in output_video_frames:
    #     cv2.imwrite("output_videos/frame_{}.jpg".format(save_count), frame)
    #     save_count+=1


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
#         

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