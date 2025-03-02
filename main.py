from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('C:/Users/moham/Downloads/test.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best_e7na.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=False,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    print(tracks.keys())
    tracks["Ball"] = tracker.interpolate_ball_positions(tracks["Ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['Player Home'][0])
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['Player Away'][0])
    
    for frame_num, player_track in enumerate(tracks['Player Home']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['Player Home'][frame_num][player_id]['team'] = team 
            tracks['Player Home'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    for frame_num,player_track in enumerate(tracks['Player Away']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['Player Away'][frame_num][player_id]['team'] = team
            tracks['Player Away'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    # player_assigner =PlayerBallAssigner()
    # team_ball_control= []
    # for frame_num, player_track in enumerate(tracks['Player Home']):
    #     ball_bbox = tracks['Ball'][frame_num][1]['bbox']
    #     assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

    #     if assigned_player != -1:
    #         tracks['Player Home'][frame_num][assigned_player]['has_ball'] = True
    #         team_ball_control.append(tracks['Player Home'][frame_num][assigned_player]['team'])
    #     else:
    #         print("FRAME_NUMBER" , frame_num)
    #         if len(team_ball_control) == 0:
    #             team_ball_control.append(1)
    #         else:
    #             team_ball_control.append(team_ball_control[-1])
    # team_ball_control= np.array(team_ball_control)

    # for frame_num, player_track in enumerate(tracks['Player Away']):
    #     ball_bbox = tracks['Ball'][frame_num][1]['bbox']
    #     assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

    #     if assigned_player != -1:
    #         tracks['Player Away'][frame_num][assigned_player]['has_ball'] = True
    #         team_ball_control.append(tracks['Player Away'][frame_num][assigned_player]['team'])
    #     else:
    #         if len(team_ball_control) == 0:
    #             team_ball_control.append(1)
    #         else:
    #             team_ball_control.append(team_ball_control[-1])


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()