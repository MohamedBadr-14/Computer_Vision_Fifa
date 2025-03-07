from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def main():
    # Read Video
    video_frames = read_video('C:/Users/moham/Downloads/testat/test_frames.mp4')
    # np.save("full_video.npy", video_frames)
    all_tracks = { 
        "players": [],
        "referees": [],
        "ball": []
    }
    
    # Initialize Tracker
    tracker = Tracker('models/best_initial.pt')
    counter = 0 
    test=0
    team_ball_control= []
    view_transformer = ViewTransformer()
    team_assigner = TeamAssigner()
    player_assigner =PlayerBallAssigner()

    
    # frame = np.load("frame.npy") to load frames here


    while(test<67): #just for testing
        frame = video_frames[test]
        test+=1 #test
        frame_tracks = tracker.get_object_track(frame)


        # Get object positions 
        tracker.add_position_to_tracks(frame_tracks)

        # camera movement estimator
        if counter == 0:
            camera_movement_estimator = CameraMovementEstimator(frame)
        
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frame,
                                                                                    read_from_stub=False,
                                                                                    stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(frame_tracks,camera_movement_per_frame)

        # # View Trasnformer
        view_transformer.add_transformed_position_to_tracks(frame_tracks)

        # Interpolate Ball Positions
        frame_tracks["ball"] = tracker.interpolate_ball_positions(frame_tracks["ball"])


        # Assign Player Teams
        if counter == 0:
            team_assigner.assign_team_color(frame, 
                                        frame_tracks['players'])
            counter += 1

        for player_id, track in frame_tracks['players'].items():
            # Assign team to the player
            team = team_assigner.get_player_team(frame, track['bbox'], player_id)

            # Add team and team color information to the player's track
            frame_tracks['players'][player_id]['team'] = team
            frame_tracks['players'][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign Ball to Player
        if frame_tracks['ball']:  
            ball_bbox = frame_tracks['ball'][1]['bbox']  
            assigned_player = player_assigner.assign_ball_to_player(frame_tracks['players'], ball_bbox)

            if assigned_player != -1:
                frame_tracks['players'][assigned_player]['has_ball'] = True
                team_ball_control.append(frame_tracks['players'][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)


        all_tracks["players"].append(frame_tracks["players"])
        all_tracks["referees"].append(frame_tracks["referees"])
        all_tracks["ball"].append(frame_tracks["ball"])

    team_ball_control= np.array(team_ball_control)
    # Draw output 
    output_video_frames = tracker.draw_annotations(video_frames, all_tracks,team_ball_control)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
