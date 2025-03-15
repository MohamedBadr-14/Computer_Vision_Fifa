from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def main():
    # Read Video
    video_frames = read_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/test_sogh.mp4')
    # np.save("full_video.npy", video_frames)
    all_tracks = { 
        "Player Home": [],
        "Player Away": [],
        "Ref": [],
        "Ball": []
    }
    
    # Initialize Tracker
    tracker = Tracker('models/best_e7na.pt')
    counter = 0 
    test=0
    team_ball_control= []
    view_transformer = ViewTransformer()
    team_assigner = TeamAssigner()
    player_assigner =PlayerBallAssigner()

    
    # frame = np.load("frame.npy") to load frames here


    while(test<98): #just for testing
        print(len(video_frames))
        frame = video_frames[test]
        test+=1 #test
        frame_tracks = tracker.get_object_track(frame)


        # Get object positions 
        tracker.add_position_to_tracks(frame_tracks)

        # camera movement estimator
        if counter == 0:
            camera_movement_estimator = CameraMovementEstimator(frame)
            counter += 1
        
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frame,
                                                                                    read_from_stub=False,
                                                                                    stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(frame_tracks,camera_movement_per_frame)

        # # View Trasnformer
        view_transformer.add_transformed_position_to_tracks(frame_tracks)

        # Interpolate Ball Positions
        frame_tracks["Ball"] = tracker.interpolate_ball_positions(frame_tracks["Ball"])


        # Assign Player Teams
        # if counter == 0:
        #     team_assigner.assign_team_color(frame, 
        #                                 frame_tracks['players'])
        #     counter += 1

        for player_id, track in frame_tracks['Player Home'].items():
            # Assign team to the player
            team = 0

            # Add team and team color information to the player's track
            frame_tracks['Player Home'][player_id]['team'] = team
            frame_tracks['Player Home'][player_id]['team_color'] = [255,255,255]

        for player_id, track in frame_tracks['Player Away'].items():
            # Assign team to the player
            team = 1

            # Add team and team color information to the player's track
            frame_tracks['Player Away'][player_id]['team'] = team
            frame_tracks['Player Away'][player_id]['team_color'] = [128,128,128]

        # Assign Ball to Player
        if frame_tracks['Ball']:  
            ball_bbox = frame_tracks['Ball'][1]['bbox']  
            players_dict = frame_tracks['Player Home'] | frame_tracks['Player Away']
            assigned_player,team_num = player_assigner.assign_ball_to_player(players_dict, ball_bbox)

            if assigned_player != -1:
                if team_num == 0:
                    frame_tracks['Player Home'][assigned_player]['has_ball'] = True
                    team_ball_control.append(frame_tracks['Player Home'][assigned_player]['team'])
                elif team_num == 1:
                    frame_tracks['Player Away'][assigned_player]['has_ball'] = True
                    team_ball_control.append(frame_tracks['Player Away'][assigned_player]['team'])
                    
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)


        all_tracks["Player Home"].append(frame_tracks["Player Home"])
        all_tracks["Player Away"].append(frame_tracks["Player Away"])
        all_tracks["Ref"].append(frame_tracks["Ref"])
        all_tracks["Ball"].append(frame_tracks["Ball"])

    team_ball_control= np.array(team_ball_control)
    # Draw output 
    output_video_frames = tracker.draw_annotations(video_frames, all_tracks,team_ball_control)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
