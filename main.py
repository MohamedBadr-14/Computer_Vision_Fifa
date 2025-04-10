from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
import cv2
import time

def get_kmeans(frame , model , team_assigner):
    tracks = model.get_object_track(frame)
    testing = time.time()
    team_assigner.assign_team_color(frame, 
                                tracks['players'])
    print("time team color" , time.time()-testing)

def main():
    # Read Video
    video_frames = read_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/general_small.mp4')
    # k_means = cv2.imread('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/kmeans.png')
    # np.save("full_video.npy", video_frames)
    all_tracks = { 
        "players": [],
        "referees": [],
        "ball": []
    }
    
    # Initialize Tracker
    tracker = Tracker('models/best_new.pt')
    counter = 0 
    test=0
    fit_transformed = True
    team_ball_control= []
    view_transformer = ViewTransformer()
    team_assign_time = time.time()
    team_assigner = TeamAssigner()
    after_team_assign  = time.time()
    print("team assign time", after_team_assign - team_assign_time)
    player_assigner =PlayerBallAssigner()
    # get_kmeans(frame=k_means, model = kmeans_track, team_assigner=team_assigner)
    

    print("video_frames",len(video_frames))
    video_frames = [video_frames[7]]
    while(test<len(video_frames)): #just for testing
        frame = video_frames[test]
        test+=1 #test
        frame_tracks , detection_supervision = tracker.get_object_track(frame)



        if all_tracks['ball'] and all_tracks['ball'][-1]:
        
            # Replace Outliers
            if frame_tracks['ball'] and frame_tracks['ball'][1]['conf'] < 0.2:    
                frame_tracks['ball'] = tracker.replace_outliers_based_on_distance(frame_tracks['ball'], 120)
        

            #interpolate again
            
            if frame_tracks['ball']:
                frame_tracks['ball'][1]['bbox'] = tracker.interpolate_ball_positions(frame_tracks['ball'])
            else: 
                frame_tracks['ball'] = {1: {'bbox': tracker.interpolate_ball_positions(frame_tracks['ball'])}}
       

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


        # Assign Player Teams
        # if counter == 0:
        #     # team_assigner.assign_team_color(frame, 
        #     #                             frame_tracks['players'])
        #     counter += 1

        time_bef_assign = time.time()
        crops = team_assigner.extract_crops(frame, detection_supervision)
        embedings = team_assigner.extract_embeddings(crops)
        clusters = team_assigner.predict_kmeans(embedings , fit_transformed)
        fit_transformed = True
        print("time assign" , time.time()-time_bef_assign)
        print("clusters", clusters)
        

        players_count = 0

        for player_id, track in frame_tracks['players'].items():
            # Assign team to the player
            # team = team_assigner.get_player_team(frame, track['bbox'], player_id)

            # Add team and team color information to the player's track
            frame_tracks['players'][player_id]['team'] = clusters[players_count]
            frame_tracks['players'][player_id]['team_color'] = [100,100,100] if clusters[players_count] == 1 else [255,255,255]
            players_count += 1

        # Assign Ball to Player
        if frame_tracks['ball']: 
            ball_bbox = frame_tracks['ball'][1]['bbox']  
            assigned_player = player_assigner.assign_ball_to_player(frame_tracks['players'], ball_bbox)

            if assigned_player != -1:
                frame_tracks['players'][assigned_player]['has_ball'] = True
                # print("assigned_player", assigned_player , test)
                team_ball_control.append(frame_tracks['players'][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)


        all_tracks["players"].append(frame_tracks["players"])
        all_tracks["referees"].append(frame_tracks["referees"])
        all_tracks["ball"].append(frame_tracks["ball"])

    team_ball_control= np.array(team_ball_control)
    # # Draw output 
    output_video_frames = tracker.draw_annotations(video_frames, all_tracks,team_ball_control)


    # # Save video
    save_video(output_video_frames , 'output_videos/yarabb.mp4')
    

if __name__ == '__main__':
    main()
