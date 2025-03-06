# from utils import read_video, save_video
# from trackers import Tracker
# import cv2
# import numpy as np
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator


# def main():
#     # Read Video
#     video_frames = read_video('C:/Users/moham/Downloads/testat/test_soghayar.mp4')

    

#     # Initialize Tracker
#     tracker = Tracker('models/best_initial.pt')

#     tracks = tracker.get_object_tracks(video_frames,
#                                        read_from_stub=False,
#                                        stub_path='stubs/track_stubs.pkl')
#     # Get object positions 
#     tracker.add_position_to_tracks(tracks)

#     # camera movement estimator
#     camera_movement_estimator = CameraMovementEstimator(video_frames[0])
#     camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
#                                                                                 read_from_stub=False,
#                                                                                 stub_path='stubs/camera_movement_stub.pkl')
#     camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


#     # View Trasnformer
#     view_transformer = ViewTransformer()
#     view_transformer.add_transformed_position_to_tracks(tracks)

#     # Interpolate Ball Positions
#     tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#     # Speed and distance estimator
#     # speed_and_distance_estimator = SpeedAndDistance_Estimator()
#     # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#     # Assign Player Teams
#     team_assigner = TeamAssigner()
#     team_assigner.assign_team_color(video_frames[0], 
#                                     tracks['players'][0])
    
#     for frame_num, player_track in enumerate(tracks['players']):
#         for player_id, track in player_track.items():
#             team = team_assigner.get_player_team(video_frames[frame_num],   
#                                                  track['bbox'],
#                                                  player_id)
#             tracks['players'][frame_num][player_id]['team'] = team 
#             tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
#     # Assign Ball Aquisition
#     player_assigner =PlayerBallAssigner()
#     team_ball_control= []
#     for frame_num, player_track in enumerate(tracks['players']):
#         print("BADR" , player_track)
#         ball_bbox = tracks['ball'][frame_num][1]['bbox']
#         assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

#         if assigned_player != -1:
#             tracks['players'][frame_num][assigned_player]['has_ball'] = True
#             team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
#         else:
#             team_ball_control.append(team_ball_control[-1])
#     team_ball_control= np.array(team_ball_control)


#     # Draw output 
#     ## Draw object Tracks
#     output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

#     ## Draw Camera movement
#     # output_video_frames = camera_movement_estimator.draw_camera_movement(video_frames,camera_movement_per_frame)

#     ## Draw Speed and Distance
#     # speed_and_distance_estimator.draw_speed_and_distance(video_frames,tracks)

#     # Save video
#     save_video(output_video_frames, 'output_videos/output_video.avi')

# if __name__ == '__main__':
#     main()


import multiprocessing
import time
import numpy as np
from mss import mss
import cv2
import keyboard  # For global key detection

def frame_processor(queue, stop_event):
    """
    Processes frames from the queue.
    
    :param queue: Multiprocessing queue to receive frames.
    :param stop_event: Event to stop processing.
    """
    while not stop_event.is_set():
        if not queue.empty():
            # Get a frame from the queue
            frame = queue.get()

            # Perform your processing here
            # Example: Display the frame
            # cv2.imshow("Processed Frame", frame)
            print('frame_processed')


            # Example: Save the frame
            # cv2.imwrite(f"frame_{int(time.time())}.png", frame)

    print("Frame processing stopped.")
    cv2.destroyAllWindows()

def screen_recorder(queue, stop_event, capture_event):
    """
    Captures the screen and sends frames to the queue when triggered by an event.
    
    :param queue: Multiprocessing queue to send frames.
    :param stop_event: Event to stop the recording.
    :param capture_event: Event to trigger a screenshot.
    """
    # Define the screen region to capture (entire screen in this case)
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust to your screen resolution

    # Initialize mss
    with mss() as sct:
        while not stop_event.is_set():
            # Wait for the capture event to be set
            capture_event.wait()

            # Capture the screen
            frame = np.array(sct.grab(monitor))

            # Convert the frame from BGRA to BGR (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Put the frame in the queue
            if not queue.full():
                queue.put(frame)

            # Clear the capture event for the next trigger
            capture_event.clear()

    print("Screen recording stopped.")

def global_key_listener(stop_event, capture_event):
    """
    Listens for global key presses to stop the processes or trigger a screenshot.
    
    :param stop_event: Event to signal stopping.
    :param capture_event: Event to trigger a screenshot.
    """
    print("Press 's' to take a screenshot, 'q' to stop the processes.")
    while not stop_event.is_set():
        if keyboard.is_pressed('s'):  # Trigger a screenshot
            capture_event.set()
            print("Screenshot triggered.")
            time.sleep(0.5)  # Debounce to avoid multiple triggers
        elif keyboard.is_pressed('q'):  # Stop the processes
            stop_event.set()
            print("'q' pressed. Stopping processes...")
            break
        time.sleep(0.1)  # Small delay to avoid busy-waiting

if __name__ == "__main__":
    # Create a queue to share frames between processes
    queue = multiprocessing.Queue(maxsize=10)  # Adjust maxsize as needed

    # Create events to signal stopping and capturing
    stop_event = multiprocessing.Event()
    capture_event = multiprocessing.Event()

    # Create and start the screen recorder process
    recorder_process = multiprocessing.Process(
        target=screen_recorder,
        args=(queue, stop_event, capture_event)
    )
    recorder_process.start()

    # Create and start the frame processor process
    processor_process = multiprocessing.Process(
        target=frame_processor,
        args=(queue, stop_event)
    )
    processor_process.start()

    # Start the global key listener in the main process
    global_key_listener(stop_event, capture_event)

    # Wait for the processes to finish
    recorder_process.join()
    processor_process.join()

    print("Main program exited.")