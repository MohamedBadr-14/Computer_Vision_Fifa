from utils import read_video, save_video
from trackers import Tracker
import numpy as np
# from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from view_transformer import  KeypointsTransformer
import cv2
import os
import time
from offside_detector import OffsideDetector
from classical_ml.player_detection import ClassicalML
from trackers.players_interpolator import SimplePlayerInterpolator
from trackers.tracker_AI import PlayerTrackCleanerAI
import json

action_list = {
    "pass_N": 0,
    "pass_NE": 1,
    "pass_E": 2,
    "pass_SE": 3,
    "pass_S": 4,
    "pass_SW": 5,
    "pass_W": 6,
    "pass_NW": 7,
    "dribble_N": 8,
    "dribble_NE": 9,
    "dribble_E": 10,
    "dribble_SE": 11,
    "dribble_S": 12,
    "dribble_SW": 13,
    "dribble_W": 14,
    "dribble_NW": 15,
    "shoot": 16
}
class ML_ComputerVision:
    def __init__(self):
        
        self.tracker = Tracker('./models/best_new.pt')
        # self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = KeypointsTransformer()
        self.attempts = None
        self.new_attempt = False
        self.classical_ml = ClassicalML()
        self.players_interpolator = SimplePlayerInterpolator()
        self.player_track_cleaner_AI = PlayerTrackCleanerAI()
        # self.offside_detector = OffsideDetector(gpu=True)
        self.vid_num = 0
        
    
    def read_frames(self,frames_path):
        # Read the frames from the specified path
        frames_actions_tuples = []
        # frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                # extarct filename only without extension
                name = os.path.splitext(filename)[0]
                # Extract action part if it exists after the frame number
                parts = name.split("_")
                action = None
                # Check if filename has format like "frame_000592_dribble_E.jpg"
                if len(parts) > 2:
                    # Reconstruct the action by joining all parts after the frame number
                    action = "_".join(parts[2:])
                
                action_taken = None
                if action and action in action_list:
                    action_taken = action_list[action]
                else:
                    action_taken = None
                # return tuple frame and action_taken
                    
                frame = cv2.imread(frame_path)
                
                frames_actions_tuples.append((frame, action_taken))
                
                # frames.append(frame)
        # return frames
        return frames_actions_tuples
        

    def extract_action_tracks(self, all_tracks):
        """
        Extract only the tracks with non-None actions from all_tracks.
        
        Args:
            all_tracks (dict): Dictionary containing arrays of tracks for players, ball, etc.
                            The "action" key contains action identifiers.
        
        Returns:
            dict: Dictionary with the same structure as all_tracks but only containing frames with actions.
        """
        action_tracks = {
            "players": [],
            "referees": [],
            "ball": [],
            "goalkeeper": [],
            "goalpost": [],
            "action": []
        }
        
        # Get the indices where actions are not None
        action_indices = [i for i, action in enumerate(all_tracks["action"]) if action is not None]
        
        # Extract tracks at those indices, with safety checks
        for key in action_tracks.keys():
            if key in all_tracks:
                source_array = all_tracks[key]
                # Filter indices to ensure they're valid for this array
                valid_indices = [i for i in action_indices if i < len(source_array)]
                action_tracks[key] = [source_array[i] for i in valid_indices]
            else:
                action_tracks[key] = []  # Default to empty if key doesn't exist
        
        return action_tracks

    def run(self,video_frames):

        
        all_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : [],"action": []}
        # our_tracks = {"players": [], "referees": [], "ball": [] , "goalkeeper": [] , "goalpost" : [],"action": []}

        for test,(frame,action) in enumerate(video_frames):
            print("TEST: ", test)
            print("ACTION: ", action)
            print("Frame shape: ", frame.shape)
            start_time = time.time()
            height, width, _ = frame.shape
            frame_tracks = {"players": {}, "referees": {}, "ball": {}, "goalkeeper": {} , "goalpost" : {},"action": action}
            before_ball = time.time()
            frame_tracks['ball'] = self.tracker.get_ball_tracks(frame)
            # print("BALL TRACKS: ", frame_tracks['ball'])
            after_ball = time.time()
            # print("Ball detection time: ", after_ball - before_ball)
            before_player = time.time()
            frame_tracks["players"] = self.classical_ml.run(frame)
            after_player = time.time()
            # print("Player detection time: ", after_player - before_player)

            if frame_tracks['players']:
                self.tracker.remove_out_of_pitch_detections(frame_tracks["players"], height)
                frame_tracks['players'] , unmatched_det_ids  = self.tracker.ml_get_object_tracks(frame_tracks["players"])
                print("UNMATCHED IDS: ", unmatched_det_ids)
            #adjust goalkeeper detections
            

            # Replace Outliers
            # if frame_tracks["ball"]:
            # print("Before outliers: ", frame_tracks["ball"][1]["conf"] if frame_tracks["ball"] else None)
            # if frame_tracks["ball"] and not self.new_attempt: 
            #         frame_tracks["ball"] = self.tracker.replace_outliers_based_on_distance(
            #             frame_tracks["ball"], 218
            #         )
            # self.players_interpolator.remove_ids(unmatched_det_ids)
            frame_tracks['players'] = self.players_interpolator.interpolate_players(frame_tracks["players"])
            # interpolate again
            before_interpolate = time.time()
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
                    
            after_interpolate = time.time()
            # print("Ball interpolation time: ", after_interpolate - before_interpolate)
            # Get object positions
            before_get_positions = time.time()
            self.tracker.add_position_to_tracks(frame_tracks)
            after_get_positions = time.time()
            # print("Get positions time: ", after_get_positions - before_get_positions)

            # # View Trasnformer
            before_transform = time.time()
            self.view_transformer.get_matrix_transform(frame)
            self.view_transformer.add_transformed_position_to_tracks(frame_tracks)
            goal_post = self.view_transformer.real_world_to_pixel(self.tracker.goal_position)
            frame_tracks["goalpost"] = {
                1:{
                    "position_transformed": self.tracker.goal_position,
                    "position": goal_post
                }
            }
            
            # frame_tracks = self.tracker.adjust_goalkeeper_detection(frame_tracks)
            after_transform = time.time()
            print("Transform time: ", after_transform - before_transform)
            
            # handle penalty detections
            
            
            before_penalty = time.time()
            frame_tracks["ball"],changed = self.tracker.handle_penalty_detection(frame_tracks["ball"])
            if changed:
                print("PENALTY DETECTED")
                position = np.array(frame_tracks["ball"][1]["position"])
                position_trasnformed = self.view_transformer.pixel_to_real_world(position)
                if position_trasnformed is not None:
                    position_trasnformed = position_trasnformed.squeeze().tolist()
                # print("HANDLED" , position_trasnformed)
                frame_tracks['ball'][1]['position_transformed'] = position_trasnformed    
            
            
            after_penalty = time.time()
            print("Penalty detection time: ", after_penalty - before_penalty)

            before_assign = time.time()
            # Assign Ball to Player
            if frame_tracks["ball"]:
                    ball_position = frame_tracks["ball"][1]["bbox"]
                    assigned_player = self.player_assigner.assign_ball_to_player(
                        frame_tracks["players"], ball_position,frame_tracks["ball"][1],
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
                        before_offside = time.time()
                        # isOffside = self.offside_detector.detect_offside(frame[y1:y2, x1:x2])
                        # after_offside = time.time()
                        # print("Offside detection time: ", after_offside - before_offside)
                        
                        # if isOffside:

                        #     return None, None
                        
            after_assign = time.time()
            print("Assign time: ", after_assign - before_assign)
            print("-----------------------------------------------")
            print("   ")
            
                # print ("Ball position: ", frame_tracks["ball"][1]["position_transformed"])
            # if action is not None:
            #     our_tracks["players"].append(frame_tracks["players"])
            #     our_tracks["ball"].append(frame_tracks["ball"])
            #     our_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
            #     our_tracks["goalpost"].append(frame_tracks["goalpost"])
            #     our_tracks["action"].append(frame_tracks["action"])
            
                    # print("Action: ", frame_tracks["action"])
            
            all_tracks["players"].append(frame_tracks["players"])
            all_tracks["ball"].append(frame_tracks["ball"])
            all_tracks["goalkeeper"].append(frame_tracks["goalkeeper"])
            all_tracks["goalpost"].append(frame_tracks["goalpost"])
            all_tracks["action"].append(frame_tracks["action"])
        # print("DONEEEEEE" , time.time() - start_time)
        # print("ALL TRACKS: ", all_tracks)
        # Draw annotations on the video frames
            # print('all_tracks["players"]', all_tracks)
        # all_tracks["players"] = self.players_interpolator.maintain_tracks(all_tracks["players"])
        all_tracks["players"] = self.player_track_cleaner_AI.cleanup_player_tracks(all_tracks["players"])
        # our_tracks["players"] = self.players_interpolator.maintain_tracks(our_tracks["players"])     
            # save all tracks as json file
        # action_tracks = self.extract_action_tracks(all_tracks)
        # save all_tracks to json file
        # if not os.path.exists("output_videos"):
        #     os.makedirs("output_videos")
        # # save all_tracks to json file
        # def convert_ndarray(obj):
        #     if isinstance(obj, np.ndarray):
        #         return obj.tolist()
        #     elif isinstance(obj, dict):
        #         return {k: convert_ndarray(v) for k, v in obj.items()}
        #     elif isinstance(obj, list):
        #         return [convert_ndarray(v) for v in obj]
        #     else:
        #         return obj

        # Convert all_tracks to a JSON-serializable object
        # serializable_data = convert_ndarray(action_tracks)

        # Now write to file
        # with open("output_videos/all_tracks.json", "w") as f:
        #     json.dump(serializable_data, f, indent=4)
        
        # serializable_data_all = convert_ndarray(all_tracks)
        # with open("output_videos/all_tracks_all.json", "w") as f:
        #     json.dump(serializable_data_all, f, indent=4)
            
        # # Using zip with unpacking to separate frames and actions
        frames_only, _ = zip(*video_frames)    
        output_video_frames = self.tracker.draw_annotations(
            frames_only, all_tracks
        )
        cv2.imwrite(f"output_videos/ML_{self.vid_num}.jpg", output_video_frames[-1])
        self.vid_num += 1
        # cv2.imshow("Output Video", output_video_frames[-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if self.vid_num == 0:
        #     cv2.imwrite(f"output_videos/BADR_{self.vid_num}.png" , video_frames[-1])
        # cv2.imwrite(f"output_videos/ML_{self.vid_num}.png", output_video_frames[-1])
        # cv2.imwrite(f"output_videos/ML_org_{self.vid_num}.jpg",video_frames[-1])
        # self.vid_num += 1
        # save_video(output_video_frames, "controller/actions/ML_final.mp4")

        # # # Save video
        # save_video(output_video_frames, "output_videos/FINAL_CV_ISA.mp4")
                # Check if any player in team 1 has the ball
        hasBall = any(player.get("has_ball", False) and player.get("team") == 1 for player in frame_tracks["players"].values())
        # return also the biggest x for team 1 and biggest x for team 2 using frame_tracks["players"]

        # farthest player in team 1 without has_ball = true
        # farthest player in team 2 without has_ball = true


        action_tracks = 0
        return frame_tracks, hasBall, action_tracks
    
    def reset(self):
        """
        Reset the internal state of the ML_ComputerVision instance.
        This is useful for starting a new video or attempt.
        """
        self.tracker.reset_tracker()
        self.attempts = None
        self.new_attempt = False
        self.players_interpolator.reset()
        # self.vid_num = 0