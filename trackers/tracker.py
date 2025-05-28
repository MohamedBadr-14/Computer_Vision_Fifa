from ultralytics import YOLO
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from .scratch_tracker import MultiObjectTracker
import pandas as pd
import cv2
import sys
from ball_detector.ball_detect import BallDetect
from typing import List, Dict, Any, Set, Tuple

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position,measure_distance


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.ml_tracker = DeepSort(max_age=30)  # max_age: how many frames to keep a track alive without update
        self.multi_object_tracker = MultiObjectTracker(
            min_hits=0,
            max_disappeared= 10,
            iou_threshold=0.2,
        )
        self.deep_ball_model = YOLO('models/best_initial.pt')
        self.ball_model = BallDetect()
        self.ball_positions_history = []
        self.last_valid_position = None
        self.frame_num = 0
        self.max_player_post_distance = 1000
        self.goal_position = (60,42)

    def reset_tracker(self):
        self.ball_positions_history = []
        self.last_valid_position = None
        self.frame_num = 0    

    def add_position_to_tracks(self, frame_tracks):
        for object_type, object_tracks in frame_tracks.items():
            print("OBJECT TYPE", object_type)
            print("OBJECT TRACKS", object_tracks)
            if object_type == "action":
                continue
            for track_id, track_info in object_tracks.items():
                bbox = track_info["bbox"]
                if object_type == "ball":
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                frame_tracks[object_type][track_id]["position"] = position
    

    def adjust_goalkeeper_detection(self, frame_tracks):
        # if frame_tracks["goalpost"][1]["position"][0] < 2560:
            ## print("DKHALT")
            
        post_position = frame_tracks["goalpost"][1]["position"]

        miniumum_distance = 99999
        goalkeeper=-1

        for player_id, player in frame_tracks["players"].items():
            player_pos = player['position']

            distance = measure_distance(player_pos,post_position)

            if distance < miniumum_distance:
                if distance < self.max_player_post_distance:
                    miniumum_distance = distance
                    goalkeeper = player_id
                    ## print("FRAME_TRACKS" , frame_tracks["players"])
        if goalkeeper != -1:
            frame_tracks["goalkeeper"][goalkeeper] = {
                "bbox": frame_tracks["players"][goalkeeper]["bbox"],
                "conf": None, #BADRR
                "position": frame_tracks["players"][goalkeeper]["position"],
                "position_transformed": frame_tracks["players"][goalkeeper]["position_transformed"],
            }
            del frame_tracks["players"][goalkeeper] 
        return frame_tracks
    
    def remove_out_of_pitch_detections(self,player_tracks , height):
        to_be_deleted = []
        for player_id, player in player_tracks.items():
            bbox = player["bbox"]
            y = bbox[1]
            if y < 0.1*height:
                # Remove the player from the track if they are out of the pitch
                to_be_deleted.append(player_id)
        for player_id in to_be_deleted:
            del player_tracks[player_id]

    def interpolate_ball_positions(self, ball_position):
        self.ball_positions_history.append(ball_position.get(1, {}).get("bbox", []))

        if not self.ball_positions_history or all(
            len(bbox) == 0 for bbox in self.ball_positions_history
        ):
            return ball_position

        # Convert the ball positions history to a DataFrame
        df_ball_positions = pd.DataFrame(
            self.ball_positions_history, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = (
            df_ball_positions.bfill()
        )  # Backward fill for any remaining NaNs

        # Get the interpolated ball position for the current frame
        interpolated_bbox = df_ball_positions.iloc[-1].tolist()

        # Format the result
        # interpolated_ball_position = {1: {"bbox": interpolated_bbox}}
        self.last_valid_position = interpolated_bbox[0:2]
        return interpolated_bbox

    def detect_frame(self, frame):
        # Process a single frame
        detection = self.model(frame , verbose = False) #BADR MIDO
        detection_ball = self.deep_ball_model(frame , conf = 0.1,verbose = False) #BADR MIDO
        return detection,detection_ball
    
    def ml_get_object_tracks(self, players):

        detections = []
        player_id_list = []  # to track which detection came from which player
        ## print("PLAYERS", players)
        # ow("frame", frame)
        # cv2.waitKey(1)  # Allow OpenCV to process the frame
        # cv2.destroyAllWindocv2.imshws()
        for player_id, player in players.items():
            bbox = player["bbox"]
            conf = player.get("conf", 0.99)
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, '0' , player_id , player.get("team", 0)])

        # detections = np.array(detections)
        ## print("DETECTIONS", detections)
        unmacthed_ids = []
        tracks,mapping,unmacthed_ids = self.multi_object_tracker.update(detections)
        reverse_map = {track_id: det_id  for det_id , track_id in mapping.items()}

        ## print("deep stracks", tracks)
        # [track_id, x1, y1, x2, y2],
        # [track_id, x1, y1, x2, y2],

        new_players = {}
        for track in tracks:
            ## print(vars(track))
            ## print("TRACK info", track.is_confirmed(), track[0],track.original_ltwh)
            # if not track.is_confirmed():
            #     continue
            original_id = None
            original_id = reverse_map.get(track[0], None)
            # for id,player in players.items():
            #     if track == []:
            #        # print("track.original_ltwh is None")
            #         continue
            #     ## print("PLAYER BBOX", player['bbox'])

            # if det_idx >= len(player_id_list):
            #     continue  # safety check
           # print("TRACK ID", track)
           # print("detidc", original_id)
            if original_id is None:
               # print("Original ID not found for track", track[0])
                continue
            old_player = players[original_id]
            new_players[track[0]] = old_player
            new_players[track[0]]["track_id"] = track[0]  # Optional: store new ID
       # print("TRACKS", new_players)

        return new_players , unmacthed_ids

    
    
    
    def get_object_track(self, frame):

        # Detect objects in the current frame+
        detection,detection_ball = self.detect_frame(frame)
        ball_count = 0

        # Initialize tracks for the current frame
        frame_tracks = {"players": {},"ball": {} , "goalkeeper": {} , "goalpost": {}}

        cls_names = detection[0].names
        cls_names_ball = detection_ball[0].names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        cls_names_inv_ball = {v: k for k, v in cls_names_ball.items()}

        # Convert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection[0])
        
        detections_nms = detection_supervision.with_nms(threshold=0.5, class_agnostic=True)
        detection_supervision_ball = sv.Detections.from_ultralytics(detection_ball[0])

        for i,frame_detection in enumerate(detections_nms):
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = i
            conf = frame_detection[2]
            

            if cls_id == cls_names_inv["player"] or cls_id == cls_names_inv["referee"] or cls_id == cls_names_inv["goalkeeper"]:
                if conf >=0.25:
                    frame_tracks["players"][track_id] = {"bbox": bbox, "conf": conf}
                
        for i,frame_detection in enumerate(detection_supervision_ball):
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = i
            conf = frame_detection[2]
            if cls_id == cls_names_inv_ball["ball"]:
                if (
                    ball_count > 0
                    and frame_detection[2] < frame_tracks["ball"][1]["conf"]
                ):
                    # If the ball is already detected, skip adding it again
                    continue
                frame_tracks["ball"][1] = {"conf": frame_detection[2], "bbox": bbox}
                ## print("BALLLL CONFFF" , frame_detection[2])
                ball_count += 1
 
        return frame_tracks
   
    def get_ball_tracks(self,frame):
        ball_tracks = {}
        if self.last_valid_position is not None:
            self.ball_model.last_ball_position = self.last_valid_position   
        ball_bbox = self.ball_model.svm_predict(frame,1)
        if ball_bbox is not None:
            ball_tracks[1] = {
                "conf": 0.11,
                "bbox": ball_bbox,
            }
        else:
            ball_tracks = {}

           
        return ball_tracks
    
    def scale_bouding_box(self, bbox):
        scale_x = 2560 / 640  
        scale_y = 1600 / 640
        x_min, y_min, x_max, y_max = bbox
        scaled_box = [
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y
        ]
        return scaled_box
        
                    
    def replace_outliers_based_on_distance(self, ball_track, distance_threshold: float):
        position = ball_track.get(1, {}).get("bbox", [])
        if not position:
            # If the current position is already empty, just add it to the cleaned positions
            ## print("Empty ball")
            return {}
        else:
            if self.last_valid_position is None:
                # If there's no valid last position, accept the first valid one
                self.last_valid_position = position[0:2]
                return ball_track
            else:
                # Calculate the distance from the last valid position
                # #print("position", position)
                # #print("last_valid_position", last_valid_position)
                distance = np.linalg.norm(
                    np.array(position[0:2]) - np.array(self.last_valid_position)
                )
                ## print("distance", distance)
                if distance > distance_threshold:
                    # Replace with empty array if the distance exceeds the threshold
                   # print("Distance exceeded threshold\n" , distance)
                    return {}
                else:
                    self.last_valid_position = position[0:2]
                    return ball_track
                
    def handle_penalty_detection(self, ball_tracks):
        changed = False
        if ball_tracks and 1 in ball_tracks:
            x,y = ball_tracks[1]['position_transformed'][0], ball_tracks[1]['position_transformed'][1]
            if 48 <= x <= 50 and 44 <= y <= 46:
               # print("Penalty detected")
                ball_tracks = {}
                self.ball_positions_history.pop() if self.ball_positions_history else None
                bbox = self.interpolate_ball_positions(ball_tracks)
                if bbox:
                    ball_tracks = {
                        1: {
                            "bbox": bbox,
                            "conf": 0.11
                        }
                    }  
                    ball_tracks[1]["position"] = get_center_of_bbox(bbox)
                    changed = True
        return ball_tracks,changed


    def cleanup_player_tracks(self, player_tracks: List[Dict]) -> List[Dict]:
        """
        Clean up player tracks to maintain consistent IDs across frames.
        
        Args:
            player_tracks: List of dictionaries, each containing player data for a frame
            
        Returns:
            Cleaned player tracks with consistent IDs
        """
        if not player_tracks:
            return player_tracks
        
        # Filter out None frames and calculate expected number of players
        valid_frames = [frame for frame in player_tracks if frame is not None]
        num_players_per_scene = [len(scene) for scene in valid_frames]
        expected_players = round(np.mean(num_players_per_scene)) if num_players_per_scene else 0
        
        if expected_players == 0:
            return player_tracks
        
        # Initialize canonical ID set (0 to expected_players-1)
        canonical_ids = set(range(expected_players))
        
        # Process each frame
        for frame_idx, current_frame in enumerate(player_tracks):
            # Skip None frames
            if current_frame is None:
                continue
                
            # FIRST: Immediately remap any non-canonical IDs to prevent them from appearing
            current_frame = self._immediate_remap_non_canonical(current_frame, canonical_ids)
            player_tracks[frame_idx] = current_frame
            
            # Handle both string and integer keys
            current_ids = set()
            for k in current_frame.keys():
                try:
                    current_ids.add(int(k))
                except (ValueError, TypeError):
                    continue
            
            if len(current_frame) == expected_players and current_ids == canonical_ids:
                # Perfect frame - no changes needed
                continue
                
            elif len(current_frame) < expected_players:
                # Missing players - restore with team from previous frame and interpolated position/bbox
                missing_ids = canonical_ids - current_ids
                if frame_idx > 0:
                    prev_frame = player_tracks[frame_idx - 1]
                    # Make sure prev_frame is not None
                    if prev_frame is not None:
                        for missing_id in missing_ids:
                            # Check if missing_id exists in previous frame (handle both key types)
                            prev_player_data = None
                            for k in prev_frame.keys():
                                try:
                                    if int(k) == missing_id:
                                        prev_player_data = prev_frame[k]
                                        break
                                except (ValueError, TypeError):
                                    continue
                            
                            if prev_player_data:
                                # Get historical data for this player
                                historical_data = self._get_player_historical_data(player_tracks, missing_id, frame_idx)
                                
                                # Create new player data with interpolated position and bbox
                                new_player_data = {
                                    'team': prev_player_data['team'],  # Copy team from previous frame
                                    'track_id': missing_id,
                                    'has_ball': False  # Default to no ball for missing players
                                }
                                
                                # Interpolate position and bbox from historical data
                                if historical_data:
                                    interpolated_data = self._interpolate_player_data(historical_data, frame_idx)
                                    new_player_data.update(interpolated_data)
                                else:
                                    # Fallback to previous frame data if no historical data
                                    new_player_data['position'] = prev_player_data.get('position', [0, 0])
                                    new_player_data['position_transformed'] = prev_player_data.get('position_transformed', [0, 0])
                                    new_player_data['bbox'] = prev_player_data.get('bbox', [0, 0, 0, 0])
                                
                                current_frame[missing_id] = new_player_data
                                
            elif len(current_frame) > expected_players:
                # Too many players - remove excess detections
                current_frame = self._remove_excess_players(current_frame, expected_players, canonical_ids)
                player_tracks[frame_idx] = current_frame
                
            # Note: The else case (correct number but wrong IDs) is now handled by _immediate_remap_non_canonical
        
        return player_tracks

    def _remove_excess_players(self, frame: Dict, expected_players: int, canonical_ids: Set[int]) -> Dict:
        """
        Remove excess players from a frame, prioritizing keeping canonical IDs.
        """
        # Handle both string and integer keys
        current_ids = set()
        key_to_id_map = {}
        for k in frame.keys():
            try:
                player_id = int(k)
                current_ids.add(player_id)
                key_to_id_map[player_id] = k
            except (ValueError, TypeError):
                continue
        
        # Keep players with canonical IDs first
        canonical_present = current_ids & canonical_ids
        non_canonical_present = current_ids - canonical_ids
        
        # If we have more canonical IDs than expected, keep the lowest ones
        if len(canonical_present) >= expected_players:
            ids_to_keep = sorted(canonical_present)[:expected_players]
        else:
            # Keep all canonical IDs and fill with non-canonical ones
            ids_to_keep = list(canonical_present)
            remaining_slots = expected_players - len(canonical_present)
            ids_to_keep.extend(sorted(non_canonical_present)[:remaining_slots])
        
        # Create new frame with only kept players
        cleaned_frame = {}
        for player_id in ids_to_keep:
            original_key = key_to_id_map[player_id]
            cleaned_frame[player_id] = frame[original_key]
        
        return cleaned_frame

    def _immediate_remap_non_canonical(self, frame: Dict, canonical_ids: Set[int]) -> Dict:
        """
        Immediately remap any non-canonical IDs to canonical ones to prevent non-canonical IDs from appearing.
        """
        # Handle both string and integer keys
        current_ids = set()
        key_to_id_map = {}
        for k in frame.keys():
            try:
                player_id = int(k)
                current_ids.add(player_id)
                key_to_id_map[player_id] = k
            except (ValueError, TypeError):
                continue
        
        non_canonical_ids = current_ids - canonical_ids
        
        if not non_canonical_ids:
            return frame  # All IDs are already canonical
        
        # Find available canonical IDs
        available_canonical = canonical_ids - current_ids
        
        if not available_canonical:
            # No available canonical IDs - remove excess players
            return self._remove_excess_players(frame, len(canonical_ids), canonical_ids)
        
        # Remap non-canonical IDs to available canonical IDs
        remapped_frame = {}
        
        # Keep all canonical IDs as they are
        for player_id in current_ids & canonical_ids:
            original_key = key_to_id_map[player_id]
            remapped_frame[player_id] = frame[original_key]
        
        # Remap non-canonical IDs to available canonical IDs
        non_canonical_sorted = sorted(non_canonical_ids)
        available_canonical_sorted = sorted(available_canonical)
        
        for i, non_canonical_id in enumerate(non_canonical_sorted):
            if i < len(available_canonical_sorted):
                canonical_id = available_canonical_sorted[i]
                original_key = key_to_id_map[non_canonical_id]
                
                # Copy player data and update track_id
                player_data = frame[original_key].copy()
                player_data['track_id'] = canonical_id
                remapped_frame[canonical_id] = player_data
        
        return remapped_frame  # This line was missing!
    def _get_player_historical_data(self, player_tracks: List[Dict], player_id: int, current_frame_idx: int) -> List[Dict]:
        """
        Get historical data for a specific player from previous frames.
        """
        historical_data = []
        
        # Look back through previous frames to collect player data
        for frame_idx in range(max(0, current_frame_idx - 10), current_frame_idx):  # Look back up to 10 frames
            frame = player_tracks[frame_idx]
            
            # Find player data in this frame
            for k in frame.keys():
                try:
                    if int(k) == player_id:
                        player_data = frame[k].copy()
                        player_data['frame_idx'] = frame_idx
                        historical_data.append(player_data)
                        break
                except (ValueError, TypeError):
                    continue
        
        return historical_data

    def _interpolate_player_data(self, historical_data: List[Dict], target_frame_idx: int) -> Dict:
        """
        Interpolate player position and bbox from historical data.
        """
        if not historical_data:
            return {}
        
        if len(historical_data) == 1:
            # Only one data point, use it as is
            data = historical_data[0]
            return {
                'position': data.get('position', [0, 0]),
                'position_transformed': data.get('position_transformed', [0, 0]),
                'bbox': data.get('bbox', [0, 0, 0, 0])
            }
        
        # Sort by frame index
        historical_data.sort(key=lambda x: x['frame_idx'])
        
        # Get the two most recent data points for interpolation
        recent_data = historical_data[-2:]
        
        if len(recent_data) >= 2:
            # Linear interpolation between two most recent points
            prev_data = recent_data[0]
            curr_data = recent_data[1]
            
            # Calculate time difference
            frame_diff = curr_data['frame_idx'] - prev_data['frame_idx']
            target_diff = target_frame_idx - curr_data['frame_idx']
            
            if frame_diff > 0:
                # Interpolation factor (extrapolate forward)
                t = target_diff / frame_diff
                
                # Interpolate position
                pos_interp = self._interpolate_coordinates(
                    prev_data.get('position', [0, 0]),
                    curr_data.get('position', [0, 0]),
                    t
                )
                
                # Interpolate transformed position
                pos_trans_interp = self._interpolate_coordinates(
                    prev_data.get('position_transformed', [0, 0]),
                    curr_data.get('position_transformed', [0, 0]),
                    t
                )
                
                # Interpolate bbox
                bbox_interp = self._interpolate_bbox(
                    prev_data.get('bbox', [0, 0, 0, 0]),
                    curr_data.get('bbox', [0, 0, 0, 0]),
                    t
                )
                
                return {
                    'position': pos_interp,
                    'position_transformed': pos_trans_interp,
                    'bbox': bbox_interp
                }
        
        # Fallback to most recent data
        most_recent = historical_data[-1]
        return {
            'position': most_recent.get('position', [0, 0]),
            'position_transformed': most_recent.get('position_transformed', [0, 0]),
            'bbox': most_recent.get('bbox', [0, 0, 0, 0])
        }

    def _interpolate_coordinates(self, coord1: List[float], coord2: List[float], t: float) -> List[int]:
        """
        Interpolate between two coordinate points.
        """
        if len(coord1) != 2 or len(coord2) != 2:
            return [0, 0]
        
        x = coord1[0] + t * (coord2[0] - coord1[0])
        y = coord1[1] + t * (coord2[1] - coord1[1])
        return [int(round(x)), int(round(y))]

    def _interpolate_bbox(self, bbox1: List[float], bbox2: List[float], t: float) -> List[int]:
        """
        Interpolate between two bounding boxes.
        """
        if len(bbox1) != 4 or len(bbox2) != 4:
            return [0, 0, 0, 0]
        
        x1 = bbox1[0] + t * (bbox2[0] - bbox1[0])
        y1 = bbox1[1] + t * (bbox2[1] - bbox1[1])
        x2 = bbox1[2] + t * (bbox2[2] - bbox1[2])
        y2 = bbox1[3] + t * (bbox2[3] - bbox1[3])
        
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
      

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            # if track_id > 99:
            #     x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, center, color):
        # y = int(bbox[1])
        # x, _ = get_center_of_bbox(bbox)
        x,y = center[0], center[1]
        if x > 2581:
            return frame

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ],
            dtype=np.int32  # Ensure the points are integers

        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1*100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2*100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        # #print("len(video_frames)", len(video_frames))
        ## print("len(tracks['players'])", len(tracks['players']))
        ## print("len(tracks['ball'])", len(tracks['ball']))
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
           
            
            cv2.putText(
                frame,
                f"Frame: {frame_num}",
                (50, 50),  # Position (x=50, y=50)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                1,  # Font scale (adjust as needed)
                (0, 0, 255),  # Color (red in BGR)
                2,  # Thickness
                cv2.LINE_AA  # Line type
            ) 

            player_dict = tracks["players"][frame_num]
            ball_dict = (
                tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            )
            goal_keeper_dict = tracks["goalkeeper"][frame_num] if frame_num < len(tracks["goalkeeper"]) else {}
            goalpost_dict = tracks["goalpost"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                team = player.get("team")
                color = (0, 255, 0) if team == 1 else (255, 0, 0)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["position"], (0, 0, 255))
            
            for _, goalkeeper in goal_keeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 0, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["position"], (0, 255, 0))

            # Draw Goalpost
            for _, goalpost in goalpost_dict.items():
                frame = self.draw_traingle(frame, goalpost["position"], (255, 0, 0))

            # Draw Team Ball Control
            # frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    