from collections import defaultdict, deque
import numpy as np
import pandas as pd


class SimplePlayerInterpolator:
    def __init__(self, max_missing_frames=10, persistence_threshold=5):
        self.max_missing_frames = max_missing_frames
        self.persistence_threshold = persistence_threshold  # New parameter
        self.player_histories = defaultdict(deque)
        self.current_frame = 0
        self.last_seen_frame = defaultdict(int)
        self.players_to_remove = set()
        self.persistent_players = set()  # New set to track persistent players
        self.frames_detected = defaultdict(
            int
        )  # Track how many frames each player appeared in
        self.results_history = {}  # Dictionary to store previous results by frame number


    def interpolate_player_positions(self, players_tracks):
        """Interpolate player positions and maintain all player attributes"""
        self.current_frame += 1
        result = {}

        # Update histories for detected players
        for track_id, player in players_tracks.items():
            bbox = player.get("bbox", None)
            if bbox and len(bbox) == 4:
                # Store the full player data (including team, etc.)
                self.player_histories[track_id].append((self.current_frame, bbox))
                self.last_seen_frame[track_id] = self.current_frame
                self.frames_detected[track_id] += 1  # Increment detection count

                # Check if player has become persistent
                if self.frames_detected[track_id] >= self.persistence_threshold:
                    self.persistent_players.add(track_id)

                # Add complete player data to result
                result[track_id] = player
        self.results_history[self.current_frame] = result.copy()

        # Interpolate for missing players - check all known tracks, not just ones in current frame
        for track_id in list(self.player_histories.keys()):
            # Skip players already in result
            print(f"Checking track_id: {track_id}")
            print(f"Frames detected for {track_id}: {self.frames_detected[track_id]}")
            if track_id in result or self.frames_detected[track_id] <= 1:
                continue
                
            # Check if this track_id was last seen recently enough to interpolate
            frames_missing = self.current_frame - self.last_seen_frame[track_id]
            if frames_missing <= self.max_missing_frames:
                interpolated_bbox = self._interpolate_single_player(track_id)
                if interpolated_bbox is not None:
                    self.player_histories[track_id].append(
                        (self.current_frame, interpolated_bbox)
                    )
                    
                    # Get the last known data for this player from history
                    # We need to find the last full player data to preserve team info
                    last_player_data = self._get_last_player_data(track_id)
                    if last_player_data:
                        # Update only the bbox with interpolated values
                        last_player_data["bbox"] = interpolated_bbox
                        
                        # Add to result with all original attributes
                        result[track_id] = last_player_data

        # Cleanup
        self._cleanup_inactive_players()
        return result

    def _get_last_player_data(self, track_id):
        """Get the last known full player data for a track_id"""
        # Start with default bbox
        player_data = {"bbox": None, "team": 0}
        
        # Loop through previous results to find last complete data
        frame = self.current_frame - 1
        while frame > 0:
            # You'll need to store previous results somewhere
            # This is a placeholder - implement according to your storage
            prev_result = self._get_previous_result(frame)
            if prev_result and track_id in prev_result:
                return prev_result[track_id].copy()
            frame -= 1
            
        return player_data

    def _get_previous_result(self, frame):
        """Get the result from a previous frame"""
        return self.results_history.get(frame, None)

    def _interpolate_single_player(self, track_id):
        # Same as before
        history = self.player_histories[track_id]
        if len(history) < 2:
            return None

        frames = []
        bboxes = []

        for frame_num, bbox in history:
            frames.append(frame_num)
            bboxes.append(bbox)

        frames.append(self.current_frame)
        bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(
            {
                "frame": frames,
                "x1": [bbox[0] for bbox in bboxes],
                "y1": [bbox[1] for bbox in bboxes],
                "x2": [bbox[2] for bbox in bboxes],
                "y2": [bbox[3] for bbox in bboxes],
            }
        ).set_index("frame")

        df_interpolated = df.interpolate(method="linear")

        if self.current_frame in df_interpolated.index:
            interpolated_row = df_interpolated.loc[self.current_frame]
            if not interpolated_row.isna().any():
                return interpolated_row.tolist()
        return None

    def _cleanup_inactive_players(self):
        players_to_remove = [
            track_id
            for track_id in self.player_histories.keys()
            if (
                self.current_frame - self.last_seen_frame[track_id]
                > self.max_missing_frames
                and track_id not in self.persistent_players
            )  # Don't remove persistent players
        ]
        self.players_to_remove.update(players_to_remove)
        for track_id in players_to_remove:
            del self.player_histories[track_id]
            del self.last_seen_frame[track_id]
            if track_id in self.frames_detected:
                del self.frames_detected[track_id]

    def cleanup_all_tracks(self, all_players_tracks):
        """
        Remove players from players_tracks that are in self.players_to_remove set,
        except those that are persistent players

        Args:
            players_tracks: List of player dictionaries, one dictionary per frame
                        Each dictionary has player IDs as keys and player data as values

        Returns:
            The modified players_tracks with removed players
        """
        if not self.players_to_remove:
            return all_players_tracks  # Nothing to remove

        # Exclude persistent players from removal
        players_to_actually_remove = self.players_to_remove - self.persistent_players

        if not players_to_actually_remove:
            return all_players_tracks  # No non-persistent players to remove

        # For each frame's player dictionary in the sequence
        for frame_players in all_players_tracks:
            # Remove the players that are in the removal set
            for player_id in list(frame_players.keys()):
                if player_id in players_to_actually_remove:
                    del frame_players[player_id]

        # Clear the set after processing
        self.players_to_remove.clear()

        return all_players_tracks
    
    def remove_ids(self, ids_to_remove):
        """
        Remove specific player IDs from the histories and results.

        Args:
            ids_to_remove: Set of player IDs to remove
        """
        print(f"Removing player IDs: {ids_to_remove}")
        for track_id in ids_to_remove:
            if track_id in self.player_histories:
                del self.player_histories[1]
            if track_id in self.last_seen_frame:
                del self.last_seen_frame[1]
            if track_id in self.frames_detected:
                del self.frames_detected[1]
            if track_id in self.persistent_players:
                self.persistent_players.remove(1)
            

 
                
