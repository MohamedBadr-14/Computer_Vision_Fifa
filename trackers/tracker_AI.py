from typing import List, Dict, Set
import numpy as np


class PlayerTrackCleanerAI:

    def __init__(self):
        """
        Initialize the PlayerTrackCleanerAI class.
        This class is responsible for cleaning up player tracks to maintain consistent IDs across frames.
        """
        pass

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
        expected_players = (
            round(np.mean(num_players_per_scene)) if num_players_per_scene else 0
        )

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
            current_frame = self._immediate_remap_non_canonical(
                current_frame, canonical_ids
            )
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
                                historical_data = self._get_player_historical_data(
                                    player_tracks, missing_id, frame_idx
                                )

                                # Create new player data with interpolated position and bbox
                                new_player_data = {
                                    "team": prev_player_data[
                                        "team"
                                    ],  # Copy team from previous frame
                                    "track_id": missing_id,
                                    "has_ball": False,  # Default to no ball for missing players
                                }

                                # Interpolate position and bbox from historical data
                                if historical_data:
                                    interpolated_data = self._interpolate_player_data(
                                        historical_data, frame_idx
                                    )
                                    new_player_data.update(interpolated_data)
                                else:
                                    # Fallback to previous frame data if no historical data
                                    new_player_data["position"] = prev_player_data.get(
                                        "position", [0, 0]
                                    )
                                    new_player_data["position_transformed"] = (
                                        prev_player_data.get(
                                            "position_transformed", [0, 0]
                                        )
                                    )
                                    new_player_data["bbox"] = prev_player_data.get(
                                        "bbox", [0, 0, 0, 0]
                                    )

                                current_frame[missing_id] = new_player_data

            elif len(current_frame) > expected_players:
                # Too many players - remove excess detections
                current_frame = self._remove_excess_players(
                    current_frame, expected_players, canonical_ids
                )
                player_tracks[frame_idx] = current_frame

            # Note: The else case (correct number but wrong IDs) is now handled by _immediate_remap_non_canonical

        return player_tracks

    def _remove_excess_players(
        self, frame: Dict, expected_players: int, canonical_ids: Set[int]
    ) -> Dict:
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

    def _immediate_remap_non_canonical(
        self, frame: Dict, canonical_ids: Set[int]
    ) -> Dict:
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
                player_data["track_id"] = canonical_id
                remapped_frame[canonical_id] = player_data

        return remapped_frame  # This line was missing!

    def _get_player_historical_data(
        self, player_tracks: List[Dict], player_id: int, current_frame_idx: int
    ) -> List[Dict]:
        """
        Get historical data for a specific player from previous frames.
        """
        historical_data = []

        # Look back through previous frames to collect player data
        for frame_idx in range(
            max(0, current_frame_idx - 10), current_frame_idx
        ):  # Look back up to 10 frames
            frame = player_tracks[frame_idx]

            # Find player data in this frame
            for k in frame.keys():
                try:
                    if int(k) == player_id:
                        player_data = frame[k].copy()
                        player_data["frame_idx"] = frame_idx
                        historical_data.append(player_data)
                        break
                except (ValueError, TypeError):
                    continue

        return historical_data

    def _interpolate_player_data(
        self, historical_data: List[Dict], target_frame_idx: int
    ) -> Dict:
        """
        Interpolate player position and bbox from historical data.
        """
        if not historical_data:
            return {}

        if len(historical_data) == 1:
            # Only one data point, use it as is
            data = historical_data[0]
            return {
                "position": data.get("position", [0, 0]),
                "position_transformed": data.get("position_transformed", [0, 0]),
                "bbox": data.get("bbox", [0, 0, 0, 0]),
            }

        # Sort by frame index
        historical_data.sort(key=lambda x: x["frame_idx"])

        # Get the two most recent data points for interpolation
        recent_data = historical_data[-2:]

        if len(recent_data) >= 2:
            # Linear interpolation between two most recent points
            prev_data = recent_data[0]
            curr_data = recent_data[1]

            # Calculate time difference
            frame_diff = curr_data["frame_idx"] - prev_data["frame_idx"]
            target_diff = target_frame_idx - curr_data["frame_idx"]

            if frame_diff > 0:
                # Interpolation factor (extrapolate forward)
                t = target_diff / frame_diff

                # Interpolate position
                pos_interp = self._interpolate_coordinates(
                    prev_data.get("position", [0, 0]),
                    curr_data.get("position", [0, 0]),
                    t,
                )

                # Interpolate transformed position
                pos_trans_interp = self._interpolate_coordinates(
                    prev_data.get("position_transformed", [0, 0]),
                    curr_data.get("position_transformed", [0, 0]),
                    t,
                )

                # Interpolate bbox
                bbox_interp = self._interpolate_bbox(
                    prev_data.get("bbox", [0, 0, 0, 0]),
                    curr_data.get("bbox", [0, 0, 0, 0]),
                    t,
                )

                return {
                    "position": pos_interp,
                    "position_transformed": pos_trans_interp,
                    "bbox": bbox_interp,
                }

        # Fallback to most recent data
        most_recent = historical_data[-1]
        return {
            "position": most_recent.get("position", [0, 0]),
            "position_transformed": most_recent.get("position_transformed", [0, 0]),
            "bbox": most_recent.get("bbox", [0, 0, 0, 0]),
        }

    def _interpolate_coordinates(
        self, coord1: List[float], coord2: List[float], t: float
    ) -> List[int]:
        """
        Interpolate between two coordinate points.
        """
        if len(coord1) != 2 or len(coord2) != 2:
            return [0, 0]

        x = coord1[0] + t * (coord2[0] - coord1[0])
        y = coord1[1] + t * (coord2[1] - coord1[1])
        return [int(round(x)), int(round(y))]

    def _interpolate_bbox(
        self, bbox1: List[float], bbox2: List[float], t: float
    ) -> List[int]:
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
