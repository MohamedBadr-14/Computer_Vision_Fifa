from collections import defaultdict, deque
import numpy as np
import pandas as pd


class SimplePlayerInterpolator:
    def __init__(self, maxMissingFrames=10, persistenceThresh=5):
        self.maxMissingFrames = maxMissingFrames
        self.persistenceThresh = persistenceThresh
        self.playersHistory = defaultdict(deque)
        self.currFrame = 0
        self.lastSeenFrame = defaultdict(int)
        self.playerIdsToRemove = set()
        self.playerIdsToKeep = set()
        self.detectedFrames = defaultdict(int)
        self.resultsHistory = {}

    def interpolate_players(self, players_tracks):
        self.currFrame += 1
        result = {}

        for track_id, player in players_tracks.items():
            bbox = player.get("bbox", None)
            if bbox and len(bbox) == 4:
                self.playersHistory[track_id].append((self.currFrame, bbox))
                self.lastSeenFrame[track_id] = self.currFrame
                self.detectedFrames[track_id] += 1

                if self.detectedFrames[track_id] >= self.persistenceThresh:
                    self.playerIdsToKeep.add(track_id)

                result[track_id] = player
        self.resultsHistory[self.currFrame] = result.copy()

        for track_id in list(self.playersHistory.keys()):
            if track_id in result or self.detectedFrames[track_id] <= 1:
                continue

            frames_missing = self.currFrame - self.lastSeenFrame[track_id]
            if frames_missing <= self.maxMissingFrames:
                interpolated_bbox = self._interpolate_single_player(track_id)
                if interpolated_bbox is not None:
                    self.playersHistory[track_id].append(
                        (self.currFrame, interpolated_bbox)
                    )

                    last_player_data = self._get_last_player_data(track_id)
                    if last_player_data:
                        result[track_id] = {
                            "bbox": interpolated_bbox,
                            "team": last_player_data.get("team", 0),
                        }

        self._remove_inactive_players()
        return result

    def _get_last_player_data(self, track_id):
        player_data = {"bbox": None, "team": 0}

        frame = self.currFrame - 1
        while frame > 0:
            prev_result = self._get_previous_result(frame)
            if prev_result and track_id in prev_result:
                return prev_result[track_id].copy()
            frame -= 1

        return player_data

    def _get_previous_result(self, frame):
        return self.resultsHistory.get(frame, None)

    def _interpolate_single_player(self, track_id):
        history = self.playersHistory[track_id]
        if len(history) < 2:
            return None

        frames = []
        bboxes = []

        for frame_num, bbox in history:
            frames.append(frame_num)
            bboxes.append(bbox)

        frames.append(self.currFrame)
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

        if self.currFrame in df_interpolated.index:
            interpolated_row = df_interpolated.loc[self.currFrame]
            if not interpolated_row.isna().any():
                return interpolated_row.tolist()
        return None

    def _remove_inactive_players(self):
        players_to_remove = [
            track_id
            for track_id in self.playersHistory.keys()
            if (
                self.currFrame - self.lastSeenFrame[track_id] > self.maxMissingFrames
                and track_id not in self.playerIdsToKeep
            )
        ]
        self.playerIdsToRemove.update(players_to_remove)
        for track_id in players_to_remove:
            del self.playersHistory[track_id]
            del self.lastSeenFrame[track_id]
            if track_id in self.detectedFrames:
                del self.detectedFrames[track_id]

    def maintain_tracks(self, all_players_tracks):
        if not self.playerIdsToRemove:
            return all_players_tracks

        players_to_actually_remove = self.playerIdsToRemove - self.playerIdsToKeep

        if not players_to_actually_remove:
            return all_players_tracks

        for frame_players in all_players_tracks:
            for player_id in list(frame_players.keys()):
                if player_id in players_to_actually_remove:
                    del frame_players[player_id]

        self.playerIdsToRemove.clear()

        return all_players_tracks

    def remove_ids(self, ids_to_remove):
        print(f"Removing player IDs: {ids_to_remove}")
        for track_id in ids_to_remove:
            if track_id in self.playersHistory:
                del self.playersHistory[track_id]
            if track_id in self.lastSeenFrame:
                del self.lastSeenFrame[track_id]
            if track_id in self.detectedFrames:
                del self.detectedFrames[track_id]
            if track_id in self.playerIdsToKeep:
                self.playerIdsToKeep.remove(track_id)

    def reset(self):
        self.playersHistory.clear()
        self.currFrame = 0
        self.lastSeenFrame.clear()
        self.playerIdsToRemove.clear()
        self.playerIdsToKeep.clear()
        self.detectedFrames.clear()
        self.resultsHistory.clear()
