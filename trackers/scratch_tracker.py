"""
Multi-Object Tracker for Soccer Players
======================================

A robust tracking system that handles YOLO detections and maintains persistent IDs
for soccer players through occlusions, missed detections, and complex movements.

Features:
- Kalman Filter motion prediction
- Hungarian algorithm for optimal assignment
- Robust handling of sports-specific movement patterns
- Automatic track initialization and termination
- ID persistence through re-entry

Author: AI Assistant
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import OrderedDict
import time


class KalmanTracker:
    """
    Kalman Filter for tracking bounding box center and velocity.
    
    State vector: [center_x, center_y, velocity_x, velocity_y]
    Measurement: [center_x, center_y]
    """
    
    def __init__(self, bbox, track_id):
        """
        Initialize Kalman filter with first detection.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            track_id: unique identifier for this track
        """
        self.track_id = track_id
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        
        # Convert bbox to center coordinates
        self.bbox = bbox
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        
        # Initialize Kalman filter
        # State: [center_x, center_y, velocity_x, velocity_y]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (motion uncertainty)
        # Higher values = expect more erratic movement
        process_noise = 6.0  # Tunable: increase for more erratic sports movement
        self.kf.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        
        # Measurement noise covariance (detection uncertainty)
        measurement_noise = 10.0  # Tunable: YOLO detection accuracy
        self.kf.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        
        # Initial state covariance
        self.kf.errorCovPost = 1000.0 * np.eye(4, dtype=np.float32)
        
        # Set initial state
        self.kf.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
    
    def predict(self):
        """
        Predict the next state using Kalman filter.
        Returns predicted bounding box.
        """
        # Advance the state
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Convert predicted center back to bounding box
        pred_center = self.kf.statePre[:2]
        return self._center_to_bbox(pred_center)
    
    def update(self, bbox):
        """
        Update the Kalman filter with a new detection.
        
        Args:
            bbox: [x1, y1, x2, y2] new detection
        """
        self.time_since_update = 0
        self.hit_streak += 1
        
        # Convert bbox to center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        measurement = np.array([center_x, center_y], dtype=np.float32)
        
        # Update Kalman filter
        self.kf.correct(measurement)
        
        # Store current bbox
        self.bbox = bbox
    
    def get_state(self):
        """
        Get current bounding box state.
        Returns: [x1, y1, x2, y2]
        """
        center = self.kf.statePost[:2]
        return self._center_to_bbox(center)
    
    def _center_to_bbox(self, center):
        """
        Convert center coordinates back to bounding box using last known size.
        """
        if hasattr(self, 'bbox') and self.bbox is not None:
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
        else:
            # Default size if no previous bbox
            w, h = 50, 100
        
        x1 = center[0] - w/2
        y1 = center[1] - h/2
        x2 = center[0] + w/2
        y2 = center[1] + h/2
        
        return [x1, y1, x2, y2]


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman filters and Hungarian algorithm.
    
    Handles track initialization, data association, and track termination.
    """
    
    def __init__(self, max_disappeared=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the multi-object tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            min_hits: Minimum consecutive hits before a track is considered valid
            iou_threshold: Minimum IoU for association (lower = more permissive)
        """
        self.max_disappeared = max_disappeared  # Tunable: track persistence
        self.min_hits = min_hits  # Tunable: track stability requirement
        self.iou_threshold = iou_threshold  # Tunable: association strictness
        
        self.trackers = OrderedDict()
        self.next_id = 0
        self.frame_count = 0
        self.track_teams = {} 
    
    def update(self, detections):
        """
        Update tracker with new detections , and enforce team-specific matching.

        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_id, det_id, team_id]

        Returns:
            tracks: List of [track_id, x1, y1, x2, y2]
            mapping: Dict mapping det_id -> track_id
        """
        self.frame_count += 1

        # Filter and extract det boxes & ids & teams
        bboxes, det_ids, det_teams = [], [], []
        for d in detections:
            if d[4] > 0.5:
                bboxes.append([d[0], d[1], d[2], d[3]])
                det_ids.append(d[6])
                det_teams.append(d[7])

        # Predict existing trackers
        predicted, to_del, pred_teams = [], [], []
        for tid, trk in self.trackers.items():
            if trk.time_since_update > self.max_disappeared:
                to_del.append(tid)
            else:
                predicted.append(trk.predict())
                pred_teams.append(self.track_teams[tid])
        for tid in to_del:
            del self.trackers[tid]
            del self.track_teams[tid]

        # Associate
        if predicted and bboxes:

            # print(f"\n--- Frame {self.frame_count} Debug ---")
            # print("Detections:")
            # for i, (bbox, det_id, team) in enumerate(zip(bboxes, det_ids, det_teams)):
            #     print(f"  Det[{i}] id={det_id}, team={team}, box={bbox}")

            # print("Predictions:")
            # for j, (pbox, tid, team) in enumerate(zip(predicted, track_ids, pred_teams)):
            #     print(f"  Pred[{j}] track={tid}, team={team}, box={pbox}")

            # build your IoU matrix exactly as you do:
            iou_mat = np.zeros((len(bboxes), len(predicted)))
            for i, det in enumerate(bboxes):
                for j, pr in enumerate(predicted):
                    raw_iou = self._compute_iou(det, pr)
                    # show when you zero it because of team mismatch
                    if det_teams[i] != pred_teams[j]:
                        # print(f"    IoU[{i},{j}] team mismatch (det {det_teams[i]} vs pred {pred_teams[j]}), set to 0")
                        iou_mat[i,j] = 0
                    else:
                        iou_mat[i,j] = raw_iou
                        # print(f"    IoU[{i},{j}] = {raw_iou:.3f}")

            cost = 1 - iou_mat
            # mimic your veto step:
            cost[iou_mat == 0] = 1e6

            # print("Cost matrix (1 - IoU, 1e6 vetoed):")
            # print(cost)

            row_ind, col_ind = linear_sum_assignment(cost)
            # print("Hungarian assignment results:")
            # print("  rows:", row_ind, "cols:", col_ind)

            matched, unmatched_d, unmatched_t = [], list(range(len(bboxes))), list(range(len(predicted)))
            for r, c in zip(row_ind, col_ind):
                valid = cost[r,c] < 1e6
                # print(f"  Pair (Det {r}, Pred {c}) cost={cost[r,c]:.1f} -> {'MATCH' if valid else 'VETO'}")
                if valid:
                    matched.append((r,c))
                    unmatched_d.remove(r)
                    unmatched_t.remove(c)

            # print("Matched pairs:", matched)
            # print("Unmatched detections indices:", unmatched_d)
            # print("Unmatched tracker indices:", unmatched_t)
            # print("--- end debug ---\n")
            unmatched_dets, unmatched_trks = unmatched_d, unmatched_t
        else:
            matched, unmatched_dets, unmatched_trks = [], list(range(len(bboxes))), list(range(len(predicted)))

        # Update matched
        mapping = {}
        if not self.trackers:
            # on the very first frame, consider everything matched (no true 'unmatched')
            unmatched_det_ids = []
        else:
            unmatched_det_ids = [det_ids[i] for i in unmatched_dets]
            
        track_ids = list(self.trackers.keys())
        for det_i, trk_i in matched:
            tid = track_ids[trk_i]
            self.trackers[tid].update(bboxes[det_i])
            mapping[det_ids[det_i]] = tid

        # Create new for unmatched detections
        for det_i in unmatched_dets:
            new_id = self.next_id
            self.trackers[new_id] = KalmanTracker(bboxes[det_i], new_id)
            self.track_teams[new_id] = det_teams[det_i]
            mapping[det_ids[det_i]] = new_id
            self.next_id += 1

        # Prepare output tracks
        tracks = []
        for tid, trk in self.trackers.items():
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                x1, y1, x2, y2 = trk.get_state()
                tracks.append([tid, int(x1), int(y1), int(x2), int(y2)])
        
        


        return tracks, mapping , unmatched_det_ids
    
    def _associate_detections_to_tracks(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using IoU and Hungarian algorithm.
        
        Returns:
            matched: list of (detection_idx, tracker_idx) pairs
            unmatched_detections: list of detection indices
            unmatched_trackers: list of tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._compute_iou(det, trk)
        
        # Use Hungarian algorithm for optimal assignment
        # Convert IoU to cost (higher IoU = lower cost)
        cost_matrix = 1 - iou_matrix
        
        # Only consider assignments above IoU threshold
        cost_matrix[iou_matrix < iou_threshold] = 1e6
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with cost too high (IoU too low)
        matched = []
        unmatched_detections = []
        unmatched_trackers = []
        
        for d in range(len(detections)):
            if d not in row_ind:
                unmatched_detections.append(d)
        
        for t in range(len(trackers)):
            if t not in col_ind:
                unmatched_trackers.append(t)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e6:  # Valid assignment
                matched.append([r, c])
            else:
                unmatched_detections.append(r)
                unmatched_trackers.append(c)
        
        return matched, unmatched_detections, unmatched_trackers
    
    def _compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format
        
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union


def demo_tracker():
    """
    Demonstration of how to use the tracker with YOLO detections.
    """
    
    # Initialize tracker
    tracker = MultiObjectTracker(
        max_disappeared=30,  # Keep tracks for 30 frames without detection
        min_hits=0,          # Require 3 consecutive hits to establish track
        iou_threshold=0.3    # IoU threshold for association
    )
    
    # Simulate YOLO detections for several frames
    # In real usage, you'd get these from your YOLO model
    sample_detections = [
        # Frame 0: 3 players detected
        [
            [100, 100, 150, 200, 0.9, 0],  # [x1, y1, x2, y2, confidence, class]
            [300, 150, 350, 250, 0.8, 0],
            [500, 120, 550, 220, 0.85, 0]
        ],
        # Frame 1: Same players, slightly moved
        [
            [105, 105, 155, 205, 0.9, 0],
            [305, 155, 355, 255, 0.8, 0],
            [505, 125, 555, 225, 0.85, 0]
        ],
        # Frame 2: One player occluded (missing)
        [
            [110, 110, 160, 210, 0.9, 0],
            [510, 130, 560, 230, 0.85, 0]
        ],
        # Frame 3: All players back
        [
            [115, 115, 165, 215, 0.9, 0],
            [315, 165, 365, 265, 0.8, 0],
            [515, 135, 565, 235, 0.85, 0]
        ],
        # Frame 4: New player enters
        [
            [120, 120, 170, 220, 0.9, 0],
            [320, 170, 370, 270, 0.8, 0],
            [520, 140, 570, 240, 0.85, 0],
            [200, 300, 250, 400, 0.9, 0]  # New player
        ]
    ]
    
    print("Multi-Object Tracker Demo")
    print("=" * 40)
    
    for frame_idx, detections in enumerate(sample_detections):
        print(f"\nFrame {frame_idx}:")
        print(f"Input detections: {len(detections)}")
        
        # Update tracker with detections
        tracks = tracker.update(detections)
        
        print(f"Active tracks: {len(tracks)}")
        for track in tracks:
            track_id, x1, y1, x2, y2 = track
            print(f"  Track {track_id}: bbox=({x1}, {y1}, {x2}, {y2})")


def visualize_tracking(video_path, yolo_detections_per_frame, output_path=None):
    """
    Visualize tracking results on video.
    
    Args:
        video_path: Path to input video
        yolo_detections_per_frame: List of detection lists, one per frame
        output_path: Optional path to save output video
    """
    # Initialize tracker
    tracker = MultiObjectTracker(
        max_disappeared=15,   # Shorter timeout for sports
        min_hits=1,          # Faster track establishment
        iou_threshold=0.1   # More permissive for fast movement
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer if path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color palette for different track IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame (if available)
        if frame_idx < len(yolo_detections_per_frame):
            detections = yolo_detections_per_frame[frame_idx]
        else:
            detections = []
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Draw bounding boxes and IDs
        for track in tracks:
            track_id, x1, y1, x2, y2 = track
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID: {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw frame info
        info_text = f"Frame: {frame_idx}, Tracks: {len(tracks)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save or display frame
        if out:
            out.write(frame)
        
        cv2.imshow('Multi-Object Tracking', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


# Hyperparameter tuning guide
TUNING_GUIDE = """
HYPERPARAMETER TUNING GUIDE
===========================

Key parameters to adjust based on your specific use case:

1. MultiObjectTracker Parameters:
   - max_disappeared (default: 30)
     * Higher: Keeps tracks longer during occlusions
     * Lower: Removes lost tracks faster, reduces ID switches
     * Soccer: 15-30 frames (fast-paced sport)
   
   - min_hits (default: 3)
     * Higher: More stable tracks, slower initialization
     * Lower: Faster track creation, more false positives  
     * Soccer: 2-5 hits
   
   - iou_threshold (default: 0.3)
     * Higher: Stricter association, fewer wrong matches
     * Lower: More permissive, handles fast movement better
     * Soccer: 0.2-0.4 (players move quickly)

2. KalmanTracker Parameters (in __init__):
   - process_noise (default: 4.0)
     * Higher: Expects more erratic movement
     * Lower: Assumes smoother motion
     * Soccer: 4.0-8.0 (unpredictable sports movement)
   
   - measurement_noise (default: 10.0)
     * Higher: Less trust in YOLO detections
     * Lower: More trust in detections
     * Depends on your YOLO model accuracy

3. Detection Filtering:
   - Confidence threshold (default: 0.5)
     * Adjust based on YOLO model performance
     * Soccer: 0.4-0.7 (balance detection vs false positives)

COMMON ISSUES & SOLUTIONS:
- Too many ID switches: Increase iou_threshold, decrease process_noise
- Lost tracks too quickly: Increase max_disappeared
- Slow track initialization: Decrease min_hits
- Tracks drift during occlusion: Adjust process_noise
"""

if __name__ == "__main__":
    # Run the demo
    demo_tracker()
    
    # Print tuning guide
    print("\n" + TUNING_GUIDE)
    
    # Example of how to use with real video (uncomment to use):
    # video_path = "soccer_match.mp4"
    # detections = []  # Load your YOLO detections here
    # visualize_tracking(video_path, detections, "tracked_output.mp4")