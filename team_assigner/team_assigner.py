from sklearn.cluster import KMeans
import numpy as np
import cv2



class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_confidence = {}
        

    def get_clustering_model(self, hue_2d):
        # Ensure dtype is float32 for sklearn
        hue_2d = hue_2d.astype(np.float32)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(hue_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Crop image to bbox
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Take only the top half of the player box
        top_half_image = image[:image.shape[0] // 2, :]
        image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)

        hue_channel = image_hsv[:, :, 0]  # Get Hue only
        hue_2d = hue_channel.reshape(-1, 1)  # Flatten to 2D array
        
        # Fit KMeans on hue only
        kmeans = self.get_clustering_model(hue_2d)

        # Predict cluster labels
        labels = kmeans.labels_
        clustered_image = labels.reshape(hue_channel.shape)  # Reshape back to image shape

        # Determine non-player cluster based on corners
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Choose player cluster
        assignees_count = np.bincount(labels)
        assignees_count[non_player_cluster] = -1
        player_cluster = np.argmax(assignees_count)
        player_hue = kmeans.cluster_centers_[player_cluster][0]  # Get hue value

        return player_hue

    def assign_team_color(self, frame, player_detections):
        player_hues = []
        for _, detection in player_detections.items():
            bbox = detection["bbox"]
            hue = self.get_player_color(frame, bbox)
            player_hues.append([hue])  # Wrap as list for KMeans shape

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(np.array(player_hues, dtype=np.float32))
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0][0]
        self.team_colors[2] = kmeans.cluster_centers_[1][0]
        print("Team 1 color (hue):", self.team_colors[1])
        print("Team 2 color (hue):", self.team_colors[2])


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict and self.player_team_confidence.get(player_id, 0) > 0.7:
            print("DKHAL:", player_id , "Team ID:", self.player_team_dict[player_id])
            return self.player_team_dict[player_id]
        print("Player ID:", player_id , "Team ID:", self.player_team_dict.get(player_id, None))

        hue = self.get_player_color(frame, player_bbox)
        hue_input = np.array([[hue]], dtype=np.float32)  # Correct shape and dtype

        distances = self.kmeans.transform(hue_input)
        team_id = self.kmeans.predict(hue_input)[0]

        max_distance = np.max(distances)
        normalized_confidence = 1 - (distances[0, team_id] / max_distance)

        team_id += 1  # Team IDs are 1-based

        self.player_team_dict[player_id] = team_id
        self.player_team_confidence[player_id] = normalized_confidence

        return team_id


    