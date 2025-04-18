from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_confidence = {}
        self.first_ten_team_colors = {}
        self.counter = 0

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[
            int(image.shape[0] * 0.15) : int(image.shape[0] * 0.6), :
        ]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Get the player cluster
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # player_cluster = 1-non_player_cluster
        # ##print(player_cluster)
        # Find the cluster with the maximum number of assignees that is not the non_player_cluster
        yellow_color = np.array([0, 255, 255])  # Example yellow color in BGR
        blue_color = np.array([255, 0, 0])  # Example blue color in BGR
        # Calculate the distances to yellow and blue for all clusters except the non_player_cluster
        remaining_clusters = [
            i for i in range(len(kmeans.cluster_centers_)) if i != non_player_cluster
        ]
        distances_to_yellow = [
            np.linalg.norm(kmeans.cluster_centers_[i] - yellow_color)
            for i in remaining_clusters
        ]
        distances_to_blue = [
            np.linalg.norm(kmeans.cluster_centers_[i] - blue_color)
            for i in remaining_clusters
        ]
        # print(kmeans.cluster_centers_,'cluster centers')
        # #print("distances_to_yellow", distances_to_yellow)
        # #print("distances_to_blue", distances_to_blue)
        # Find the cluster with the minimum distance to either yellow or blue
        min_distance_cluster = remaining_clusters[
            np.argmin(
                [
                    min(distances_to_yellow[i], distances_to_blue[i])
                    for i in range(len(remaining_clusters))
                ]
            )
        ]
        # #print("min_distance_cluster", min_distance_cluster)
        player_cluster = min_distance_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        # print(player_color, 'player color')
        # cv2.namedWindow(f'{player_color}', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(f'{player_color}', int(cv2.WND_PROP_FULLSCREEN/2), int(cv2.WINDOW_FULLSCREEN/2))
        # cv2.imshow(f'{player_color}', top_half_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return player_color

    def assign_team_color(self, frame, player_detections, color_tolerance=30):

        # Store the first 10 cluster centers
        # draw bounding box over frame and save it in a folder TODO
       
        
        if self.counter < 10:
            print("player_detections", player_detections)

            player_colors = []
            if len(player_detections) < 2:
                self.team_colors[1] = np.array([0, 255, 255]) 
                self.team_colors[2] = np.array([255, 0, 0]) 
                return

            for _, player_detection in player_detections.items():
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
            # #print("player_colors", player_colors)

            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(player_colors)

            self.kmeans = kmeans
            # Define reference colors for teams
            # Team 1: Yellow (BGR format)
            # Team 2: Blue (BGR format)
            yellow_color = np.array([0, 255, 255])  # Example yellow color in BGR

            # Calculate distances to reference colors
            distances_to_team_1 = [
                np.linalg.norm(center - yellow_color)
                for center in kmeans.cluster_centers_
            ]

            # Assign cluster centers to teams based on proximity to reference colors
            if distances_to_team_1[0] < distances_to_team_1[1]:
                self.team_colors[1] = kmeans.cluster_centers_[0]
                self.team_colors[2] = kmeans.cluster_centers_[1]
            else:
                self.team_colors[1] = kmeans.cluster_centers_[1]
                self.team_colors[2] = kmeans.cluster_centers_[0]

            # Store the first 10 cluster centers
            if 1 not in self.first_ten_team_colors:
                self.first_ten_team_colors[1] = []
            if 2 not in self.first_ten_team_colors:
                self.first_ten_team_colors[2] = []

            self.first_ten_team_colors[1].append(self.team_colors[1])
            self.first_ten_team_colors[2].append(self.team_colors[2])
        elif self.counter == 10:
            # After the first 10 frames, calculate the most repeated center within a range
            team_1_colors = np.array(self.first_ten_team_colors[1])
            team_2_colors = np.array(self.first_ten_team_colors[2])

            def find_most_repeated_color(colors):
                grouped_colors = []
                for color in colors:
                    found_group = False
                    for group in grouped_colors:
                        if np.linalg.norm(color - group[0]) <= color_tolerance:
                            group.append(color)
                            found_group = True
                            break
                    if not found_group:
                        grouped_colors.append([color])
                most_repeated_group = max(grouped_colors, key=len)
                return np.mean(most_repeated_group, axis=0)

            self.team_colors[1] = find_most_repeated_color(team_1_colors)
            self.team_colors[2] = find_most_repeated_color(team_2_colors)

        self.counter += 1

    # def assign_team_color(self, frame, player_detections, color_tolerance=30):
    #     # Store the first 10 cluster centers
    #     # Draw bounding boxes over frame and save it in a folder
        
    #     # Create output directory if it doesn't exist
    #     import os
    #     output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "team_assignment_debug")
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Create a copy of the frame to draw on
    #     debug_frame = frame.copy()
        
    #     if self.counter < 10:
    #         print("player_detections", player_detections)
    
    #         player_colors = []
    #         for player_id, player_detection in player_detections.items():
    #             bbox = player_detection["bbox"]
    #             player_color = self.get_player_color(frame, bbox)
    #             player_colors.append(player_color)
                
    #             # Draw bounding box with player ID and extracted color
    #             x1, y1, x2, y2 = [int(c) for c in bbox]
    #             cv2.rectangle(debug_frame, (x1, y1), (x2, y2), player_color.tolist(), 2)
                
    #             # Draw player ID and color values
    #             color_text = f"ID:{player_id} BGR:{int(player_color[0])},{int(player_color[1])},{int(player_color[2])}"
    #             cv2.putText(debug_frame, color_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
    #         # Proceed with KMeans clustering as before
    #         kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
    #         kmeans.fit(player_colors)
    
    #         self.kmeans = kmeans
    #         # Define reference colors for teams
    #         # Team 1: Yellow (BGR format)
    #         # Team 2: Blue (BGR format)
    #         yellow_color = np.array([0, 255, 255])  # Example yellow color in BGR
    
    #         # Calculate distances to reference colors
    #         distances_to_team_1 = [
    #             np.linalg.norm(center - yellow_color)
    #             for center in kmeans.cluster_centers_
    #         ]
    
    #         # Assign cluster centers to teams based on proximity to reference colors
    #         if distances_to_team_1[0] < distances_to_team_1[1]:
    #             self.team_colors[1] = kmeans.cluster_centers_[0]
    #             self.team_colors[2] = kmeans.cluster_centers_[1]
    #         else:
    #             self.team_colors[1] = kmeans.cluster_centers_[1]
    #             self.team_colors[2] = kmeans.cluster_centers_[0]
    
    #         # Store the first 10 cluster centers
    #         if 1 not in self.first_ten_team_colors:
    #             self.first_ten_team_colors[1] = []
    #         if 2 not in self.first_ten_team_colors:
    #             self.first_ten_team_colors[2] = []
    
    #         self.first_ten_team_colors[1].append(self.team_colors[1])
    #         self.first_ten_team_colors[2].append(self.team_colors[2])
            
    #         # Add team color information to the debug image
    #         team1_color = self.team_colors[1].tolist()
    #         team2_color = self.team_colors[2].tolist()
            
    #         cv2.putText(debug_frame, f"Team 1: BGR({int(team1_color[0])},{int(team1_color[1])},{int(team1_color[2])})", 
    #                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team1_color, 2)
    #         cv2.putText(debug_frame, f"Team 2: BGR({int(team2_color[0])},{int(team2_color[1])},{int(team2_color[2])})", 
    #                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team2_color, 2)
                       
    #     elif self.counter == 10:
    #         # After the first 10 frames, calculate the most repeated center within a range
    #         team_1_colors = np.array(self.first_ten_team_colors[1])
    #         team_2_colors = np.array(self.first_ten_team_colors[2])
    
    #         def find_most_repeated_color(colors):
    #             grouped_colors = []
    #             for color in colors:
    #                 found_group = False
    #                 for group in grouped_colors:
    #                     if np.linalg.norm(color - group[0]) <= color_tolerance:
    #                         group.append(color)
    #                         found_group = True
    #                         break
    #                 if not found_group:
    #                     grouped_colors.append([color])
    #             most_repeated_group = max(grouped_colors, key=len)
    #             return np.mean(most_repeated_group, axis=0)
    
    #         self.team_colors[1] = find_most_repeated_color(team_1_colors)
    #         self.team_colors[2] = find_most_repeated_color(team_2_colors)
            
    #         # Add final team color information to the debug image
    #         team1_color = self.team_colors[1].tolist()
    #         team2_color = self.team_colors[2].tolist()
            
    #         # Draw boxes for detected players
    #         for player_id, player_detection in player_detections.items():
    #             bbox = player_detection["bbox"]
    #             x1, y1, x2, y2 = [int(c) for c in bbox]
                
    #             # Get team assignment for this player
    #             player_color = self.get_player_color(frame, bbox)
                
    #             # Calculate distances to team colors
    #             dist_team1 = np.linalg.norm(player_color - self.team_colors[1])
    #             dist_team2 = np.linalg.norm(player_color - self.team_colors[2])
                
    #             # Assign team based on color distance
    #             team_id = 1 if dist_team1 < dist_team2 else 2
    #             team_color = self.team_colors[team_id].tolist()
                
    #             cv2.rectangle(debug_frame, (x1, y1), (x2, y2), team_color, 2)
    #             cv2.putText(debug_frame, f"ID:{player_id} Team:{team_id}", 
    #                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    #         cv2.putText(debug_frame, f"FINAL Team 1: BGR({int(team1_color[0])},{int(team1_color[1])},{int(team1_color[2])})", 
    #                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team1_color, 2)
    #         cv2.putText(debug_frame, f"FINAL Team 2: BGR({int(team2_color[0])},{int(team2_color[1])},{int(team2_color[2])})", 
    #                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team2_color, 2)
    #     else:
    #         # For frames after the 10th, just draw the player bounding boxes with their team colors
    #         for player_id, player_detection in player_detections.items():
    #             bbox = player_detection["bbox"]
    #             x1, y1, x2, y2 = [int(c) for c in bbox]
                
    #             # Get team assignment for this player using get_player_team
    #             team_id = self.get_player_team(frame, bbox, player_id)
    #             team_color = self.team_colors[team_id].tolist()
                
    #             cv2.rectangle(debug_frame, (x1, y1), (x2, y2), team_color, 2)
    #             cv2.putText(debug_frame, f"ID:{player_id} Team:{team_id}", 
    #                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    #     # Save the debug frame
    #     debug_filename = os.path.join(output_dir, f"team_assignment_frame_{self.counter}.jpg")
    #     cv2.imwrite(debug_filename, debug_frame)
            
    #     self.counter += 1
    def get_player_team(self, frame, player_bbox, player_id):
        # ##print("conffffff" , team_confidence , team_confidence and team_confidence > 0.7)
        if (
            player_id in self.player_team_dict
            and self.player_team_confidence[player_id]
            and self.player_team_confidence[player_id] > 0.75
        ):
            # print("dakhal", player_id )
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        # Calculate distances to team colors
        distances = {
            team_id: np.linalg.norm(player_color - team_color)
            for team_id, team_color in self.team_colors.items()
        }
       
        team_id = min(distances, key=distances.get)
        team_id = 1 #OUDA

        # Compute confidence
        dist_values = list(distances.values())
        gap = abs(dist_values[0] - dist_values[1])
        confidence = gap / max(dist_values)
        # print('player_id', player_id,'player_color', player_color)
        # print("distances", distances, "confidence", confidence)
        

        self.player_team_dict[player_id] = team_id
        self.player_team_confidence[player_id] = confidence

        # return team_id
        return 1 #OUDA
