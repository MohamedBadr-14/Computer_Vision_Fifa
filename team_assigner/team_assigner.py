from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_confidence = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=3, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        # player_cluster = 1-non_player_cluster
        # print(player_cluster)
        # Find the cluster with the maximum number of assignees that is not the non_player_cluster
        assignees_count = np.bincount(labels)
        assignees_count[non_player_cluster] = -1  # Exclude the non_player_cluster
        player_cluster = np.argmax(assignees_count)
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        # print("Team 1 color: ",self.team_colors[1])
        # print("Team 2 color: ",self.team_colors[2])

    def get_player_team(self,frame,player_bbox,player_id):
        # print("conffffff" , team_confidence , team_confidence and team_confidence > 0.7)
        if player_id in self.player_team_dict and self.player_team_confidence[player_id] and self.player_team_confidence[player_id] > 0.85:
            # print("dakhal" )
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        
        distances = self.kmeans.transform(player_color.reshape(1, -1))  # Distance to all centroids
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]  # Predicted cluster

        # Normalize confidence score
        max_distance = np.max(distances)
        normalized_confidence = 1 - (distances[0, team_id] / max_distance)
        # team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1
        # if player_id in self.player_team_dict and self.player_team_confidence[player_id] and self.player_team_confidence[player_id] < 0.83:
        #     print("confidence" , normalized_confidence , player_id)

        #get average color of given bounding box using np.mean()
        # image = frame[int(player_bbox[1]):int(player_bbox[3]),int(player_bbox[0]):int(player_bbox[2])]
        # player_color_rgb = np.mean(image, axis=(0,1))
        # player_avg = np.mean(player_color_rgb)
        # if player_id == 12:
        #     print(player_avg)
        # if player_avg < 100:
        #     team_id = 1
        # else:
        #     team_id = 2



        if player_id == 91:
            team_id=1

        self.player_team_dict[player_id] = team_id
        self.player_team_confidence[player_id] = normalized_confidence

        return team_id
