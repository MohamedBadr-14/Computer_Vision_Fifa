from sklearn.cluster import KMeans
import numpy as np
import cv2
import supervision as sv
import torch
from more_itertools import chunked
from transformers import AutoProcessor, SiglipVisionModel
import umap
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError



SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


class TeamAssigner:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_confidence = {}
        self.reducer = joblib.load("models/reducer_model_2.pkl")
        self.kmeans_trained = joblib.load("models/kmeans_trained_2.pkl")
        self.EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.DEVICE)
        self.EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        

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

    def get_player_team(self,frame,player_bbox,player_id):
        # print("conffffff" , team_confidence , team_confidence and team_confidence > 0.7)
        if player_id in self.player_team_dict and self.player_team_confidence[player_id] and self.player_team_confidence[player_id] > 0.7:
            print("dakhal" , player_id)
            return self.player_team_dict[player_id]
        print("not dakhal" , player_id)

        player_color = self.get_player_color(frame,player_bbox)

        
        distances = self.kmeans.transform(player_color.reshape(1, -1))  # Distance to all centroids
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]  # Predicted cluster

        # Normalize confidence score
        max_distance = np.max(distances)
        normalized_confidence = 1 - (distances[0, team_id] / max_distance)
        if player_id == 25:
            print("normalized_confidence" , normalized_confidence)

        team_id+=1

        if player_id == 91:
            team_id=1

        self.player_team_dict[player_id] = team_id
        self.player_team_confidence[player_id] = normalized_confidence

        return team_id
    
    def extract_crops(self,frame, detections):
        PLAYER_ID = 4
        
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops = []
        for xyxy in detections.xyxy:
            full_crop = sv.crop_image(frame, xyxy)
            height = full_crop.shape[0]
            top_half = full_crop[int(height*0.15):int(height*0.6)]  # keep top half
            crops.append(top_half)
        print("crops" , len(crops))

        return crops
    
    def extract_embeddings(self,crops):
        # SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

        # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(DEVICE)
        # EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
        # EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        BATCH_SIZE = 32

        crop_pillow = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crop_pillow, BATCH_SIZE)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(self.DEVICE)
                outputs = self.EMBEDDINGS_MODEL(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        data = np.concatenate(data)
        print("data" , data.shape)

        return data

    def predict_kmeans(self,data , fit_transformed): 

        try:
            check_is_fitted(self.reducer)
            print("fitted")
        except NotFittedError:
            print("The reducer is not fitted yet.")
        # projections = self.reducer.transform(data)
        # print("projections" , projections.shape)
        # clusters = self.kmeans_trained.predict(projections)
        clusters = [0,0,0,0,0,0]
        
        
        return clusters

        