from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.utils.validation import check_is_fitted
import torch
from transformers import AutoProcessor
from transformers import SiglipVisionModel
import supervision as sv
from more_itertools import chunked
import joblib


class TeamAssigner:
    def __init__(self):
        self.player_team_dict = {}
        self.SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(DEVICE)
        self.EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(self.SIGLIP_MODEL_PATH).to(self.DEVICE)
        self.EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(
            self.SIGLIP_MODEL_PATH, 
            use_fast=True  # Enable fast tokenizer if available
        )
        self.reducer = joblib.load('models/reducer_final.pkl')
        self.kmeans = joblib.load('models/kmeans_final.pkl')
    
    def extract_crops(self, tracks, frame):
        crops = []
        crop_ids = []
        for player_id , track in tracks.items():
            if player_id in self.player_team_dict.keys():
                track["team"] = self.player_team_dict[player_id]
                continue
            bbox = track["bbox"]
            image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            top_half_image = image[
                int(image.shape[0] * 0.15) : int(image.shape[0] * 0.6), :
            ]
            crops.append(top_half_image)
            crop_ids.append(player_id)
        print("crops" , len(crops))

        return crops, crop_ids

    def extract_embeddings(self,crops):
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
        if len(data) == 0:
            return np.array([])
        data = np.concatenate(data)
        print("data" , data.shape)

        return data

    def predict_kmeans(self , data):
        
        if data.size == 0:
            return np.array([])
        projections = self.reducer.transform(data)
        clusters = self.kmeans.predict(projections)

        return clusters

    def assign_clusters_to_players(self, clusters, crop_ids , frame_tracks):
        for i, id in enumerate(crop_ids):
            self.player_team_dict[id] = 2 if clusters[i] == 0 else 1
            frame_tracks[id]["team"] = 2 if clusters[i] == 0 else 1
