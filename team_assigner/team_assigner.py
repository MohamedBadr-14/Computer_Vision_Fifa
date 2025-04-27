import numpy as np
import time
import torch
from transformers import AutoProcessor
from transformers import SiglipVisionModel
import supervision as sv
from more_itertools import chunked
import joblib
from transformers.models.siglip.modeling_siglip import SiglipVisionModel



# 'google/siglip-base-patch16-224'
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
        self.reducer = joblib.load('../drive/MyDrive/models/reducer_model_gpu.pkl')
        self.kmeans = joblib.load('../drive/MyDrive/models/kmeans_trained_gpu.pkl')
    
    def extract_crops(self, tracks, frame):
        crops = []
        crop_ids = []
        start_crops = time.time()
        for player_id , track in tracks.items():
            # if player_id in self.player_team_dict.keys():
            #     track["team"] = self.player_team_dict[player_id]
            #     continue
            bbox = track["bbox"]
            image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            top_half_image = image[
                int(image.shape[0] * 0.15) : int(image.shape[0] * 0.6), :
            ]
            crops.append(top_half_image)
            crop_ids.append(player_id)
        # print("crops" , len(crops))
        #print("CROPS" , time.time() - start_crops)
        return crops, crop_ids

    def extract_embeddings(self,crops):
        BATCH_SIZE = 32

        crop_pillow = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crop_pillow, BATCH_SIZE)
        data = []
        start_embeddings = time.time()
        with torch.no_grad():
            for batch in batches:
                inputs = self.EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(self.DEVICE)
                embeddings_time = time.time()
                #print("EBEDDINGS_PROCESSOR" , embeddings_time - start_embeddings)
                outputs = self.EMBEDDINGS_MODEL(**inputs)
                outputs_time = time.time()
                #print("OUTPUTS" , outputs_time - embeddings_time)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
                #print("EMBEDDINGS" , time.time() - outputs_time)
        if len(data) == 0:
            return np.array([])
        data = np.concatenate(data)
        # print("data" , data.shape)

        return data

    def predict_kmeans(self , data):
        
        if data.size == 0:
            return np.array([])
        transform_start_time = time.time()
        projections = self.reducer.transform(data)
        transform_end_time = time.time()
        # print("TRANSFORM" , transform_end_time - transform_start_time)
        clusters = self.kmeans.predict(projections)
        # print("CLUSTERS" , time.time() - transform_end_time)

        return clusters

    def assign_clusters_to_players(self, clusters, crop_ids , frame_tracks):
        for i, id in enumerate(crop_ids):
            self.player_team_dict[id] = 1 if clusters[i] == 0 else 2
            frame_tracks[id]["team"] = 1 if clusters[i] == 0 else 2
