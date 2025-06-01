import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def transform_data(input_data):
    scenes_arr = []
    
    for i in range(len(input_data['players'])):
        scene = {
            'players': {},
            'referees': {},
            'ball': {},
            'goalkeeper': {},
            'goalpost': {},
            'action': None
        }
        
        players_frame = input_data['players'][i]
        for player_id, player_data in players_frame.items():
            scene['players'][player_id] = {
                'team': player_data['team'],
                'position_transformed': player_data['position_transformed']
            }
            if 'has_ball' in player_data:
                scene['players'][player_id]['has_ball'] = True
            if 'bbox' in player_data:
                scene['players'][player_id]['bbox'] = tuple(player_data['bbox'])
            if 'position' in player_data:
                scene['players'][player_id]['position'] = tuple(player_data['position'])
        
        if i < len(input_data['ball']):
            ball_frame = input_data['ball'][i]
            for ball_id, ball_data in ball_frame.items():
                scene['ball'][ball_id] = {
                    'conf': ball_data.get('conf', 0.11),
                    'position_transformed': ball_data['position_transformed']
                }
                if 'bbox' in ball_data:
                    scene['ball'][ball_id]['bbox'] = ball_data['bbox']
                if 'position' in ball_data:
                    scene['ball'][ball_id]['position'] = tuple(ball_data['position'])
        
        
        if i < len(input_data['goalpost']):
            goalpost_frame = input_data['goalpost'][i]
            for post_id, post_data in goalpost_frame.items():
                scene['goalpost'][post_id] = {
                    'position_transformed': tuple(post_data['position_transformed']),
                    'position': np.array(post_data['position'], dtype=np.float32)
                }
        
        if i < len(input_data['action']):
            action_frame = input_data['action'][i]
            scene['action'] = action_frame
        scenes_arr.append(scene)
    
    return scenes_arr

with open('all_tracks.json') as f:
    input_data = json.load(f)

scenes_arr = transform_data(input_data)

print(scenes_arr)

with open('transformed_tracks.json', 'w') as f:
    json.dump(scenes_arr, f, indent=4, cls=NumpyEncoder)
    