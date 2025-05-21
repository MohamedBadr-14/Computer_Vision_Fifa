import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance,get_foot_position

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 150
        # self.max_player_ball_distance = 125
    
    def assign_ball_to_player(self,players,ball_bbox,ball):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            foot_position = get_foot_position(player['bbox'])
            

            distance = measure_distance(ball_position, foot_position)


            if distance < miniumum_distance:
                # print(f"Player {player_id} is closer to the ball with distance {distance}")
                # print(f"Ball position: {ball['position_transformed']}")
                # print(f"con: {ball['conf']}")
                if distance < self.max_player_ball_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player