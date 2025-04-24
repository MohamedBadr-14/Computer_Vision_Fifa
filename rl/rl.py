import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt



feature_dict = {
    'dist_goal': 0,
    'angle_goal': 1,
    'closest_opp_dist': 2,
    'closest_opp_angle': 3,
    'no_opponent_in_front': 4,
    'closest_N': 5,
    'closest_NE': 6,
    'closest_E': 7,
    'closest_SE': 8,
    'closest_S': 9,
    'closest_SW': 10,
    'closest_W': 11,
    'closest_NW': 12,
    'pass_N_flag': 13,
    'pass_NE_flag': 14,
    'pass_E_flag': 15,
    'pass_SE_flag': 16,
    'pass_S_flag': 17,
    'pass_SW_flag': 18,
    'pass_W_flag': 19,
    'pass_NW_flag': 20,
    'attemps': 21,
    'score': 22,
    'top_box': 23,
    'middle_box': 24,
    'bottom_box': 25,
    'goaldetected': 26,
    'agent_box':27,
    'agent_x': 28,
    'agent_y': 29,
}

FIELD_WIDTH = 60 
FIELD_HEIGHT = 90 # haytghyar
GOAL_POSITION = (105, 34)  # lama yerga3
NUM_ACTIONS = 17 

action_list = {
    0: "pass_N",
    1: "pass_NE",
    2: "pass_E",
    3: "pass_SE",
    4: "pass_S",
    5: "pass_SW",
    6: "pass_W",
    7: "pass_NW",
    8: "dribble_N",
    9: "dribble_NE",
    10: "dribble_E",
    11: "dribble_SE",
    12: "dribble_S",
    13: "dribble_SW",
    14: "dribble_W",
    15: "dribble_NW",
    16: "shoot",
}


boxes = {
    1: "top_box",
    2: "middle_box",
    3: "bottom_box",
}

def count_points_in_rectangle(teammates,opponents, rect_top_left, rect_bottom_right):
    """
    Counts how many points are inside a given rectangle.

    :param points: List of tuples representing points [(x1, y1), (x2, y2), ...].
    :param rect_top_left: Tuple (x, y) representing the top-left corner of the rectangle.
    :param rect_bottom_right: Tuple (x, y) representing the bottom-right corner of the rectangle.
    :return: Integer count of points inside the rectangle.
    """
    counter_all = 0
    count = 0
    for x, y in teammates:
        if rect_top_left[0] <= x <= rect_bottom_right[0] and rect_top_left[1] >= y >= rect_bottom_right[1]:
            count += 1
            counter_all += 1
    for x, y in opponents:
        if rect_top_left[0] <= x <= rect_bottom_right[0] and rect_top_left[1] >= y >= rect_bottom_right[1]:
            count -=1    
            counter_all += 1     
    return count/counter_all if counter_all!=0 else 0

def count_agent_in_rectangle(agent_x, agent_y):
    """
    Counts how many points are inside a given rectangle.

    :param points: List of tuples representing points [(x1, y1), (x2, y2), ...].
    :param rect_top_left: Tuple (x, y) representing the top-left corner of the rectangle.
    :param rect_bottom_right: Tuple (x, y) representing the bottom-right corner of the rectangle.
    :return: Integer count of points inside the rectangle.
    """
    top_rect=[(agent_x,65),(FIELD_WIDTH,51)]
    middle_rect=[(agent_x,51),(FIELD_WIDTH,39)]
    bottom_rect=[(agent_x,39),(FIELD_WIDTH,25)]    
    count = 0
    if top_rect[0][0] <= agent_x <= top_rect[1][0] and top_rect[0][1] >= agent_y >= top_rect[1][1]:
        count = 1
    if middle_rect[0][0] <= agent_x <= middle_rect[1][0] and middle_rect[0][1] >= agent_y >= middle_rect[1][1]:
        count = 2
    if bottom_rect[0][0] <= agent_x <= bottom_rect[1][0] and bottom_rect[0][1] >= agent_y >= bottom_rect[1][1]:
        count = 3
    return count

    
class SoccerFeatureExtractor:
    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    @staticmethod
    def get_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(dy, dx))
    

    ################################################################### features

    @staticmethod


    def get_opponent_and_pass_data(agent_x, agent_y, teammates, opponents):
        
        # Initialize closest opponent variables
        closest_opp_dist = 75
        closest_opp_angle = 0.0
        no_opponent_in_front = True
        last_opp_x=0
        
        # Initialize 8-direction tracking (will only store opponents)
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dir_data = {d: {'dist': 75, 'pos': None} for d in directions}
        
        # Temporary storage for teammate positions and distances
        team_data = {d: {'dist': 75, 'pos': None} for d in directions}
        pass_directions = {d: 0 for d in directions}  # 1 if teammate is closest, 0 otherwise

        # Process teammates first (store temporarily)
        for mate_x, mate_y in teammates:
            dx =mate_x-agent_x
            dy =mate_y-agent_y  
            dist = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            # plt.figure(figsize=(8, 6))
            # plt.plot([agent_x, mate_x], [agent_y, mate_y], 'b-', label=f"Distance: {dist:.2f}, Angle: {angle:.2f}°")
            # plt.scatter([agent_x, mate_x], [agent_y, mate_y], color=['red', 'green'], s=100, label=['Agent', 'Mate'])
            # plt.title(f"Agent to Mate: Distance = {dist:.2f}, Angle = {angle:.2f}°")
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.legend()
            # plt.grid(True)
            # plt.axis('equal')  # Equal scaling for x and y
            # plt.show()
             
            if -22.5 <= angle < 22.5:
                direction = 'E'
            elif 22.5 <= angle < 67.5:
                direction = 'NE'
            elif 67.5 <= angle < 112.5:
                direction = 'N'
            elif 112.5 <= angle < 157.5:
                direction = 'NW'
            elif angle >= 157.5 or angle < -157.5:
                direction = 'W'
            elif -157.5 <= angle < -112.5:
                direction = 'SW'
            elif -112.5 <= angle < -67.5:
                direction = 'S'
            elif -67.5 <= angle < -22.5:
                direction = 'SE'
            
            
            # Update team data if this is the closest teammate in this direction
            if dist < team_data[direction]['dist']:
                team_data[direction] = {'dist': dist, 'pos': (mate_x, mate_y)}

        # Process opponents (store in dir_data)
        if opponents:
            min_dist = 75
            closest_opp = None
            all_opp_behind = True
            
            for opp_x, opp_y in opponents:
                # Check if opponent is in front
                if opp_x>last_opp_x:
                    last_opp_x=opp_x



                if agent_x <= opp_x:
                    all_opp_behind = False
                

                dist = math.hypot(opp_x - agent_x, opp_y - agent_y)
                # print(opp_x, opp_y, agent_x, agent_y, dist)
            
                
                # Track closest opponent overall
                if dist < min_dist:
                    min_dist = dist
                    closest_opp = (opp_x, opp_y)
                
                # Classify direction
                dx = opp_x - agent_x
                dy = opp_y - agent_y
                angle = math.degrees(math.atan2(dy, dx))
                
                if -22.5 <= angle < 22.5:
                    direction = 'E'
                elif 22.5 <= angle < 67.5:
                    direction = 'NE'
                elif 67.5 <= angle < 112.5:
                    direction = 'N'
                elif 112.5 <= angle < 157.5:
                    direction = 'NW'
                elif angle >= 157.5 or angle < -157.5:
                    direction = 'W'
                elif -157.5 <= angle < -112.5:
                    direction = 'SW'
                elif -112.5 <= angle < -67.5:
                    direction = 'S'
                elif -67.5 <= angle < -22.5:
                    direction = 'SE'
                
                # Update opponent data if this is the closest opponent in this direction
                if dist < dir_data[direction]['dist']:
                    dir_data[direction] = {'dist': dist, 'pos': (opp_x, opp_y)}
            
            # Set opponent metrics
            closest_opp_dist = min_dist 
            # closest_opp_angle = math.degrees(math.atan2(
            #     agent_y-closest_opp[1] , agent_x-closest_opp[0])) 
            dx=closest_opp[0]-agent_x  
            dy =closest_opp[1]  -agent_y
            closest_opp_angle = math.degrees(math.atan2(dy, dx))
    
     
            no_opponent_in_front = all_opp_behind
            
        
        # Determine pass directions by comparing team_data and dir_data

        ### checck here for offside rule ba3dennnnnnnnnn mesh ma3mola

        for direction in directions:
            angle=90
            team_dist = team_data[direction]['dist']
            opp_dist = dir_data[direction]['dist']
            opp_x, opp_y = dir_data[direction]['pos'] if dir_data[direction]['pos'] else (0, 0)
            mate_x, mate_y =team_data[direction]['pos'] if team_data[direction]['pos'] else (0, 0)
            if team_dist != 75 and opp_dist != 75:
                dx =mate_x-agent_x
                dy =mate_y-agent_y  
                angle_mate = math.degrees(math.atan2(dy, dx))
                dx =opp_x-agent_x
                dy =opp_y-agent_y  
                angle_opp = math.degrees(math.atan2(dy, dx))
                angle=abs(angle_mate-angle_opp)
   
            # plt.figure(figsize=(8, 6))
            # plt.plot([agent_x, mate_x], [agent_y, mate_y], 'b-', label=f"Distance: {dist:.2f}, Angle: {angle:.2f}°")
            # plt.scatter([agent_x, mate_x], [agent_y, mate_y], color=['red', 'green'], s=100, label=['Agent', 'Mate'])
            # plt.title(f"Agent to Mate: Distance = {dist:.2f}, Angle = {angle:.2f}°")
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.legend()
            # plt.grid(True)
            # plt.axis('equal')  # Equal scaling for x and y
            # plt.show()
            




         
       

            
            # If there's a teammate and either no opponent or teammate is closer
            if team_dist != 75  and (mate_x<last_opp_x or (no_opponent_in_front and mate_x<agent_x ))and angle>10:
                pass_directions[direction] = 1
        
        # Prepare return values (dir_data always contains opponent positions)
        return (
            closest_opp_dist, 
            closest_opp_angle, 
            no_opponent_in_front,
            dir_data['N']['dist']/75, dir_data['NE']['dist']/75, dir_data['E']['dist']/75, dir_data['SE']['dist']/75,
            dir_data['S']['dist']/75, dir_data['SW']['dist']/75, dir_data['W']['dist']/75, dir_data['NW']['dist']/75,
            pass_directions['N'], pass_directions['NE'], pass_directions['E'], pass_directions['SE'],
            pass_directions['S'], pass_directions['SW'], pass_directions['W'], pass_directions['NW']
        )
        




    @staticmethod
    def extract_features(teammates, opponents,attemps, score,goalpost,goalkeeper,agent_x = 0, agent_y = 0):
        # Normalized features
        top_rect=[(agent_x,65),(FIELD_WIDTH,51)]
        middle_rect=[(agent_x,51),(FIELD_WIDTH,39)]
        bottom_rect=[(agent_x,39),(FIELD_WIDTH,25)]     
    # lama nozbot el points

        # max_dist = math.sqrt(FIELD_WIDTH**2 + FIELD_HEIGHT**2)
        goaldetected=False
        dist_goal = 75
        angle_goal =180
        if goalpost:
            goaldetected=True
            goalpost[1]=FIELD_HEIGHT-goalpost[1]
            dist_goal = SoccerFeatureExtractor.get_distance(agent_x, agent_y, goalpost[0], goalpost[1])
            angle_goal = SoccerFeatureExtractor.get_angle(agent_x, agent_y, goalpost[0], goalpost[1])
    
        closest_opp_dist, closest_opp_angle, no_opponent_in_front, closest_N, closest_NE, closest_E, closest_SE, closest_S, closest_SW, closest_W, closest_NW,pass_N_flag,pass_NE_flag,pass_E_flag,pass_SE_flag,pass_S_flag,pass_SW_flag,pass_W_flag,pass_NW_flag = SoccerFeatureExtractor.get_opponent_and_pass_data(agent_x, agent_y, teammates, opponents)
        top_box=count_points_in_rectangle(teammates,opponents,top_rect[0],top_rect[1])
        bottom_box=count_points_in_rectangle(teammates,opponents,bottom_rect[0],bottom_rect[1])
        middle_box=count_points_in_rectangle(teammates,opponents,middle_rect[0],middle_rect[1])
        agent_box=count_agent_in_rectangle(agent_x, agent_y)
        return np.array([
            (dist_goal/75),
            (angle_goal/180),
            (closest_opp_dist/75),
            (closest_opp_angle/180),
            no_opponent_in_front,
            closest_N,
            closest_NE,
            closest_E,
            closest_SE,
            closest_S,
            closest_SW,
            closest_W,
            closest_NW,
            pass_N_flag,
            pass_NE_flag,
            pass_E_flag,
            pass_SE_flag,
            pass_S_flag,
            pass_SW_flag,
            pass_W_flag,
            pass_NW_flag,
            attemps,
            score,
            top_box,
            middle_box,
            bottom_box,
            goaldetected,
            (agent_box/3),
            (agent_x/FIELD_WIDTH),
            (agent_y/FIELD_HEIGHT),
            
        ], dtype=np.float32)




######################################################################################################


def plot_pitch(teammates,opponets,me):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, FIELD_WIDTH)
    plt.ylim(0, 90)
    plt.title("Soccer Field")
    plt.xlabel("Width")
    plt.ylabel("Height")
    top_rect=[(me[0],65),(FIELD_WIDTH,51)]
    middle_rect=[(me[0],51),(FIELD_WIDTH,39)]
    bottom_rect=[(me[0],39),(FIELD_WIDTH,25)]     
        
        # Draw rectangles
    plt.gca().add_patch(plt.Rectangle(
        top_rect[0],  # Top-left corner
        top_rect[1][0] - top_rect[0][0],  # Width (x2 - x1)
        top_rect[1][1] - top_rect[0][1],  # Height (y2 - y1)
        color='red', alpha=0.5, label='Top Rectangle'
    ))
    plt.gca().add_patch(plt.Rectangle(
        middle_rect[0],
        middle_rect[1][0] - middle_rect[0][0],
        middle_rect[1][1] - middle_rect[0][1],
        color='blue', alpha=0.5, label='Middle Rectangle'
    ))
    plt.gca().add_patch(plt.Rectangle(
        bottom_rect[0],
        bottom_rect[1][0] - bottom_rect[0][0],
        bottom_rect[1][1] - bottom_rect[0][1],
        color='green', alpha=0.5, label='Bottom Rectangle'
    ))

    plt.scatter(me[0], me[1], color='yellow', label='Agent')
    for mate_x, mate_y in teammates:
        plt.scatter(mate_x, mate_y, color='yellow', label='Teammate')
    
    # Plot opponents
    for opp_x, opp_y in opponets:
        plt.scatter(opp_x, opp_y, color='blue', label='Opponent')

    # for printing scene
    plt.grid()
    plt.show()


def get_state_from_positions(scene):
    """Convert raw positions to state for specific player"""
    teammates = []
    opponents = []
    player_pos = (0,0)
    attemps=scene['Attemps'] if scene.get('Attemps', False) else 0
    score=scene['Score'] if scene.get('Score', False) else  0
    print("ATTEMPS IN RL ",attemps)
    print("SCORE IN RL ",score)
    print("SCENE IN RL ",scene)
    goalpost=None
    goalkeeper=None
    if scene.get('goalpost', False):
        goalpost = scene['goalpost'][list(scene['goalpost'].keys())[0]]['position_transformed']
    if scene.get('goalkeeper', False):
        first_key = next(iter(scene['goalkeeper']))
        x, y = scene['goalkeeper'][first_key]['position_transformed']  
        goalkeeper=((x, FIELD_HEIGHT-y))
        # print("goalkeeper",scene['goalkeeper'][first_key]['position_transformed'])
    if scene.get('players', False):
        for player_id, player_data in scene['players'].items():
            x, y = player_data['position_transformed']  
            if player_data['team'] == 1:
                if player_data.get('has_ball', False):
                    player_pos = (x, FIELD_HEIGHT-y)
                else:
                    teammates.append((x, FIELD_HEIGHT-y))
            else:
                opponents.append((x, FIELD_HEIGHT-y))



    # plot_pitch(teammates,opponents,player_pos)        

    return SoccerFeatureExtractor.extract_features(
        teammates, opponents,attemps, score,goalpost,goalkeeper,player_pos[0],player_pos[1]
    )

def calculate_reward(prev_state, new_state, action):
    """
    Normalized reward function that keeps values in a reasonable range (-10 to +10)
    """
    reward = 0
    
    # Distance to goal improvement (normalize from -5 to +5)
    if prev_state[feature_dict['dist_goal']] != 0 and new_state[feature_dict['dist_goal']] != 0:
        # Scale based on field size, assuming max dist ~FIELD_WIDTH
        dist_change = 75*(prev_state[feature_dict['dist_goal']] - new_state[feature_dict['dist_goal']])
        normalized_dist_change = max(min(dist_change, 5), -5)  # Clamp between -5 and 5
        reward += normalized_dist_change
    
    # Facing goal (normalized to +1)
    if (abs(prev_state[feature_dict['angle_goal']]) - abs(new_state[feature_dict['angle_goal']])) > 0:
        reward += 1
    else:
        reward -= 0.5    
    
    # Clear path vs opponent in front (normalized to +3/-1)
    reward += 3 if new_state[feature_dict['no_opponent_in_front']] else -0.1
    
    # Goal visibility rewards/penalties (normalized to +2/-2)
    if not new_state[feature_dict['goaldetected']] and not prev_state[feature_dict['goaldetected']]:
        reward += -2  # Still can't see goal
    elif new_state[feature_dict['goaldetected']] and not prev_state[feature_dict['goaldetected']]:
        reward += 2   # Now can see goal
    elif new_state[feature_dict['goaldetected']] and prev_state[feature_dict['goaldetected']]:
        reward += 0.1   # Still can see goal
    elif not new_state[feature_dict['goaldetected']] and prev_state[feature_dict['goaldetected']]:
        reward += -2  # Lost sight of goal
    
    # Time penalty (small negative reward for each step in same attack)
    if prev_state[feature_dict['attemps']] == new_state[feature_dict['attemps']]:
        reward -= 0.2
    
    # Goal scored (maximum reward of +10)
    if new_state[feature_dict['score']] - prev_state[feature_dict['score']] > 500:
        reward += 100  # Goal scored
        return 100
    elif new_state[feature_dict['score']] - prev_state[feature_dict['score']] > 0:
        reward += 3   # Shot on target but missed
    
    # Turnover (normalized to -5)
    if (new_state[feature_dict['score']] == prev_state[feature_dict['score']]) and \
       new_state[feature_dict['attemps']] != prev_state[feature_dict['attemps']]:
        # reward -= 5   # Turnover/lost possession
        if action == 16:
            reward -= 1
            return -2
        else:
            reward -= 10
            return -10
    
    # Pass reward/penalty based on pass direction flags
    # Check if action was a pass (actions 0-7 correspond to pass_N_flag through pass_NW_flag)
    if 0 <= action <= 7:
        # Map action to direction index
        direction_idx = action - 7 + feature_dict['pass_N_flag']  # Calculate the feature index
        
        # Severe penalty for passing when no teammate available
        if prev_state[direction_idx] == 0:
            reward -= 8  # Bad pass decision (normalized from -1000)
        # Reward for successful pass to open teammate
        elif prev_state[direction_idx] == 1:
            reward += 2  # Good pass (normalized from +10)
    
    # Congestion change reward (assess how crowded the area around player has become)
    prev_cong_num = int(prev_state[feature_dict['agent_box']] * 3)
    new_cong_num = int(new_state[feature_dict['agent_box']] * 3)
    print(prev_cong_num)
    print(new_cong_num)

    if new_cong_num == 0:
        reward -= 1
    else:
        if prev_cong_num == 0:
            reward += 1
        else:
            prev_congestion = prev_state[feature_dict[boxes[prev_cong_num]]]
            new_congestion = new_state[feature_dict[boxes[new_cong_num]]]
            if new_congestion > prev_congestion:
                reward += 1
            elif new_congestion < prev_congestion:
                reward -= 1
        
    return reward
#DQn classs

class DQNAgent:
    def __init__(self, load_path=None):
        """
        Initialize DQN Agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Number of possible actions
            load_path (str, optional): Path to load pre-trained model. If None, creates new model.
        """
        self.state_size = len(feature_dict) - 2 
        self.action_size = len(action_list)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0 if load_path is None else 0.01  # Start with low epsilon if loading
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.prev_state = None
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if load_path is not None:
            self.load(load_path)
        else:
            self.update_target_model()
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if np.random.rand() < 0.2:
                return 16  # Shoot action
            else:
                # 80% chance to choose other actions (0-15) uniformly
                return random.randrange(16)  # Actions 0-15 (pass or dribble)
            # return random.randrange(self.action_size)


        # drop score and attemps from the state before passing it to the model
        state = np.delete(state, [feature_dict['score'], feature_dict['attemps']])  
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) <= batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        # drop score and attemps from the state before passing it to the model
        minibatch = [(np.delete(t[0], [feature_dict['score'], feature_dict['attemps']]), t[1], t[2], np.delete(t[3], [feature_dict['score'], feature_dict['attemps']]), t[4]) for t in minibatch]

        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename, log_filename="model_log.txt"):
        """Save the model and training parameters, and log them to a text file."""
        # Save the model and training parameters
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory),
            'epsilon': self.epsilon
        }, filename)

        # Log parameters to a text file
        with open(log_filename, "w") as log_file:
            log_file.write("Model Weights:\n")
            for name, param in self.model.named_parameters():
                log_file.write(f"{name}:\n{param.data}\n\n")

            log_file.write("\nTarget Model Weights:\n")
            for name, param in self.target_model.named_parameters():
                log_file.write(f"{name}:\n{param.data}\n\n")

            log_file.write("\nReplay Memory:\n")
            for i, memory in enumerate(self.memory):
                log_file.write(f"Memory {i}: {memory}\n")

            log_file.write(f"\nEpsilon: {self.epsilon}\n")
    def load(self, filename):
        """Load the model and training parameters."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = deque(checkpoint['memory'], maxlen=10000)
        self.epsilon = checkpoint['epsilon']
    
    def get_action(self, scene):
        """
        Get the best action for a given game scene.
        
        Args:
            scene (dict): Game scene containing player positions, ball possession, etc.
        
        Returns:
            str: The chosen action from the action_list dictionary
        """
        # Convert scene to state
        state = get_state_from_positions(scene)
        
        # Get action using the trained model
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # drop score and attemps from the state before passing it to the model
            state = np.delete(state, [feature_dict['score'], feature_dict['attemps']])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = self.model(state_tensor)
            action_idx = torch.argmax(action_values).item()
        
        # Return the corresponding action string
        return action_list[action_idx]
