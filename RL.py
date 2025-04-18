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
    'bottom_box': 24,
    'middle_box': 25,
    'goaldetected': 26
}

scene={
  "players": {


        "42": {
      "bbox": [712.0142822265625, 574.5263671875, 769.1848754882812, 652.4767456054688],
      "conf": 0.7869588,
      "position": [740, 652],
      "position_transformed": [31.89659881591797, 37.69550132751465],
      "team": 1,
      "team_color": [123.16, 196.4, 188.72]
    },

    "4": {
      "bbox": [712.0142822265625, 574.5263671875, 769.1848754882812, 652.4767456054688],
      "conf": 0.7869588,
      "position": [740, 652],
      "position_transformed": [31.89659881591797, 30.69550132751465],
      "team": 1,
      "team_color": [123.16, 196.4, 188.72]
    },
    "16": {
      "bbox": [972.9927978515625, 450.2742919921875, 1012.3912353515625, 510.6546936035156],
      "conf": 0.77576387,
      "position": [992, 510],
      "position_transformed": [23.38132095336914, 30.34224510192871],
      "team": 2,
      "team_color": [112.48, 44.436, 32.245]
    },
        "162": {
      "bbox": [972.9927978515625, 450.2742919921875, 1012.3912353515625, 510.6546936035156],
      "conf": 0.77576387,
      "position": [992, 510],
      "position_transformed": [20.38132095336914, 30.34224510192871],
      "team": 2,
      "team_color": [112.48, 44.436, 32.245]
    },
    "5": {
      "bbox": [811.2969360351562, 372.0921936035156, 848.4050903320312, 427.7196350097656],
      "conf": 0.712132,
      "position": [829, 427],
      "position_transformed": [21.55144691467285,35.908668518066406],
      "team": 1,
      "team_color": [123.16, 196.4, 188.72],
        "has_ball": True


    },

  },
  "referees": {},
  "ball": {
    "1": {
      "bbox": [741.3411865234375, 657.851318359375, 762.0593872070312, 680.5961303710938],
      "conf": 0.11,
      "position": [751, 669],
      "position_transformed": [32.20328140258789, 31.575361251831055]
    }
  },
  "goalkeeper": {
    "22": {
      "bbox": [1703.3719482421875, 465.69305419921875, 1746.9635009765625, 544.3524780273438],
      "conf": None,
      "position": [1725, 544],
      "position_transformed": [48.55965805053711, 25.108715057373047]
    }
  },
  "goalpost": {
    "1": {
      "bbox": [1794.958740234375, 348.4344177246094, 1824.78955078125, 506.9165954589844],
      "conf": 0.74547535,
      "position": [1809, 427],
      "position_transformed": [51.344871520996094, 16.90476417541504]
    }
  }
}

FIELD_WIDTH = 50 
FIELD_HEIGHT = 90 # haytghyar
GOAL_POSITION = (105, 34)  # lama yerga3
NUM_PLAYERS = 22  # 11 players per team
NUM_ACTIONS = 22 




def count_points_in_rectangle(teammates,opponents, rect_top_left, rect_bottom_right):
    """
    Counts how many points are inside a given rectangle.

    :param points: List of tuples representing points [(x1, y1), (x2, y2), ...].
    :param rect_top_left: Tuple (x, y) representing the top-left corner of the rectangle.
    :param rect_bottom_right: Tuple (x, y) representing the bottom-right corner of the rectangle.
    :return: Integer count of points inside the rectangle.
    """
    count = 0
    for x, y in teammates:
        if rect_top_left[0] <= x <= rect_bottom_right[0] and rect_top_left[1] >= y >= rect_bottom_right[1]:
            count += 1
    for x, y in opponents:
        if rect_top_left[0] <= x <= rect_bottom_right[0] and rect_top_left[1] >= y >= rect_bottom_right[1]:
            count -=1         
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
        closest_opp_dist = 1.0
        closest_opp_angle = 0.0
        no_opponent_in_front = True
        last_opp_x=0
        
        # Initialize 8-direction tracking (will only store opponents)
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dir_data = {d: {'dist': float('inf'), 'pos': None} for d in directions}
        
        # Temporary storage for teammate positions and distances
        team_data = {d: {'dist': float('inf'), 'pos': None} for d in directions}
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
            min_dist = float('inf')
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
            if team_dist != float('inf') and opp_dist != float('inf'):
                dx =mate_x-agent_x
                dy =mate_y-agent_y  
                angle_mate = math.degrees(math.atan2(dy, dx))
                dx =opp_x-agent_x
                dy =opp_y-agent_y  
                angle_opp = math.degrees(math.atan2(dy, dx))
                angle=abs(angle_mate-angle_opp)
                print("direction",direction)
                print("angle",angle)
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
            if team_dist != float('inf')  and (mate_x<last_opp_x or (no_opponent_in_front and mate_x<agent_x ))and angle>10:
                pass_directions[direction] = 1
        
        # Prepare return values (dir_data always contains opponent positions)
        return (
            closest_opp_dist, 
            closest_opp_angle, 
            no_opponent_in_front,
            dir_data['N']['dist'], dir_data['NE']['dist'], dir_data['E']['dist'], dir_data['SE']['dist'],
            dir_data['S']['dist'], dir_data['SW']['dist'], dir_data['W']['dist'], dir_data['NW']['dist'],
            pass_directions['N'], pass_directions['NE'], pass_directions['E'], pass_directions['SE'],
            pass_directions['S'], pass_directions['SW'], pass_directions['W'], pass_directions['NW']
        )
        




    @staticmethod
    def extract_features(agent_x, agent_y,teammates, opponents,attemps, score,goalpost,goalkeeper):
        # Normalized features
        top_rect=[(agent_x,65),(50,51)]
        middle_rect=[(agent_x,51),(50,39)]
        bottom_rect=[(agent_x,39),(50,25)]     
    # lama nozbot el points

        # max_dist = math.sqrt(FIELD_WIDTH**2 + FIELD_HEIGHT**2)
        goaldetected=False

        if goalpost:
            goaldetected=True
            goalpost[1]=FIELD_HEIGHT-goalpost[1]
            dist_goal = SoccerFeatureExtractor.get_distance(agent_x, agent_y, goalpost[0], goalpost[1])
            angle_goal = SoccerFeatureExtractor.get_angle(agent_x, agent_y, goalpost[0], goalpost[1])
    
        closest_opp_dist, closest_opp_angle, no_opponent_in_front, closest_N, closest_NE, closest_E, closest_SE, closest_S, closest_SW, closest_W, closest_NW,pass_N_flag,pass_NE_flag,pass_E_flag,pass_SE_flag,pass_S_flag,pass_SW_flag,pass_W_flag,pass_NW_flag = SoccerFeatureExtractor.get_opponent_and_pass_data(agent_x, agent_y, teammates, opponents)
        top_box=count_points_in_rectangle(teammates,opponents,top_rect[0],top_rect[1])
        bottom_box=count_points_in_rectangle(teammates,opponents,bottom_rect[0],bottom_rect[1])
        middle_box=count_points_in_rectangle(teammates,opponents,middle_rect[0],middle_rect[1])
        return np.array([
            dist_goal,
            angle_goal,
            closest_opp_dist,
            closest_opp_angle,
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
            bottom_box,
            middle_box,
            goaldetected

            
        ], dtype=np.float32)




######################################################################################################
















def plot_pitch(teammates,opponets,me):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, FIELD_WIDTH)
    plt.ylim(0, 90)
    plt.title("Soccer Field")
    plt.xlabel("Width")
    plt.ylabel("Height")
    top_rect=[(me[0],65),(50,51)]
    middle_rect=[(me[0],51),(50,39)]
    bottom_rect=[(me[0],39),(50,25)]     
        
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
    player_pos = None
    attemps=scene['Attemps'] if scene.get('Attemps', False) else 0
    score=scene['Score'] if scene.get('Score', False) else  0
    goalpost=None
    goalkeeper=None
    if scene.get('goalpost', False):
        goalpost = scene['goalpost'][list(scene['goalpost'].keys())[0]]['position_transformed']
    if scene.get('goalkeeper', False):
        first_key = next(iter(scene['goalkeeper']))
        x, y = scene['goalkeeper'][first_key]['position_transformed']  
        goalkeeper=((x, FIELD_HEIGHT-y))
        # print("goalkeeper",scene['goalkeeper'][first_key]['position_transformed'])
    for player_id, player_data in scene['players'].items():
        x, y = player_data['position_transformed']  
        if player_data['team'] == 1:
            if player_data.get('has_ball', False):
                player_pos = (x, FIELD_HEIGHT-y)
            else:
                teammates.append((x, FIELD_HEIGHT-y))
        else:
            opponents.append((x, FIELD_HEIGHT-y))



    plot_pitch(teammates,opponents,player_pos)        

    return SoccerFeatureExtractor.extract_features(
        player_pos[0],player_pos[1],  teammates, opponents,attemps, score,goalpost,goalkeeper
    )

def calculate_reward(prev_state, new_state, action):
    """Simple reward function"""
    reward = 0
    #(todo) law basa mara wahda be3ed fa teb2a fe rakam ahsan
    if prev_state[feature_dict['dist_goal']] != 0 and new_state[feature_dict['dist_goal']] != 0:
        reward += (prev_state[feature_dict['dist_goal']] - new_state[feature_dict['dist_goal']]) * 5 
    
    # Reward for facing goal (angle closer to 0)
    reward += 1 if (abs(prev_state[feature_dict['angle_goal']]) - abs(new_state[feature_dict['angle_goal']]))>0 else 0 
    reward += 10 if new_state[feature_dict['no_opponent_in_front']] else -1
    
    if not new_state[feature_dict['goaldetected']] and not prev_state[feature_dict['goaldetected']]:
        reward +=-10
    elif new_state[feature_dict['goaldetected']] and not prev_state[feature_dict['goaldetected']]:
        reward += -5
    
    # Small penalty for each step
    if prev_state['attemps']==new_state['attemps']:
        reward -= 0.1
    if new_state['score']-prev_state['score']>700:
        reward += 100 # goalll
    elif new_state['score']-prev_state['score']>0:
        reward -= 10    #missed    
    if (new_state['score']==prev_state['score']) and new_state['attemps']!=prev_state['attemps']:
        reward -= 100        # turnover

    
    return reward


#DQn classs

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
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
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
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
    
    def save(self, filename):
        """Save the model and training parameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Load the model and training parameters."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = deque(checkpoint['memory'], maxlen=10000)
        self.epsilon = checkpoint['epsilon']


# def simulate_random_training(agent, steps=10):
#     """Generate random positions and train on them"""
#     print("Training with random position data...")

# # henaaaaaaaaaa
    
#     for step in range(steps):
#         # Generate random positions for all players + ball
#         ## hena negeb el positions el 3adeya

        
#         # Pick a random player to control (first team only)
        
#         # Get initial state
#         state = get_state_from_positions(positions, player_idx)
        
#         # Agent chooses action (1-22)
#         action = agent.act(state)
        
#         # Generate new random positions to simulate action effect
#         new_positions = generate_random_positions()
#         new_state = get_state_from_positions(new_positions, player_idx)
        
#         # Calculate reward
#         reward = calculate_reward(state, new_state, action)
        
#         # Random done flag (10% chance episode ends)
#         done = random.random() < 0.1
        
#         # Remember experience
#         agent.remember(state, action, reward, new_state, done)
        
#         # Train on batch
#         agent.replay(32)
        
#         print(f"Step {step+1}: Action {action}, Reward {reward:.2f}")
    
#     print("Training completed!")

# Initialize agent
# state_size = 7  # hena nehot rakamna
# action_size = NUM_ACTIONS
# agent = DQNAgent(state_size, action_size)

# # Train with random position data
# simulate_random_training(agent, steps=10)
# # save experience replay memory on text file
# with open("experience_replay.txt", "w") as f:
#     for experience in agent.memory:
#         f.write(f"{experience}\n")

# # Save trained model
# agent.save("soccer_dqn_random_train.pth")

values=get_state_from_positions(scene).tolist()
for key, index in feature_dict.items():
        print(f"{key}: {values[index]}")


 