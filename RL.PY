import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

scene={
  "players": {
    "1": {
      "bbox": [1050.006103515625, 492.58447265625, 1097.567138671875, 558.914794921875],
      "position": [1073, 558],
      "position_adjusted": [1073, 558],
      "position_transformed": [55.88541793823242, 17.049999237060547],
      "team": 1,
      "team_color": [46.477, 34.997, 28.281]
    },
    "2": {
      "bbox": [618.8065795898438, 443.68743896484375, 670.1077880859375, 504.7702331542969],
      "position": [644, 504],
      "position_adjusted": [644, 504],
      "position_transformed": [33.54166793823242, 15.399999618530273],
      "team": 2,
      "team_color": [80.469, 172.11, 154.09]
    },
    "3": {
      "bbox": [696.1344604492188, 596.143798828125, 758.9864501953125, 678.9885864257812],
      "position": [727, 678],
      "position_adjusted": [727, 678],
      "position_transformed": [37.86458206176758, 20.71666717529297],
      "team": 2,
      "team_color": [80.469, 172.11, 154.09],
      "has_ball": True
    },
    "4": {
      "bbox": [711.8679809570312, 353.0068664550781, 748.5867919921875, 408.91070556640625],
      "position": [730, 408],
      "position_adjusted": [730, 408],
      "position_transformed": [38.02083206176758, 12.466666221618652],
      "team": 2,
      "team_color": [46.477, 34.997, 28.281]
    },
    "5": {
      "bbox": [923.8346557617188, 788.353271484375, 972.9783325195312, 892.04736328125],
      "position": [948, 892],
      "position_adjusted": [948, 892],
      "position_transformed": [49.375, 27.255556106567383],
      "team": 1,
      "team_color": [46.477, 34.997, 28.281]
    },
    "6": {
      "bbox": [837.7654418945312, 441.9215087890625, 877.78662109375, 502.35601806640625],
      "position": [857, 502],
      "position_adjusted": [857, 502],
      "position_transformed": [44.63541793823242, 15.338889122009277],
      "team": 1,
      "team_color": [46.477, 34.997, 28.281]
    }
  },
  "referees": {},
  "ball": {
    "1": {
      "conf": 0.35058585,
      "bbox": [741.3411865234375, 657.851318359375, 762.0593872070312, 680.5961303710938],
      "position": [751, 669],
      "position_adjusted": [751, 669],
      "position_transformed": [39.11458206176758, 20.441667556762695]
    }
  }
}
FIELD_WIDTH = 105
FIELD_HEIGHT = 33
GOAL_POSITION = (105, 34)  # Right side goal
NUM_PLAYERS = 22  # 11 players per team
NUM_ACTIONS = 22  # 8 moves + 8 dribbles + 3 shoots + 3 passes

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
            plt.figure(figsize=(8, 6))
            plt.plot([agent_x, mate_x], [agent_y, mate_y], 'b-', label=f"Distance: {dist:.2f}, Angle: {angle:.2f}°")
            plt.scatter([agent_x, mate_x], [agent_y, mate_y], color=['red', 'green'], s=100, label=['Agent', 'Mate'])
            plt.title(f"Agent to Mate: Distance = {dist:.2f}, Angle = {angle:.2f}°")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Equal scaling for x and y
            plt.show()
    
             
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
            
            print(direction)    
            
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
                if agent_x <= opp_x:
                    all_opp_behind = False
                
                dist = math.hypot(opp_x - agent_x, opp_y - agent_y)
                
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
            plt.figure(figsize=(8, 6))
            plt.plot([closest_opp[0],agent_x ], [ closest_opp[1],agent_y], 'b-', label=f"Distance: {closest_opp_dist:.2f}, Angle: {closest_opp_angle:.2f}°")
            plt.scatter([closest_opp[0],agent_x], [ closest_opp[1],agent_y], color=['red', 'green'], s=100, label=['Agent', 'Mate'])
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Equal scaling for x and y
            plt.show()
            # Classify direction
            print("dx,dy")
            print(agent_x,agent_y)
            print(dx,dy)
            print(closest_opp[0],closest_opp[1])    
           
            print(closest_opp_angle)
            no_opponent_in_front = all_opp_behind
        
        # Determine pass directions by comparing team_data and dir_data

        ### checck here for offside rule ba3dennnnnnnnnn mesh ma3mola
        for direction in directions:
            team_dist = team_data[direction]['dist']
            opp_dist = dir_data[direction]['dist']
            
            # If there's a teammate and either no opponent or teammate is closer
            if team_dist != float('inf') and (opp_dist == float('inf') or team_dist < opp_dist):
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
    def extract_features(agent_x, agent_y,teammates, opponents):
        # Normalized features
        max_dist = math.sqrt(FIELD_WIDTH**2 + FIELD_HEIGHT**2)
        
        # Distance to goal (normalized)
        # dist_goal = SoccerFeatureExtractor.get_distance(agent_x, agent_y, *GOAL_POSITION) / max_dist
        
        # Angle to goal (normalized -1 to 1)
        # angle_goal = SoccerFeatureExtractor.get_angle(agent_x, agent_y, *GOAL_POSITION) / 180.0

        
        # Angle to goal (in degrees) negebhommmmmmmmmmmmmmmm
        dist_goal=0
        angle_goal=0

        # Closest opponent distance and 
        closest_opp_dist, closest_opp_angle, no_opponent_in_front, closest_N, closest_NE, closest_E, closest_SE, closest_S, closest_SW, closest_W, closest_NW,pass_N_flag,pass_NE_flag,pass_E_flag,pass_SE_flag,pass_S_flag,pass_SW_flag,pass_W_flag,pass_NW_flag = SoccerFeatureExtractor.get_opponent_and_pass_data(agent_x, agent_y, teammates, opponents)

      
        # Closest teammate distance and angle (excluding self)

        
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
        ], dtype=np.float32)




######################################################################################################

















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
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))


def get_state_from_positions(scene):
    """Convert raw positions to state for specific player"""
    teammates = []
    opponents = []
    player_pos = None
    for player_id, player_data in scene['players'].items():
        x, y = player_data['position_transformed']  
        if player_data['team'] == 2:
            if player_data.get('has_ball', False):
                player_pos = (x, FIELD_HEIGHT-y)
            else:
                teammates.append((x, FIELD_HEIGHT-y))
        else:
            opponents.append((x, FIELD_HEIGHT-y))

    print(player_pos)
    return SoccerFeatureExtractor.extract_features(
        player_pos[0],player_pos[1],  teammates, opponents
    )

def calculate_reward(prev_state, new_state, action):
    """Simple reward function"""
    reward = 0
    
    # Reward for moving closer to goal
    reward += (prev_state[0] - new_state[0]) * 5  # Distance to goal decreased
    
    # Reward for facing goal (angle closer to 0)
    reward += (abs(prev_state[1]) - abs(new_state[1])) * 2
    
    # Reward for having ball
    if new_state[6] > 0.5:
        reward += 0.2
    
    # Small penalty for each step
    reward -= 0.1
    
    return reward

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

print(get_state_from_positions(scene))
 