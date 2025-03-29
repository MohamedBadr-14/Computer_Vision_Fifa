import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Constants
FIELD_WIDTH = 105
FIELD_HEIGHT = 68
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

    @staticmethod
    def extract_features(agent_x, agent_y, ball_x, ball_y, teammates, opponents):
        # Normalized features
        max_dist = math.sqrt(FIELD_WIDTH**2 + FIELD_HEIGHT**2)
        
        # Distance to goal (normalized)
        dist_goal = SoccerFeatureExtractor.get_distance(agent_x, agent_y, *GOAL_POSITION) / max_dist
        
        # Angle to goal (normalized -1 to 1)
        angle_goal = SoccerFeatureExtractor.get_angle(agent_x, agent_y, *GOAL_POSITION) / 180.0
        
        # Ball possession (1 if has ball, else 0)
        has_ball = 1.0 if SoccerFeatureExtractor.get_distance(agent_x, agent_y, ball_x, ball_y) < 2.0 else 0.0
        
        # Closest opponent distance and angle
        if opponents:
            closest_opp_dist = min(SoccerFeatureExtractor.get_distance(agent_x, agent_y, x, y) for x, y in opponents) / max_dist
            closest_opp = min(opponents, key=lambda p: SoccerFeatureExtractor.get_distance(agent_x, agent_y, p[0], p[1]))
            closest_opp_angle = SoccerFeatureExtractor.get_angle(agent_x, agent_y, *closest_opp) / 180.0
        else:
            closest_opp_dist = 1.0
            closest_opp_angle = 0.0
        
        # Closest teammate distance and angle (excluding self)
        if len(teammates) > 1:
            teammates_other = [p for p in teammates if not (math.isclose(p[0], agent_x) and math.isclose(p[1], agent_y))]
            closest_team_dist = min(SoccerFeatureExtractor.get_distance(agent_x, agent_y, x, y) for x, y in teammates_other) / max_dist
            closest_team = min(teammates_other, key=lambda p: SoccerFeatureExtractor.get_distance(agent_x, agent_y, p[0], p[1]))
            closest_team_angle = SoccerFeatureExtractor.get_angle(agent_x, agent_y, *closest_team) / 180.0
        else:
            closest_team_dist = 1.0
            closest_team_angle = 0.0
        
        return np.array([
            dist_goal,
            angle_goal,
            closest_opp_dist,
            closest_opp_angle,
            closest_team_dist,
            closest_team_angle,
            has_ball
        ], dtype=np.float32)

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

def generate_random_positions():
    """Generate random positions for all 22 players and ball"""
    positions = []
    for _ in range(NUM_PLAYERS + 1):  # 22 players + ball
        x = random.uniform(0, FIELD_WIDTH)
        y = random.uniform(0, FIELD_HEIGHT)
        positions.append((x, y))
    return positions

def get_state_from_positions(positions, player_idx):
    """Convert raw positions to state for specific player"""
    ball_pos = positions[-1]  # Last position is ball
    player_pos = positions[player_idx]
    
    # Split into teammates and opponents
    teammates = [positions[i] for i in range(NUM_PLAYERS) if i < NUM_PLAYERS//2]  # First 11 are teammates
    opponents = [positions[i] for i in range(NUM_PLAYERS) if i >= NUM_PLAYERS//2]  # Last 11 are opponents
    
    # Remove self from teammates if present
    if player_idx < NUM_PLAYERS//2:
        teammates = [p for i, p in enumerate(teammates) if i != player_idx]
    
    return SoccerFeatureExtractor.extract_features(
        *player_pos, *ball_pos, teammates, opponents
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

def simulate_random_training(agent, steps=10):
    """Generate random positions and train on them"""
    print("Training with random position data...")

# henaaaaaaaaaa
    
    for step in range(steps):
        # Generate random positions for all players + ball
        positions = generate_random_positions()
        
        # Pick a random player to control (first team only)
        player_idx = random.randint(0, NUM_PLAYERS//2 - 1)
        
        # Get initial state
        state = get_state_from_positions(positions, player_idx)
        
        # Agent chooses action (1-22)
        action = agent.act(state)
        
        # Generate new random positions to simulate action effect
        new_positions = generate_random_positions()
        new_state = get_state_from_positions(new_positions, player_idx)
        
        # Calculate reward
        reward = calculate_reward(state, new_state, action)
        
        # Random done flag (10% chance episode ends)
        done = random.random() < 0.1
        
        # Remember experience
        agent.remember(state, action, reward, new_state, done)
        
        # Train on batch
        agent.replay(32)
        
        print(f"Step {step+1}: Action {action}, Reward {reward:.2f}")
    
    print("Training completed!")

# Initialize agent
state_size = 7  # Number of features
action_size = NUM_ACTIONS
agent = DQNAgent(state_size, action_size)

# Train with random position data
simulate_random_training(agent, steps=10)
# save experience replay memory on text file
with open("experience_replay.txt", "w") as f:
    for experience in agent.memory:
        f.write(f"{experience}\n")

# Save trained model
agent.save("soccer_dqn_random_train.pth")