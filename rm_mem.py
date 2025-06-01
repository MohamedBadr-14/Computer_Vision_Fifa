import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from rl import get_state_from_positions  # Import your state function
from rl import *
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import json

def process_json_objects(json_file_path):
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
            
        # Process each object and collect results
        actions = []
        results = []
        for index, item in enumerate(data):
            try:
                result = get_state_from_positions(item)
                actions.append(item['action'])
                results.append(result)
                print(f"Processed item {index+1}/{len(data)}")
            except Exception as e:
                print(f"Error processing item {index+1}: {e}")
        
        return results, actions
        
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {json_file_path}")
        return [], []
    except Exception as e:
        print(f"Error: {e}")
        return [], []

def process_folder(folder_path):
    """
    Process all JSON files in a folder
    
    Args:
        folder_path: Path to folder containing JSON files
    
    Returns:
        all_results: List of lists of processed data
        all_actions: List of lists of actions
    """
    all_results = []
    all_actions = []
    
    # List all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return all_results, all_actions
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        print(f"Processing {file_path}...")
        
        # Process the file
        results, actions = process_json_objects(file_path)
        
        if results:
            print(f"Successfully processed {len(results)} items from {json_file}")
            all_results.extend(results)
            all_actions.extend(actions)
        else:
            print(f"No results obtained from {json_file}")
    
    return all_results, all_actions
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def predict_action(state, model , scaler,model_path='state_action_model.pth'):
    """
    Load the model and predict an action for the given state
    
    Args:
        state: The state vector to predict from
        model_path: Path to the saved model
        
    Returns:
        action_id: The predicted action ID
        probabilities: The probability distribution over all actions
    """
    # Load the saved model
    # checkpoint = torch.load(model_path ,weights_only=False)
    
    # # Extract model configuration
    # input_size = checkpoint['config']['input_size']
    # hidden_sizes = checkpoint['config']['hidden_sizes']
    # num_classes = checkpoint['config']['num_classes']
    
    # Recreate the model
    # model = SimpleMLP(input_size, hidden_sizes, num_classes)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()  # Set to evaluation mode
    
    # Get the scaler
    # scaler = checkpoint['scaler']
    
    # Ensure state is numpy array and correct shape
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    
    if state.ndim == 1:
        # Single state, reshape to 2D
        state = state.reshape(1, -1)
    
    # Scale the state
    scaled_state = scaler.transform(state)
    
    # Convert to tensor
    state_tensor = torch.tensor(scaled_state, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(state_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get the top action and probability
        top_p, top_class = torch.max(probabilities, dim=1)
        
        # Get second best action
        sorted_prob, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        second_best = sorted_indices[0, 1].item()
        second_prob = sorted_prob[0, 1].item()
        
    # Convert to Python types
    action_id = top_class.item()
    action_prob = top_p.item()
    
    print(f"Predicted action: {action_id} with probability {action_prob:.4f}")
    print(f"Second best action: {second_best} with probability {second_prob:.4f}")
    if action_id not in range(0,7) or action_id != 16:
        if second_best in range(0,7) or second_best == 16:
            action_id = second_best
    
    return action_id, probabilities.numpy()

def main():
       state_file = 'test_rl_jsons'
       states, actions = process_folder(state_file)
       print(states)
    
    
       
       for  x,action in zip(states,actions):         

            # Predict action
            action_id, probs = predict_action(x)
            print(action_id,action)
            
            # Print top 3 actions with probabilities


if __name__ == "__main__":
    main()