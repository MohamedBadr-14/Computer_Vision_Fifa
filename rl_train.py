from rl import *
import json
import os
import numpy as np
import torch
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
        prev_state= None
        for index, item in enumerate(data):
                if index == 0:
                    prev_state = get_state_from_positions(item)
                    continue
                if index== len(data) - 1:
                    results.append([get_state_from_positions(item),item['action'],15, [0 for i in prev_state]])
                    continue
                current_state = get_state_from_positions(item)
                results.append([current_state,item['action'],calculate_reward(prev_state,current_state,item['action']), prev_state])
                prev_state = current_state
                    
                    
                
                
                print(f"Processed item {index+1}/{len(data)}")
        
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



if __name__ == "__main__":
    # Define the folder containing JSON files
    agent=DQNAgent()
    folder_path = 'attempts_jsons'  # Change this to your folder path
        # def remember(self, state, action, reward, next_state, done):


    # Process all JSON files in the folder
    all_processed_data, all_actions = process_folder(folder_path)
    for i,item in enumerate(all_processed_data):
        if item[1] == 16:
        
                agent.remember(item[0],item[1],item[2],item[3],True)
        else:
            
                agent.remember(item[0],item[1],item[2],item[3],False)  
    agent.save(filename='dqn__offline.pth')                  
    
    
   

    