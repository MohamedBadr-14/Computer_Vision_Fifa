import json
import copy
import os
import glob

def flip_json_data(data):
    """
    Flips the JSON data by:
    1. Transforming y-coordinates: new_y = 90 - old_y
    2. Flipping directional actions (N<->S, NE<->SE, etc.)
    """
    
    # Action mapping for flipping directions
    action_flip_map = {
        0: 4,   # pass_N -> pass_S
        1: 3,   # pass_NE -> pass_SE
        2: 2,   # pass_E -> pass_E (stays same)
        3: 1,   # pass_SE -> pass_NE
        4: 0,   # pass_S -> pass_N
        5: 7,   # pass_SW -> pass_NW
        6: 6,   # pass_W -> pass_W (stays same)
        7: 5,   # pass_NW -> pass_SW
        8: 12,  # dribble_N -> dribble_S
        9: 11,  # dribble_NE -> dribble_SE
        10: 10, # dribble_E -> dribble_E (stays same)
        11: 9,  # dribble_SE -> dribble_NE
        12: 8,  # dribble_S -> dribble_N
        13: 15, # dribble_SW -> dribble_NW
        14: 14, # dribble_W -> dribble_W (stays same)
        15: 13, # dribble_NW -> dribble_SW
        16: 16, # shoot -> shoot (stays same)
    }
    
    # Create a deep copy to avoid modifying the original data
    flipped_data = copy.deepcopy(data)
    
    # Process each frame in the data
    for frame in flipped_data:
        # Flip player positions
        if "players" in frame:
            for player_id, player_info in frame["players"].items():
                if "position_transformed" in player_info:
                    x, y = player_info["position_transformed"]
                    # Flip the y-coordinate
                    player_info["position_transformed"] = [x, 90 - y]
        
        # Flip the action
        if "action" in frame:
            original_action = frame["action"]
            frame["action"] = action_flip_map.get(original_action, original_action)
    
    return flipped_data

def process_all_json_files(base_dir, output_suffix="_flipped"):
    """
    Process all JSON files in the base directory and its subdirectories.
    
    Args:
        base_dir (str): Base directory to search for JSON files
        output_suffix (str): Suffix to add to output filenames (default: "_flipped")
    
    Returns:
        dict: Summary of processed files
    """
    
    # Find all JSON files in subdirectories
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in directory: {base_dir}")
        return {"processed": 0, "errors": 0, "files": []}
    
    processed_files = []
    error_files = []
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            print(f"Processing: {json_file}")
            
            # Read the original JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flip the data
            flipped_data = flip_json_data(data)
            
            # Create output filename
            file_dir = os.path.dirname(json_file)
            file_name = os.path.basename(json_file)
            name_without_ext = os.path.splitext(file_name)[0]
            output_file = os.path.join(file_dir, f"{name_without_ext}{output_suffix}.json")
            
            # Write the flipped data to new file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(flipped_data, f, indent=2)
            
            processed_files.append({
                "input": json_file,
                "output": output_file,
                "frames": len(data) if isinstance(data, list) else 1
            })
            
            print(f"  -> Saved to: {output_file}")
            
        except Exception as e:
            error_msg = f"Error processing {json_file}: {str(e)}"
            print(f"  -> ERROR: {error_msg}")
            error_files.append({"file": json_file, "error": str(e)})
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {len(processed_files)} files")
    print(f"Errors: {len(error_files)} files")
    
    if error_files:
        print("\nFiles with errors:")
        for error in error_files:
            print(f"  - {error['file']}: {error['error']}")
    
    return {
        "processed": len(processed_files),
        "errors": len(error_files),
        "files": processed_files,
        "error_files": error_files
    }

def process_single_json_file(file_path, output_path=None):
    """
    Process a single JSON file.
    
    Args:
        file_path (str): Path to input JSON file
        output_path (str): Path for output file (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the original JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flip the data
        flipped_data = flip_json_data(data)
        
        # Determine output path
        if output_path is None:
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]
            output_path = os.path.join(file_dir, f"{name_without_ext}_flipped.json")
        
        # Write the flipped data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(flipped_data, f, indent=2)
        
        print(f"Successfully processed: {file_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Process all JSON files in the offline_attempts directory
    base_dir = "cleaned_jsons"
    
    # Make sure the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist. Creating example structure...")
        # You can create your directory structure here if needed
    else:
        # Process all JSON files
        summary = process_all_json_files(base_dir)
        
        print(f"\nProcessed {summary['processed']} files successfully!")
        
        # Show details of processed files
        if summary['files']:
            print("\nProcessed files details:")
            for file_info in summary['files']:
                print(f"  {file_info['input']} ({file_info['frames']} frames) -> {file_info['output']}")
    
    # Alternative: Process a single file
    # process_single_json_file("path/to/your/file.json")