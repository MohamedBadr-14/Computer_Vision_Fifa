import json
import os
import glob
import shutil

def remove_keys_from_json(data, keys_to_remove):
    """
    Remove specified keys from JSON data recursively and round position_transformed values.
    
    Args:
        data: JSON data (dict, list, or primitive type)
        keys_to_remove: Set of keys to remove
    
    Returns:
        Cleaned JSON data with specified keys removed and position_transformed values rounded
    """
    if isinstance(data, dict):
        # Create new dict without the unwanted keys
        cleaned_dict = {}
        for key, value in data.items():
            if key not in keys_to_remove:
                # Special handling for position_transformed to round values
                if key == 'position_transformed' and isinstance(value, list) and len(value) == 2:
                    cleaned_dict[key] = [round(float(value[0]), 2), round(float(value[1]), 2)]
                else:
                    # Recursively clean the value
                    cleaned_dict[key] = remove_keys_from_json(value, keys_to_remove)
        return cleaned_dict
    
    elif isinstance(data, list):
        # Process each item in the list
        return [remove_keys_from_json(item, keys_to_remove) for item in data]
    
    else:
        # Return primitive values as-is
        return data

def clean_json_file(input_file, output_file=None, keys_to_remove=None):
    """
    Clean JSON file by removing specified keys.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional, defaults to input_file with '_cleaned' suffix)
        keys_to_remove: List of keys to remove (optional, uses default set if not provided)
    """
    # Default keys to remove
    if keys_to_remove is None:
        keys_to_remove = {'bbox', 'position', 'referees', 'ball', 'goalkeeper', 'goalpost'}
    else:
        keys_to_remove = set(keys_to_remove)
    
    # Read the JSON file
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{input_file}': {e}")
        return
    
    # Clean the data
    cleaned_data = remove_keys_from_json(data, keys_to_remove)
    
    # Determine output file name
    if output_file is None:
        if input_file.endswith('.json'):
            output_file = input_file.replace('.json', '_cleaned.json')
        else:
            output_file = input_file + '_cleaned.json'
    
    # Write the cleaned data
    try:
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Cleaned JSON saved to: {output_file}")
    except Exception as e:
        print(f"Error writing to file '{output_file}': {e}")

def clean_json_string(json_string, keys_to_remove=None):
    """
    Clean JSON string by removing specified keys.
    
    Args:
        json_string: JSON data as string
        keys_to_remove: List of keys to remove (optional, uses default set if not provided)
    
    Returns:
        Cleaned JSON string
    """
    # Default keys to remove
    if keys_to_remove is None:
        keys_to_remove = {'bbox', 'position', 'referees', 'ball', 'goalkeeper', 'goalpost'}
    else:
        keys_to_remove = set(keys_to_remove)
    
    try:
        # Parse JSON string
        data = json.loads(json_string)
        
        # Clean the data
        cleaned_data = remove_keys_from_json(data, keys_to_remove)
        
        # Return cleaned JSON string
        return json.dumps(cleaned_data, indent=2)
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON string: {e}")
        return None

def batch_clean_json_files(base_dir="offline_attempts", output_dir="cleaned_jsons", keys_to_remove=None):
    """
    Find all JSON files in subdirectories and clean them, saving to a new directory structure.
    
    Args:
        base_dir: Base directory to search for JSON files
        output_dir: Output directory to save cleaned JSON files
        keys_to_remove: List of keys to remove (optional, uses default set if not provided)
    """
    # Default keys to remove
    if keys_to_remove is None:
        keys_to_remove = {'bbox', 'position', 'referees', 'ball', 'goalkeeper', 'goalpost'}
    else:
        keys_to_remove = set(keys_to_remove)
    
    # Find all JSON files in subdirectories
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in '{base_dir}' and its subdirectories.")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    processed_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # Calculate relative path from base_dir
            relative_path = os.path.relpath(json_file, base_dir)
            
            # Create corresponding output path
            output_file = os.path.join(output_dir, relative_path)
            
            # Create subdirectories in output if they don't exist
            output_subdir = os.path.dirname(output_file)
            if output_subdir and not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Read and clean the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Clean the data
            cleaned_data = remove_keys_from_json(data, keys_to_remove)
            
            # Write cleaned data to output file
            with open(output_file, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            
            print(f"✓ Processed: {relative_path}")
            processed_count += 1
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON Error in {json_file}: {e}")
            error_count += 1
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")
            error_count += 1
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files")
    print(f"Cleaned files saved to: {output_dir}")

def clean_directory_structure(base_dir="offline_attempts", output_dir="cleaned_jsons", keys_to_remove=None):
    """
    Alternative function name for batch cleaning with directory structure preservation.
    """
    return batch_clean_json_files(base_dir, output_dir, keys_to_remove)
    # Example 1: Clean a JSON file
    # clean_json_file('input.json', 'output.json')
    
    # Example 2: Clean with custom keys to remove
    # clean_json_file('input.json', keys_to_remove=['bbox', 'position', 'custom_key'])
    

if __name__ == "__main__":
    batch_clean_json_files()