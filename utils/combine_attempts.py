import os
import json
import glob

def combine_json_files():
    # Path to the offline_attempts folder
    base_dir = "cleaned_jsons"
    
    # Find all JSON files in subdirectories
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    # Combined data as array of arrays
    combined_data = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                # Load the JSON content
                file_data = json.load(f)
                # Add the content to our combined data
                combined_data.append(file_data)
                print(f"Processed: {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Save the combined data to a new file
    output_file = "dba.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined {len(combined_data)} JSON files into {output_file}")

if __name__ == "__main__":
    combine_json_files()

