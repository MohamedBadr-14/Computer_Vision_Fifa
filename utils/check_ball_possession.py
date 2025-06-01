import json
import sys

def check_ball_possession(json_file):
    try:
        # Read the file as a string to later calculate line numbers
        with open(json_file, 'r') as f:
            json_str = f.read()
        
        # Load the JSON data
        data = json.loads(json_str)
        
        # List to store positions of frames without ball possession
        frames_without_ball = []  # Will store (outer_index, inner_index, player_groups, line_num)

        # Function to estimate line number based on position in string
        def estimate_line_number(json_string, array_index, frame_index):
            # Find the start of the specified outer array
            outer_array_start = 0
            for i in range(array_index):
                outer_array_start = json_string.find("[", outer_array_start + 1)
                if outer_array_start == -1:
                    return -1
            
            # Find the start of the specified frame within the outer array
            frame_start = outer_array_start
            for i in range(frame_index + 1):
                frame_start = json_string.find("{", frame_start + 1)
                if frame_start == -1:
                    return -1
            
            # Count newlines up to this position to estimate line number
            return json_string[:frame_start].count('\n') + 1

        # Check each outer array in the data
        for outer_index, outer_array in enumerate(data):
            # Check each frame in the inner array
            for inner_index, frame in enumerate(outer_array):
                # Flag to track if at least one player has the "has_ball" key
                has_ball_found = False
                
                # Check if the frame has a "players" key
                if "players" not in frame:
                    line_num = estimate_line_number(json_str, outer_index, inner_index)
                    frames_without_ball.append((outer_index, inner_index, [], line_num))
                    continue
                
                # Check each player in the frame
                for player_id, player_data in frame["players"].items():
                    if "has_ball" in player_data:
                        has_ball_found = True
                        break
                
                # If no player has the "has_ball" key, add the position to frames_without_ball
                if not has_ball_found:
                    # Calculate approximate line number
                    line_num = estimate_line_number(json_str, outer_index, inner_index)
                    player_groups = list(frame["players"].keys())
                    frames_without_ball.append((outer_index, inner_index, player_groups, line_num))
        
        # Print the results
        if frames_without_ball:
            print(f"Found {len(frames_without_ball)} frames where no player has the 'has_ball' key:")
            print("Positions (array_number, frame_number, player_groups, line_number):")
            for i, position in enumerate(frames_without_ball, 1):
                print(f"  {i}. Array {position[0]}, Frame {position[1]}, Player Groups {position[2]}, Line ~{position[3]}")
        else:
            print("All frames have at least one player with the 'has_ball' key.")

        return frames_without_ball
    
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{json_file}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Use command line argument if provided, otherwise default to "db.json"
    json_file = sys.argv[1] if len(sys.argv) > 1 else "dba.json"
    check_ball_possession(json_file)